"""实机运动排障：独立接收 SocketCAN，并保存 SDK 逐帧发送结果。

此模块不发送任何 CAN 指令，不修改接口、速度或队列长度。独立 socket 能
看到同机发送帧的 local echo；echo 不是机械臂对整条目标的接收/执行 ACK。
"""

from __future__ import annotations

import json
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Thread
import time


class ArmCanTrace:
    def __init__(self, channel, directory):
        import can

        self.channel = channel
        self.path = Path(directory) / "arm-can.jsonl"
        self._pending = SimpleQueue()
        self._stop = Event()
        # 与 SDK 分开接收，不能在 SDK 的接收 socket 上抢走反馈。仅保留 J
        # 模式/四组目标和关节/状态反馈，避免全总线高频数据淹没排障信息。
        ids = (0x151, 0x152, 0x153, 0x154, 0x155, 0x156, 0x157, 0x170,
               0x2A1, 0x2A5, 0x2A6, 0x2A7, 0x2A9)
        self._bus = can.Bus(interface="socketcan", channel=channel, ignore_config=True,
                            can_filters=[{"can_id": i, "can_mask": 0x7FF,
                                          "extended": False} for i in ids])
        self._thread = Thread(target=self._record, name="cloth-can-trace", daemon=True)
        self._thread.start()

    def sent(self, **values):
        # SDK 发送路径只放入内存队列，不打印、不落盘、不等待被动接收线程。
        self._pending.put({"event": "sdk_send", **values})

    def _interface(self):
        root = Path("/sys/class/net") / self.channel
        # 这是接口累计值；必须比较本轮前后差值，不能把历史 dropped 归到本轮。
        return {name: int((root / relative).read_text().strip()) for name, relative in {
            "tx_queue_len": "tx_queue_len", "tx_packets": "statistics/tx_packets",
            "tx_dropped": "statistics/tx_dropped", "tx_errors": "statistics/tx_errors",
            "rx_dropped": "statistics/rx_dropped", "rx_errors": "statistics/rx_errors",
        }.items()}

    def _record(self):
        try:
            with self.path.open("x", encoding="utf-8", buffering=1) as stream:
                def write(event, **values):
                    stream.write(json.dumps({"description": "机械臂 CAN 排障 Arm CAN trace",
                                             "timestamp_monotonic": time.monotonic(),
                                             "timestamp_unix": time.time(),
                                             "event": event, **values}, ensure_ascii=False) + "\n")

                write("started", channel=self.channel, interface=self._interface(),
                      note="local echo != robot command ACK")
                next_stats = time.monotonic() + 1
                while not self._stop.is_set():
                    message = self._bus.recv(.01)
                    if message is not None:
                        # timestamp 为内核接收时间；日志写入的主机时间另存，避免
                        # 线程调度延迟被误认为总线收发延迟。local_tx 表示同机来源。
                        write("frame", frame_timestamp_unix=message.timestamp,
                              source="bus_rx" if message.is_rx else "local_tx",
                              can_id=f"0x{message.arbitration_id:03X}",
                              data=bytes(message.data).hex(), error_frame=message.is_error_frame)
                    self._drain(write)
                    if time.monotonic() >= next_stats:
                        write("interface", interface=self._interface())
                        next_stats = time.monotonic() + 1
                self._drain(write)
                write("closed", interface=self._interface())
        except Exception as exc:
            # 抓包失败必须可见，否则缺帧会被误判为目标没有发送。只报告诊断
            # 故障，不从日志线程发运动、重试、复位或改变人工保持路径。
            print(json.dumps({"description": "CAN 抓包失败 CAN trace failed",
                              "reason": str(exc), "path": str(self.path)}, ensure_ascii=False), flush=True)

    def _drain(self, write):
        while True:
            try:
                value = self._pending.get_nowait()
            except Empty:
                return
            event = value.pop("event")
            write(event, **value)

    def close(self):
        self._stop.set()
        self._thread.join(2)
        self._bus.shutdown()
