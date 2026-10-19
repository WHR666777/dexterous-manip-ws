"""重播视频 3～4 秒的拇指/食指动作，其余三指先设为官方张开姿态。"""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robot_control.l20 import L20_JOINT_NAMES, LinkerHandL20, _L20_OPEN_PRESET

DATASET = ROOT / "data/lerobot/nero_l20_act"
INDICES = [0, 5, 10, 15, 1, 6, 16]  # 拇指四通道、食指三通道


def extract():
    fps = json.loads((DATASET / "meta/info.json").read_text())["fps"]
    first, end = round(3 * fps), round(4 * fps)
    with np.load(DATASET / "diagnostics/episode-000000.npz") as data:
        if len(data["hand_action_raw"]) < end:
            raise ValueError("录制不足视频时间 4 秒。")
        actions = data["hand_action_raw"][first:end].copy()
        feedback = data["hand_position_raw"][first:end].copy()
    if (actions.shape != (end - first, 20) or not np.isfinite(actions).all()
            or np.any(actions != np.floor(actions))
            or np.any((actions < 0) | (actions > 255))):
        raise ValueError("手部目标必须是 20 槽、0～255 的整数。")

    output = DATASET / "thumb_index_3_4s.csv"
    names = [L20_JOINT_NAMES[i] for i in INDICES]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "video_time_s"]
                        + [f"action_{name}" for name in names]
                        + [f"feedback_{name}" for name in names])
        for frame, action, actual in zip(range(first, end), actions, feedback):
            writer.writerow([frame, f"{frame / fps:.3f}"]
                            + action[INDICES].tolist() + actual[INDICES].tolist())
    print(f"已导出视频 3≤t<4 秒：帧 {first}～{end - 1}，{len(actions)} 帧，{fps} FPS。")
    print(f"CSV: {output}")
    return actions.astype(int), fps


def replay(actions, fps, config):
    hand = LinkerHandL20(hand_type=config["type"], can_channel=config["can_channel"])
    try:
        hand.connect()
        hand.set_speed(config["speed"])
        # 先初始化其余三指；重播过程中只向拇指、食指发送命令。
        driver = hand._require_connected().hand
        opened = driver.cmd_range_to_joint_range(list(_L20_OPEN_PRESET))
        driver.set_middle_positions(opened[2])
        driver.set_ring_positions(opened[3])
        driver.set_little_positions(opened[4])
        print("其余三指已发送官方张开目标，等待 1 秒后重播拇指、食指。")
        # time.sleep(1.0)
        started = time.monotonic()
        for frame, action in enumerate(actions):
            time.sleep(max(0.0, started + frame / fps - time.monotonic()))
            fingers = driver.cmd_range_to_joint_range(action.tolist())
            driver.set_thumb_positions(fingers[0])
            driver.set_index_positions(fingers[1])
        time.sleep(max(0.0, started + len(actions) / fps - time.monotonic()))
        print("重播完成，保持最后姿态。")
    finally:
        hand.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="实际重播；默认只导出 CSV")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pico4_nero_l20.yaml")
    args = parser.parse_args()
    actions, fps = extract()
    if args.execute:
        config = yaml.safe_load(args.config.read_text())["hand"]
        recorded_side = json.loads((DATASET / "meta/recording.json").read_text())["hand_side"]
        if config["type"] != recorded_side:
            raise ValueError("配置手侧与录制手侧不一致。")
        replay(actions, fps, config)


if __name__ == "__main__":
    main()
