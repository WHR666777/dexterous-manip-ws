# %% 直接执行的 ACT 检查 cell；不启动 TaskSession、Codex、MCP 或任务启动流程。
"""回记录关节、设置训练起始手型，实际执行 20 步双臂 ACT。

nero 环境运行：python scripts/act_start_cell.py --execute
模型由现有 ActPolicyClient 在 act 环境加载。Ctrl+C 请求普通位置保持。
"""

from pathlib import Path
import argparse
import json
from queue import Empty, SimpleQueue
import sys
from threading import Event, Thread
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cloth_agent.act import TRAINING_START_HANDS
from cloth_agent.bimanual import RobotController
from cloth_agent.config import load_config
from cloth_agent.robo import Robo


class LocalConnection:
    """直接调用当前控制对象，让同一个 robo API 在本 cell 内运行，无 RPC 服务。"""

    def __init__(self, controller):
        self.controller = controller

    def send(self, request):
        self.request = request

    def recv(self):
        try:
            return {"ok": True, "result": self.controller.dispatch(self.request, owner=False)}
        except Exception as exc:
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}


def wait_quiescent(robot, timeout=3.):
    deadline = time.monotonic() + timeout
    while True:
        state = robot.snapshot()
        if state["fault"] or state["manual_hold_latched"]:
            raise RuntimeError(f"控制异常 Control fault: {state['fault']}")
        if state["quiescent"]:
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError("位置保持未确认 Hold not confirmed")
        time.sleep(.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="连接硬件并执行本 cell")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if not args.execute:
        parser.error("实际执行需添加 --execute")
    if args.steps < 1:
        parser.error("--steps 必须为正整数")
    config = load_config(ROOT / "configs/cloth_agent.yaml")
    directory = ROOT / "data/agent_runs" / ("act-cell-" + time.strftime("%Y%m%d-%H%M%S"))
    directory.mkdir(parents=True, exist_ok=False)
    print(f"DIRECT ACT CELL: {directory}", flush=True)
    (directory / "cell.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    (directory / "robot_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
    stop, failed, shutdown = Event(), Event(), Event()
    robot = None
    thread = None
    samples = SimpleQueue()
    result = {"training_episode": 36, "training_frame": 0, "steps_requested": args.steps,
              "start_hand_targets": TRAINING_START_HANDS}
    try:
        robot = RobotController(config, directory, stop, failed, True)
        robot.on_control_sample = samples.put

        def supervise():
            while not shutdown.is_set():
                started = time.monotonic()
                try:
                    robot.tick(robot.period)
                except Exception as exc:
                    robot.fail(f"独立 cell 控制故障 Direct cell fault: {exc}")
                shutdown.wait(max(0., robot.period - (time.monotonic() - started)))

        thread = Thread(target=supervise, name="direct-act-supervision", daemon=True)
        thread.start()
        robot.dispatch({"operation": "begin_cell", "cell_id": 1}, owner=True)
        robo = Robo(LocalConnection(robot), config)
        robo._begin_cell(1)
        result["home"] = {}
        for side in ("left", "right"):
            home = robo.move_J(config["arms"][side]["start_joint_positions"], arm=side)
            result["home"][side] = vars(home)
            print(f"{side} 回位: {home}", flush=True)
            if not home.reached:
                raise RuntimeError(f"{side} 回位失败 Home failed: {home.reason}")
        wait_quiescent(robot)
        result["hands"] = robo.prepare_act_hands()
        print("起始手型: " + json.dumps(result["hands"], ensure_ascii=False), flush=True)
        before = robo.observe()
        result["before_rgb"] = before._rgb_path
        result["before_state"] = before.state
        print(f"执行 ACT {args.steps} 步", flush=True)
        result["act"] = robo.run_act(policy="fold", steps=args.steps)
        print("ACT RESULT: " + json.dumps(result["act"], ensure_ascii=False), flush=True)
        after = robo.observe()
        result["after_rgb"] = after._rgb_path
        result["success"] = result["act"]["status"] == "completed"
    except BaseException as exc:
        result.update(success=False, error=f"{type(exc).__name__}: {exc}")
        print(result["error"], flush=True)
    finally:
        if robot is not None:
            try:
                robot.pause("独立 ACT cell 结束 Direct ACT cell ended")
                wait_quiescent(robot)
                result["final_state"] = robot.snapshot()
            except Exception as exc:
                result.update(success=False, hold_error=str(exc))
            robot.act.close()
            shutdown.set()
            if thread is not None:
                thread.join(2)
            robot.close()
        with (directory / "control-samples.jsonl").open("w") as stream:
            while True:
                try:
                    stream.write(json.dumps(samples.get_nowait(), ensure_ascii=False) + "\n")
                except Empty:
                    break
        (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"记录已保存: {directory}", flush=True)
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
