"""直接通过持久 worker 执行一次固定上升 cell，不启动 Codex 或 MCP。

使用与 agent 相同的规划、运动执行、反馈监督和人工 hold 路径。
从当前真实 TCP 沿 task +Z 上升 4 cm，保持当前方向；不回初始位姿、
不操作手指。默认仅上升一次；显式 --repeat N 时每次成功上升后返回起点，
在同一持久 worker/控制进程中重复同一目标。失败立即结束，不自动重试。
只有现场监护下才运行本脚本。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import yaml

# 与现有 scripts/nero 脚本一致，允许从项目根目录直接 python 执行。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cloth_agent.config import load_config
from cloth_agent.paths import DEFAULT_CONFIG, configured_output_dir
from cloth_agent.session import TaskSession


LIFT_CELL = """# 只执行一个固定 cell，不需要语言模型生成代码。
# 保持真实起始 TCP 的方向，避免把姿态对齐混入垂直上升排障。
before = robo.state()
start_tcp = np.array(before['tcp'], dtype=float)
target_tcp = start_tcp.copy()
target_tcp[2, 3] += 0.04
print('上升前真实反馈 Before lift:', before)
print('固定目标 TCP，沿 task +Z 0.04 m Target:', target_tcp.tolist())
lift = robo.move_L(target_tcp)
print('完整运动反馈 Lift MotionResult:', lift)
after = robo.state()
print('上升后真实反馈 After lift:', after)
print('实测 TCP 位移 Measured displacement m:',
      (np.array(after['tcp'])[:3, 3] - start_tcp[:3, 3]).tolist())
if lift.status != 'completed' or not lift.reached:
    raise RuntimeError('固定上升失败，停止且不重试 Lift failed: ' + str(lift.reason))
"""


# 后续轮次不重新捕获起点/朝向，避免反馈量化误差累积成缓慢漂移。
# 返回也使用正式 move_L 执行器；它本身同样可能触发待排查的分段故障。
REPEAT_LIFT_CELL = """print('重复固定上升 Repeat fixed lift; target:', target_tcp.tolist())
lift = robo.move_L(target_tcp)
print('完整运动反馈 Lift MotionResult:', lift)
print('上升后真实反馈 After lift:', robo.state())
if lift.status != 'completed' or not lift.reached:
    raise RuntimeError('固定上升失败，停止且不重试 Lift failed: ' + str(lift.reason))
"""

RETURN_CELL = """print('返回固定起点 Return to fixed start; target:', start_tcp.tolist())
returned = robo.move_L(start_tcp)
print('完整返回反馈 Return MotionResult:', returned)
print('返回后真实反馈 After return:', robo.state())
if returned.status != 'completed' or not returned.reached:
    raise RuntimeError('返回失败，停止且不重试 Return failed: ' + str(returned.reason))
"""


def main():
    parser = argparse.ArgumentParser(description="直接执行固定上升 cell Direct worker lift check")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", help="省略时沿用配置中的 session.output_dir；不覆盖旧记录")
    parser.add_argument("--repeat", type=int, default=1,
                        help="多轮时每次上升成功后返回同一起点；失败停止，不重复失败动作")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat 必须为正整数 Repeat count must be positive")
    directory = Path(args.output_dir or configured_output_dir(args.config)).resolve()
    config = load_config(args.config)
    # load_config 已把标定文件等引用解析为绝对路径，并计算完整工具/坐标变换。
    # 写临时配置只为跳过宿主准备 PTP，不修改日常 agent 的 startup 配置。
    # 本轮 robot_config.json 仍会保存实际使用的冻结配置，便于事后 review。
    config["startup"]["enabled"] = False
    config["session"]["output_dir"] = str(directory)
    with tempfile.TemporaryDirectory(prefix="cloth-lift-") as temporary:
        config_path = Path(temporary) / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
        with TaskSession(directory, config_path=config_path, allow_hardware=True,
                         cell_timeout_s=90, task_timeout_s=max(120, args.repeat * 180),
                         max_cells=1 if args.repeat == 1 else args.repeat * 2) as session:
            print(json.dumps({"description": "直接上升测试 Direct lift cell; no Codex/MCP/startup PTP",
                              "session_file": str(directory / "session.json")}, ensure_ascii=False), flush=True)
            checks = []
            for round_index in range(1, args.repeat + 1):
                phases = [("lift", LIFT_CELL if round_index == 1 else REPEAT_LIFT_CELL)]
                if args.repeat > 1:
                    phases.append(("return", RETURN_CELL))
                for phase, source in phases:
                    print(f"轮次与阶段 Round/phase: {round_index}/{args.repeat} {phase}", flush=True)
                    result = session.execute(source)
                    # 每段原始结果仍由 TaskSession 保存；这里只补轮次/阶段索引。
                    # 不因失败继续返回或重新开始；现场可以用独立 hold 接口接管。
                    checks.append({"round": round_index, "phase": phase,
                                   "cell_id": result.get("cell_id"), "status": result["status"]})
                    print(result.get("stdout", ""), flush=True)
                    if result.get("exception"):
                        print(result["exception"], flush=True)
                    if result["status"] != "completed" or session.status != "running":
                        break
                if result["status"] != "completed" or session.status != "running":
                    break
            (directory / "lift_checks.json").write_text(
                json.dumps({"requested_rounds": args.repeat, "checks": checks},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            if session.status == "running":
                session.finish("success" if result["status"] == "completed" else "failure",
                               "固定上升 cell 已返回 Direct lift cell returned; see measured feedback")
            print(json.dumps(session.summary(), ensure_ascii=False, indent=2), flush=True)
            return 0 if session.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
