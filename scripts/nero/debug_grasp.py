"""简单的 Nero/L20 抓取调试：CLI 单键增量 move_L、拍照、开合手及 TCP 投影。

复用 TaskSession/robo，保留启动回位；不启动 Codex 或 MCP 服务。
平移按 task 系虚拟 TCP 的实测位置增量移动，yaw 按 base 系法兰角度增量旋转。
roll/pitch 固定取遥操作配置的 wrist_only_roll_pitch_deg。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import termios
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PRESET_FILE = ROOT / "configs/grasp_debug_preset.json"

from cloth_agent.config import load_config
from cloth_agent.geometry import tcp_to_flange, transform
from cloth_agent.paths import DEFAULT_CONFIG
from cloth_agent.session import TaskSession
from teleop.keyboard import Keyboard
from teleop.hand_replay import HandReplay


HELP = """单键操作 Keys（无需回车，方向为 task 坐标轴，不是图像方向）：
  W / S     X+ / X-       A / D     Y+ / Y-
  R / F     Z+ / Z-       J / L     法兰 yaw- / yaw+
  - / +     减小/增大平移步长：1、2、5、10、20 mm（= 也可增大）
  [ / ]     减小/增大 yaw 步长：1、2、5、10 °
  P         拍照并记录实际 TCP
  C         拍照 → 闭合拇指/食指并等待到位（有限重试）→ 拍照
  O         张开拇指/食指
  V         拇指/食指到 replay_clip_file 对应 CSV 第一帧（不回放）
  B / G     保存当前实际 TCP 到预设文件 / move_L 回预设（含朝向，跨次运行保留）
  I         显示实际位姿       H    帮助       Q    结束并生成 TCP 叠加图
每次运动到位后再按下一键；运动中人工 hold 仍使用启动时打印的独立命令。
"""


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def draw_references(directory, photos, references):
    """把每次闭合前的固定 TCP 投到每张原图；保留原图，另存 annotated。

    T_task_camera 的逆把 task 系坐标转换到彩色相机光学系。使用照片自身
    保存的输出内参（含裁剪缩放与畸变），不能直接用相机原生分辨率内参。
    三轴长 3 cm，X 红/Y 绿/Z 蓝；这是几何重投影，不处理遮挡。
    """
    if not photos:
        return
    output = directory / "annotated"
    output.mkdir(exist_ok=True)
    for photo in photos:
        metadata = json.loads((directory / "observations" / f"{photo['id']}.json").read_text())
        image = cv2.imread(metadata["_rgb_path"])
        if image is None:
            raise OSError(f"读取照片失败 Image unavailable: {metadata['_rgb_path']}")
        intr = metadata["intrinsics"]
        # 当前 L515 彩色相机使用 Brown-Conrady，与 OpenCV 的五参数模型一致。
        # 不把其他畸变模型静默当作这一模型，以免生成位置错误的调试图。
        if intr["model"].split(".")[-1] not in {"brown_conrady", "none"}:
            raise ValueError(f"不支持的投影畸变模型 Unsupported distortion: {intr['model']}")
        camera_matrix = np.array([[intr["fx"], 0, intr["ppx"]],
                                  [0, intr["fy"], intr["ppy"]], [0, 0, 1]], dtype=float)
        distortion = np.asarray(intr["coeffs"], dtype=float)
        camera_task = np.linalg.inv(transform(metadata["task_from_camera"]))
        for index, reference in enumerate(references, 1):
            camera_tcp = camera_task @ transform(reference["task_from_tcp"])
            # 四个点分别是 TCP 原点和沿其局部 XYZ 方向的轴端点。
            points = np.vstack([np.zeros(3), np.eye(3) * .03])
            points = points @ camera_tcp[:3, :3].T + camera_tcp[:3, 3]
            label = f"G{index}"
            if np.any(points[:, 2] <= 0):
                cv2.putText(image, label + " behind camera", (8, 20 * index),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                continue
            pixels = cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                                       camera_matrix, distortion)[0].reshape(-1, 2)
            height, width = image.shape[:2]
            if not (0 <= pixels[0, 0] < width and 0 <= pixels[0, 1] < height):
                cv2.putText(image, label + " outside image", (8, 20 * index),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
                continue
            pixels = np.rint(pixels).astype(int)
            origin = tuple(pixels[0])
            cv2.circle(image, origin, 4, (0, 255, 255), -1)
            cv2.putText(image, label, (origin[0] + 5, origin[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
            for endpoint, color in zip(pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
                cv2.arrowedLine(image, origin, tuple(endpoint), color, 2, tipLength=.2)
        target = output / f"{photo['number']:03d}_{photo['label']}.png"
        if not cv2.imwrite(str(target), image):
            raise OSError(f"叠加图保存失败 Cannot save overlay: {target}")
    print(f"TCP 叠加图已保存 Overlays: {output}")


class Console:
    """只把终端命令翻译成已有 robo 调用，执行/保持逻辑仍由现有控制进程负责。"""

    def __init__(self, session, roll_pitch_deg, photos, references):
        self.session = session
        self.directory = session.directory
        self.settings = session.robot_config["arms"]["right"]
        self.roll_pitch_deg = roll_pitch_deg
        self.photos, self.references = photos, references
        flange = self.flange(self.state()["tcp"])
        # 初始 yaw 来自准备完成后的实际法兰，不假定绝对 yaw=0，也不采用
        # agent top_down_pose 的相对 yaw 定义。按键增量作用在法兰 yaw 上，
        # 再把得到的绝对角传给内部 move()。
        self.yaw_deg = float(Rotation.from_matrix(flange[:3, :3]).as_euler("xyz", degrees=True)[2])
        self.position_steps_mm = (1, 2, 5, 10, 20, 50, 100)
        self.yaw_steps_deg = (1, 2, 5, 10)
        self.position_step_index, self.yaw_step_index = 2, 1
        # 固定文件独立于每轮带时间戳的记录目录，重启后仍能复用唯一预设。
        # 保存完整 task 系 TCP 位置和朝向，而不是仅保存终端打印的 XYZ。
        self.preset_tcp = None
        if PRESET_FILE.exists():
            self.preset_tcp = transform(json.loads(PRESET_FILE.read_text(encoding="utf-8"))["task_from_tcp"])
            print(f"预设已加载 Preset loaded: {PRESET_FILE}; TCP xyz={self.preset_tcp[:3, 3]}", flush=True)

    def state(self):
        return self.session.control.request("status")

    def flange(self, tcp):
        return tcp_to_flange(tcp, self.settings["task_from_base"], self.settings["flange_from_tcp"])

    def execute(self, source):
        result = self.session.execute(source)
        if result.get("stdout"):
            print(result["stdout"], end="", flush=True)
        if result["status"] != "completed":
            raise RuntimeError(result.get("exception") or result.get("reason") or str(result))
        return result

    def capture(self, label):
        # 只在同步运动已收尾时拍照。等过相机最大缓存年龄及一个控制周期，
        # 避免 close 后仍取得 close 前的缓存帧；沿用 observe 的图像/状态记录。
        # 两者是静止条件下的邻近采样，不声称相机与 CAN 有硬件同步。
        delay = float(self.session.robot_config["camera"]["max_age_s"]) + .1
        result = self.execute(f"time.sleep({delay!r})\nrobo.show(robo.observe())")
        item = result["images"][0]
        metadata = json.loads((self.directory / "observations" / f"{item['observation_id']}.json").read_text())
        photo = {"number": len(self.photos) + 1, "label": label, "id": item["observation_id"],
                 "task_from_tcp": metadata["state"]["tcp"]}
        self.photos.append(photo)
        self.save_index()
        print(f"照片与 TCP 已保存 Photo {photo['number']}: {item['rgb_path']}")
        return photo

    def save_index(self):
        write_json(self.directory / "photos.json", {"photos": self.photos, "pre_close_tcp": self.references})

    def move(self, xyz, yaw_deg):
        # wrist-only 锁定的是 base 系法兰 roll/pitch，不能直接把这两个数
        # 当成虚拟 TCP 的欧拉角。先构造法兰朝向，再乘工具旋转得到 TCP 朝向。
        flange_rotation = Rotation.from_euler("xyz", [*self.roll_pitch_deg, yaw_deg], degrees=True).as_matrix()
        target = np.eye(4)
        target[:3, :3] = (transform(self.settings["task_from_base"])[:3, :3]
                          @ flange_rotation @ transform(self.settings["flange_from_tcp"])[:3, :3])
        target[:3, 3] = xyz
        self.move_tcp(target)

    def move_tcp(self, target):
        """增量运动与回预设共用同一 move_L 到位检查；预设保留保存时的完整朝向。"""
        self.execute(f"motion = robo.move_L({target.tolist()!r})\n"
                     "print('移动结果 Motion:', motion)\n"
                     "if motion.status != 'completed' or not motion.reached:\n"
                     "    raise RuntimeError('移动未到位 Motion failed: ' + repr(motion))")
        # 回预设后让后续平移沿用预设的 yaw，避免下一步又转回此前的朝向。
        self.yaw_deg = float(Rotation.from_matrix(self.flange(target)[:3, :3]).as_euler("xyz", degrees=True)[2])

    def show_steps(self):
        print(f"步长 Step: {self.position_steps_mm[self.position_step_index]} mm / "
              f"{self.yaw_steps_deg[self.yaw_step_index]} deg", flush=True)

    def run(self):
        print(HELP)
        self.show_steps()
        # 复用 Linux 终端 cbreak 读取：不需要 Enter，退出/异常会恢复终端。
        # 仍使用同步 move_L，不增加后台任务队列或第二套执行器。
        directions = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1),
                      "r": (2, 1), "f": (2, -1)}
        with Keyboard() as keyboard:
            while self.session.status == "running":
                key = keyboard.read()
                if key is None:
                    time.sleep(.05)
                    continue
                if key == "q":
                    self.session.finish("success", "交互调试结束 Console complete; grasp success not evaluated")
                    return
                if key in {"-", "+", "="}:
                    delta = -1 if key == "-" else 1
                    self.position_step_index = int(np.clip(self.position_step_index + delta,
                                                          0, len(self.position_steps_mm) - 1))
                    self.show_steps()
                elif key in {"[", "]"}:
                    delta = -1 if key == "[" else 1
                    self.yaw_step_index = int(np.clip(self.yaw_step_index + delta,
                                                     0, len(self.yaw_steps_deg) - 1))
                    self.show_steps()
                elif key in directions or key in {"j", "l"}:
                    # 每一步以最新实测 TCP 为起点，不把尚未到达的旧目标累加。
                    tcp = np.asarray(self.state()["tcp"], dtype=float)
                    xyz, yaw = tcp[:3, 3].copy(), self.yaw_deg
                    if key in directions:
                        axis, sign = directions[key]
                        xyz[axis] += sign * self.position_steps_mm[self.position_step_index] / 1000
                    else:
                        actual_yaw = Rotation.from_matrix(self.flange(tcp)[:3, :3]).as_euler("xyz", degrees=True)[2]
                        yaw = float(actual_yaw + (1 if key == "l" else -1)
                                    * self.yaw_steps_deg[self.yaw_step_index])
                    print(f"增量运动 Jog: TCP={xyz.tolist()}, flange yaw={yaw:.3f} deg", flush=True)
                    self.move(xyz, yaw)
                elif key == "p":
                    self.capture("photo")
                elif key == "c":
                    number = len(self.references) + 1
                    before = self.capture(f"G{number}_before")
                    self.references.append({"label": f"G{number}", "photo_id": before["id"],
                                            "task_from_tcp": before["task_from_tcp"]})
                    self.save_index()
                    # 与 hardware.py 相同的阻塞到位入口，仅操作拇指/食指；
                    # 到位后才拍后照，超时抛出异常，不再用固定 1 s 代替检查。
                    self.execute("print('闭合 Close:', robo.close_gripper())")
                    self.capture(f"G{number}_after")
                elif key == "o":
                    self.execute("print('张开 Open:', robo.open_gripper())")
                elif key == "v":
                    self.execute("print('CSV 第一帧 Replay first frame:', robo.gripper_replay_start())")
                elif key == "b":
                    # 保存实测位姿，不保存上一条运动目标；位置和朝向一起记录。
                    tcp = transform(self.state()["tcp"])
                    write_json(PRESET_FILE, {"task_from_tcp": tcp.tolist()})
                    self.preset_tcp = tcp  # 文件写入成功后才更新内存并提示保存成功。
                    self.session.log("preset_saved", "已保存唯一预设 TCP preset saved",
                                     task_from_tcp=self.preset_tcp.tolist(), file=str(PRESET_FILE))
                    print(f"预设已写入 Preset saved: {PRESET_FILE}; TCP xyz={self.preset_tcp[:3, 3]}", flush=True)
                elif key == "g":
                    if self.preset_tcp is None:
                        print("尚未保存预设，先按 B Save a preset with B first", flush=True)
                    else:
                        print("返回预设 Moving to preset with move_L", flush=True)
                        self.move_tcp(self.preset_tcp)
                elif key == "i":
                    tcp = self.state()["tcp"]
                    angles = Rotation.from_matrix(self.flange(tcp)[:3, :3]).as_euler("xyz", degrees=True)
                    print("实际 task TCP xyz / base flange RPY °:", np.asarray(tcp)[:3, 3], angles)
                    self.show_steps()
                elif key == "h":
                    print(HELP)
                    self.show_steps()
                # 清掉本次同步操作期间的按键积压，避免动作结束后补执行旧按键。
                # Ctrl+C 仍由终端生成信号，沿用现有 session 中断/保持收尾。
                if key in directions or key in {"j", "l", "p", "c", "o", "v", "g"}:
                    termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
                    print("就绪 Ready", flush=True)
        raise RuntimeError(self.session.reason)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="复用 agent 的标定、设备、夹爪及运动配置")
    parser.add_argument("--teleop-config", default=str(ROOT / "configs/pico4_nero_l20.yaml"),
                        help="读取 wrist_only_roll_pitch_deg 与 hand.replay_clip_file，不连接 VR")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/grasp_debug" / time.strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--execute", action="store_true", help="允许启动回位与交互实机操作")
    args = parser.parse_args()
    if not args.execute:
        parser.error("实机调试需 --execute；没有 EXECUTE 交互确认 Hardware requires --execute")
    if not sys.stdin.isatty():
        parser.error("单键控制需要交互终端 Run in an interactive terminal")

    config = load_config(args.config)
    teleop = yaml.safe_load(Path(args.teleop_config).read_text(encoding="utf-8"))
    roll_pitch = np.asarray(teleop["arm"]["wrist_only_roll_pitch_deg"], dtype=float)
    if roll_pitch.shape != (2,) or not np.isfinite(roll_pitch).all():
        parser.error("wrist_only_roll_pitch_deg 必须包含两个有限角度")
    # 与 teleop.config 的路径约定一致：CSV 相对路径以项目根目录为基准。
    # 复用 HandReplay 的列名、数值校验及 20 槽位转换，只取第一帧写入本轮
    # 配置；O/C 仍分别使用原来的 open/closed，启动时不自动发送首帧。
    replay_file = Path(teleop["hand"]["replay_clip_file"]).expanduser()
    if not replay_file.is_absolute():
        replay_file = ROOT / replay_file
    config["arms"]["right"]["hand_poses"]["replay_start"] = HandReplay(replay_file).actions[0].tolist()
    if not config["camera"]["enabled"] or config["observations"]["task_from_camera"] is None:
        parser.error("需要启用相机并提供相机/TCP 标定 Camera and calibration are required")
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    # load_config 已将标定路径、工作区等解析为绝对路径/数值，另存本次快照
    # 后交给 TaskSession；不改用户配置，也不复写 agent 当轮目录。
    config["startup"]["enabled"] = True
    config["session"] = {"output_dir": str(directory)}
    config_file = directory / "debug_config.yaml"
    config_file.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    photos, references = [], []
    print(f"将执行初始回位 Startup enabled; output: {directory}", flush=True)
    print(f"人工保持入口 Hold: python -m cloth_agent hold {directory / 'session.json'}", flush=True)
    try:
        with TaskSession(directory, config_path=config_file, allow_hardware=True,
                         cell_timeout_s=120, task_timeout_s=86400, max_cells=10000) as session:
            if session.status != "running":
                raise RuntimeError(session.reason)
            Console(session, roll_pitch.tolist(), photos, references).run()
    except (KeyboardInterrupt, EOFError):
        print("交互已结束 Console ended; existing session cleanup handles hold", flush=True)
    finally:
        draw_references(directory, photos, references)


if __name__ == "__main__":
    main()
