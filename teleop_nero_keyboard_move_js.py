"""键盘点动 NERO 末端：Base 法兰目标 -> 局部IK -> move_js持续跟随。

运行：
    python3 teleop_nero_keyboard_move_js.py --channel can0

键盘：
    W/S : X +/-
    A/D : Y +/-
    R/F : Z +/-

    I/K : 绕 Base X +/-
    J/L : 绕 Base Y +/-
    U/O : 绕 Base Z +/-

    P      : 打印反馈
    Q      : 正常退出，不急停
    Space  : 电子急停并退出
    Esc    : 电子急停并退出
    Ctrl+C : 电子急停并退出

默认：
    平移步长 2 mm
    旋转步长 1 deg
    move_js 控制频率 50 Hz

说明：
1. 每次键盘点动均以当前真实 法兰 为基准，不累计未执行到位的虚拟目标。
2. command_pose() 只更新最后一个合法关节目标；
   真正的 move_js 由 control_tick() 固定频率持续发送。
3. tracker 后续也可以直接调用 command_pose(T_target) 更新目标，
   控制循环仍然固定频率调用 control_tick()。
"""

import argparse
import os
import select
import sys
import termios
import time
import tty

import numpy as np
from scipy.spatial.transform import Rotation

from config import NERO_CAN_CHANNEL, NERO_CAN_INTERFACE
from robot_control import NeroArm
from teleop_nero_tracker import pose_to_matrix, step_is_valid
from teleop_nero_tracker_move_js import (
    NeroIK,
    MAX_JOINT_STEP,
    pose_matches, pose_diagnostic, pose_delta, feedback_timing,
)


# ============================================================
# SE(3) 检查
# ============================================================

def checked_pose(value):
    """严格检查 4x4 SE(3) 齐次矩阵。"""
    pose = np.asarray(value, dtype=float)

    valid = (
        pose.shape == (4, 4)
        and np.isfinite(pose).all()
        and np.allclose(
            pose[3],
            [0.0, 0.0, 0.0, 1.0],
            atol=1e-6,
            rtol=0,
        )
        and np.allclose(
            pose[:3, :3].T @ pose[:3, :3],
            np.eye(3),
            atol=1e-6,
            rtol=0,
        )
        and np.isclose(
            np.linalg.det(pose[:3, :3]),
            1.0,
            atol=1e-6,
            rtol=0,
        )
    )

    if not valid:
        raise ValueError("目标必须是合法、有限的 4x4 SE(3) 齐次矩阵。")

    return pose.copy()


# ============================================================
# 末端控制器
# ============================================================

class EndEffectorController:
    """NERO 法兰绝对位姿 -> IK -> move_js。"""

    def __init__(self, arm, ik=None):
        self.arm = arm
        self.ik = NeroIK() if ik is None else ik

        self.stopped = False
        self.emergency = False
        self.reason = ""
        self.last_sent = False
        self.diagnostic = {}
        self.debug_ik = False

        # 当前最后一条合法关节目标。
        # control_tick() 会持续发送它。
        self.command_joints = None

        # 当前最后一条接受的 法兰 / flange 目标。
        self.target_flange = None

        # ----------------------------------------------------
        # 建立模型/实机一致性
        # ----------------------------------------------------

        joints = self.arm.get_joint_positions()

        flange = checked_pose(
            pose_to_matrix(self.arm.get_flange_pose())
        )

        fk_flange = self.ik.forward(joints)

        if not pose_matches(
            fk_flange,
            flange,
            pos_tol=0.01,
            rot_tol=np.deg2rad(5),
        ):
            raise RuntimeError(
                "URDF FK 与机械臂 flange 反馈不一致；"
                "请检查 URDF、关节零位、关节顺序和坐标系。"
            )

        print('初始 FK 对实际法兰误差：', pose_delta(flange, fk_flange))
        self.target_flange = flange.copy()

        # 初始时不主动 move_js。
        # 等第一次合法键盘/Tracker目标到来后才开始持续发送。

    # --------------------------------------------------------
    # 读取真实状态
    # --------------------------------------------------------

    def current_flange(self):
        """读取当前真实 flange 4x4。"""
        return checked_pose(
            pose_to_matrix(self.arm.get_flange_pose())
        )

    # --------------------------------------------------------
    # 更新绝对 法兰 目标
    # --------------------------------------------------------

    def command_pose(self, target_flange):
        """更新 Base 系下绝对 法兰 目标。

        成功：
            返回 True，并更新 command_joints。

        输入无效 / 越步 / IK无解：
            返回 False，不改变原来的目标。

        注意：
            这里不直接持续发送 move_js。
            真正发送由 control_tick() 完成。
        """

        if self.stopped:
            raise RuntimeError("控制器已停止。")

        self.reason = ""
        self.diagnostic = {'IK成功': None, '新目标已接受': False, '本目标已发送': False}

        if target_flange is None:
            self.reason = "无有效目标"
            return False

        target_flange = checked_pose(target_flange)


        # ----------------------------------------------------
        # 和真实当前位置比较，而不是和上一条命令比较
        # ----------------------------------------------------

        actual_flange = self.current_flange()
        self.diagnostic.update({'实际法兰': pose_diagnostic(actual_flange),
                                '请求绝对法兰': pose_diagnostic(target_flange),
                                '目标相对实际': pose_delta(target_flange, actual_flange)})
        self.diagnostic['反馈时间'] = feedback_timing(self.arm)

        if not step_is_valid(target_flange, actual_flange):
            self.reason = "Flange 相对真实位置变化超过 1 cm / 5°"
            return False

        if not self.arm.is_ok():
            raise RuntimeError("机械臂状态异常。")

        if not self.arm.is_enabled():
            raise RuntimeError("机械臂未使能。")

        # ----------------------------------------------------
        # 当前真实关节作为 IK seed
        # ----------------------------------------------------

        current_q = self.arm.get_joint_positions()
        self.diagnostic['当前关节 deg'] = np.rad2deg(current_q).tolist()
        # 默认保持注入求解器兼容；显式 debug 仅对实际 NeroIK 开启。
        if isinstance(self.ik, NeroIK):
            self.ik.debug = self.debug_ik

        target_q = self.ik.solve(
            target_flange,
            current_q,
        )

        self.diagnostic['IK成功'] = target_q is not None
        if target_q is None:
            self.reason = "局部 IK 无解：" + getattr(self.ik, 'last_report', {}).get('reason', '求解器未提供原因')
            return False

        target_q = np.asarray(target_q, dtype=float)
        self.diagnostic['目标关节 deg'] = np.rad2deg(target_q).tolist()
        self.diagnostic['目标减实际 deg'] = np.rad2deg(target_q - current_q).tolist()

        # ----------------------------------------------------
        # 相对实际关节不能突然超过 5°
        # ----------------------------------------------------

        delta = np.abs(target_q - current_q)

        if np.any(delta > MAX_JOINT_STEP):
            worst = int(np.argmax(delta))

            self.reason = (
                f"J{worst + 1} 相对当前关节变化 "
                f"{np.rad2deg(delta[worst]):.2f}°，超过 5°"
            )
            return False

        try:
            self.arm.validate_joint_command(target_q)

        except ValueError as exc:
            self.reason = str(exc)
            return False

        # ----------------------------------------------------
        # 到这里才更新目标
        # ----------------------------------------------------

        self.command_joints = target_q.copy()
        self.diagnostic['新目标已接受'] = True

        self.target_flange = target_flange.copy()

        return True

    # --------------------------------------------------------
    # 固定频率控制
    # --------------------------------------------------------

    def control_tick(self):
        """发送最后一个合法关节目标。

        应由外部固定频率调用，例如 50 Hz。
        """

        self.last_sent = False
        if self.stopped:
            return

        # 尚未收到第一条运动目标。
        if self.command_joints is None:
            return

        if not self.arm.is_connected():
            raise RuntimeError("机械臂连接已断开。")

        if not self.arm.is_ok():
            raise RuntimeError("机械臂状态异常。")

        if not self.arm.is_enabled():
            raise RuntimeError("机械臂失能。")

        # ----------------------------------------------------
        # 当前反馈异常或与目标相差过大则拒绝继续发
        # ----------------------------------------------------

        current_q = self.arm.get_joint_positions()

        delta = np.abs(
            self.command_joints - current_q
        )

        if np.any(delta > MAX_JOINT_STEP):
            worst = int(np.argmax(delta))

            raise RuntimeError(
                f"实时控制中 J{worst + 1} 目标与实际相差 "
                f"{np.rad2deg(delta[worst]):.2f}°，超过 5°。"
            )

        # 真正的 JS 指令
        self.arm.move_js(
            self.command_joints.tolist()
        )
        self.last_sent = True

    # --------------------------------------------------------
    # 正常退出
    # --------------------------------------------------------

    def stop(self):
        """停止继续发送命令，不触发电子急停。"""
        self.stopped = True
        self.command_joints = None

    # --------------------------------------------------------
    # 急停
    # --------------------------------------------------------

    def emergency_stop(self):
        """电子急停并锁住控制器。"""

        self.stopped = True
        self.emergency = True
        self.command_joints = None

        if self.arm.is_connected():
            self.arm.emergency_stop()


# ============================================================
# 键盘增量
# ============================================================

def keyboard_target(
    key,
    current_flange,
    translation_step,
    rotation_step,
):
    """根据当前真实 法兰 产生下一点动目标。

    平移：
        Base坐标系增量。

    旋转：
        左乘，因此绕 Base 固定 XYZ 轴。
    """

    translations = {
        "w": (0, +1),
        "s": (0, -1),
        "a": (1, +1),
        "d": (1, -1),
        "r": (2, +1),
        "f": (2, -1),
    }

    rotations = {
        "i": (0, +1),
        "k": (0, -1),
        "j": (1, +1),
        "l": (1, -1),
        "u": (2, +1),
        "o": (2, -1),
    }

    key = key.lower()

    target = current_flange.copy()

    # --------------------------------------------------------
    # Base XYZ 平移
    # --------------------------------------------------------

    if key in translations:
        axis, sign = translations[key]

        target[axis, 3] += (
            sign * translation_step
        )

        return target

    # --------------------------------------------------------
    # 绕 Base XYZ 旋转
    # --------------------------------------------------------

    if key in rotations:
        axis, sign = rotations[key]

        rotvec = np.zeros(3)
        rotvec[axis] = (
            sign * rotation_step
        )

        delta_R = Rotation.from_rotvec(
            rotvec
        ).as_matrix()

        # 左乘：
        # 绕 Base 固定坐标轴旋转
        target[:3, :3] = (
            delta_R @ target[:3, :3]
        )

        return target

    return None


# ============================================================
# Linux终端键盘
# ============================================================

class TerminalKeyboard:
    """Linux 非阻塞字符键盘。"""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError(
                "请在交互式 Linux 终端运行。"
            )

        self.fd = sys.stdin.fileno()

        self.original = termios.tcgetattr(
            self.fd
        )

        tty.setcbreak(self.fd)

        termios.tcflush(
            self.fd,
            termios.TCIFLUSH,
        )

        return self

    def read(self):
        """返回：

        None      没有按键
        'quit'    Q
        'estop'   Space / Esc / Ctrl+C
        其他字符   普通控制键
        """

        if not select.select(
            [self.fd],
            [],
            [],
            0,
        )[0]:
            return None

        data = os.read(
            self.fd,
            4096,
        ).decode(
            "ascii",
            errors="ignore",
        ).lower()

        if not data:
            return None

        # 急停优先
        if (
            " " in data
            or "\x1b" in data
            or "\x03" in data
        ):
            return "estop"

        if "q" in data:
            return "quit"

        # 多个重复字符只保留最后一个
        return data[-1]

    def __exit__(self, *exc):
        termios.tcsetattr(
            self.fd,
            termios.TCSADRAIN,
            self.original,
        )


# ============================================================
# 参数
# ============================================================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--channel",
        default=NERO_CAN_CHANNEL,
    )

    parser.add_argument(
        "--interface",
        default=NERO_CAN_INTERFACE,
    )

    parser.add_argument(
        "--hz",
        type=float,
        default=50,
        help="move_js 控制频率，默认 50 Hz，最大 50 Hz",
    )

    parser.add_argument(
        "--step-mm",
        type=float,
        default=2,
        help="键盘平移步长，默认 2 mm",
    )

    parser.add_argument(
        "--step-deg",
        type=float,
        default=1,
        help="键盘旋转步长，默认 1°",
    )

    parser.add_argument('--debug-ik', action='store_true', help='成功逆解也输出详细诊断')
    args = parser.parse_args(argv)

    checks = [
        ("hz", args.hz, 50),
        ("step_mm", args.step_mm, 10),
        ("step_deg", args.step_deg, 5),
    ]

    for name, value, maximum in checks:
        if (
            not np.isfinite(value)
            or value <= 0
            or value > maximum
        ):
            parser.error(
                f"{name} 必须在 (0, {maximum}] 内。"
            )

    return args


# ============================================================
# 主程序
# ============================================================

def main(argv=None):
    args = parse_args(argv)

    # 提前加载 URDF / IK
    ik = NeroIK()

    arm = None
    controller = None

    emergency = False

    try:
        with TerminalKeyboard() as keyboard:

            # ------------------------------------------------
            # 创建机械臂
            # ------------------------------------------------

            arm = NeroArm(
                can_interface=args.interface,
                can_channel=args.channel,
                max_joint_delta=MAX_JOINT_STEP,
            )

            arm.connect()

            # ------------------------------------------------
            # 等待反馈完整
            # ------------------------------------------------

            deadline = (
                time.monotonic() + 5.0
            )

            while True:
                try:
                    arm.get_joint_positions()
                    arm.get_flange_pose()
                    break

                except RuntimeError:
                    if (
                        time.monotonic()
                        >= deadline
                    ):
                        raise

                    time.sleep(0.05)

            # ------------------------------------------------
            # 先使能，再建立当前法兰锚点
            # ------------------------------------------------

            arm.enable()

            time.sleep(0.1)

            controller = EndEffectorController(
                arm,
                ik,
            )
            controller.debug_ik = args.debug_ik

            print(
                "已连接并对齐当前 法兰。"
            )

            print(
                "W/S X，A/D Y，R/F Z；"
                "I/K J/L U/O 绕 Base X/Y/Z。"
            )

            print(
                f"步长 {args.step_mm:g} mm / "
                f"{args.step_deg:g}°；"
                f"move_js = {args.hz:g} Hz。"
            )

            print(
                "P：反馈；"
                "Q：正常退出；"
                "Space/Esc/Ctrl+C：电子急停。"
            )

            # 丢弃启动期间的旧按键
            termios.tcflush(
                keyboard.fd,
                termios.TCIFLUSH,
            )

            # ------------------------------------------------
            # 固定频率控制循环
            # ------------------------------------------------

            period = 1.0 / args.hz

            next_tick = time.monotonic()

            while True:

                # --------------------------------------------
                # 键盘输入
                # --------------------------------------------

                key = keyboard.read()
                requested = False

                if key == "quit":
                    print("\n正常退出。")
                    controller.stop()
                    break

                if key == "estop":
                    print("\n电子急停。")
                    emergency = True
                    controller.emergency_stop()
                    break

                if key == "p":
                    print(
                        "法兰 [m, rad]:",
                        np.round(
                            arm.get_flange_pose(),
                            5,
                        ),
                    )

                    print(
                        "关节 [rad]:",
                        np.round(
                            arm.get_joint_positions(),
                            5,
                        ),
                    )

                elif key is not None:

                    # ----------------------------------------
                    # 关键：
                    # 每次都从真实 法兰 出发，而不是旧目标
                    # ----------------------------------------

                    actual_flange = (
                        controller.current_flange()
                    )

                    target = keyboard_target(
                        key,
                        actual_flange,
                        args.step_mm / 1000.0,
                        np.deg2rad(
                            args.step_deg
                        ),
                    )

                    if target is not None:
                        requested = True

                        if controller.command_pose(
                            target
                        ):
                            pose = np.r_[
                                controller.target_flange[
                                    :3, 3
                                ],
                                Rotation.from_matrix(
                                    controller.target_flange[
                                        :3, :3
                                    ]
                                ).as_euler(
                                    "xyz"
                                ),
                            ]

                            print(
                                "目标 法兰 [m, rad]:",
                                np.round(
                                    pose,
                                    4,
                                ),
                            )

                        else:
                            print(
                                "跳过：",
                                controller.reason,
                            )

                # --------------------------------------------
                # 固定频率发送最后一个合法 move_js
                # --------------------------------------------

                try:
                    controller.control_tick()
                except Exception as exc:
                    print('move_js 本周期未正常返回，原因：', exc, flush=True)
                    raise
                finally:
                    if requested:
                        print('[KEYBOARD] 按键：', key, flush=True)
                        for name, value in controller.diagnostic.items():
                            print(name, '=', value)
                        print('本周期 move_js 正常返回 =', controller.last_sent,
                              '本次新目标已发送 =', controller.last_sent and controller.diagnostic.get('新目标已接受', False))
                        print('拒绝原因：', controller.reason or '无',
                              '；新目标拒绝时，仍可能持续发送上一合法目标。', flush=True)

                # --------------------------------------------
                # 固定周期
                # --------------------------------------------

                next_tick += period

                now = time.monotonic()

                # 若本周期耗时过长，不补发旧周期
                if next_tick <= now:
                    next_tick = now + period

                time.sleep(
                    max(
                        0.0,
                        next_tick
                        - time.monotonic(),
                    )
                )

    except KeyboardInterrupt:
        # 一般 cbreak 下 Ctrl+C 已被 read() 捕获，
        # 这里作为额外保险。
        print("\nCtrl+C：电子急停。")
        emergency = True

        if controller is not None:
            try:
                controller.emergency_stop()
            except Exception as exc:
                print(
                    "急停请求失败：",
                    exc,
                )

    except Exception:
        # 控制过程中任何异常默认认为需要安全停止。
        emergency = True

        if controller is not None:
            try:
                controller.emergency_stop()
            except Exception as exc:
                print(
                    "异常后的急停请求失败：",
                    exc,
                )

        raise

    finally:
        if arm is not None:

            # 正常 Q 退出：
            # 不自动 emergency_stop，
            # 不 disable，避免机械臂失能下落。
            #
            # 异常 / Space / Esc / Ctrl+C：
            # 前面已经请求过 emergency_stop。

            try:
                arm.disconnect()

            except Exception as exc:
                print(
                    "disconnect 失败：",
                    exc,
                )


if __name__ == "__main__":
    main()
