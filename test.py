"""Vive -> NERO TCP/法兰目标安全测试（只计算，不运动）

用途：
1. 设置 SDK TCP offset：法兰 +X 方向 5 cm；TCP 方向为先绕自身 Y -90°，再绕自身 Z +90°。
2. 读取 Vive Tracker 和机械臂当前法兰/TCP反馈。
3. 使用与 teleop_nero_tracker.py 相同的映射，计算目标 TCP。
4. 将目标 TCP 反算成最终本应传给 move_pose() 的法兰目标。
5. 只打印和可视化；本文件没有 arm.enable()、arm.move_pose() 或任何运动命令。

注意：
- set_tcp_offset() 只写 SDK/Driver 实例，本程序每次启动都会重新设置。
- TCP_OFFSET_RPY 使用 rad；位置使用 m。
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation


# =========================
# 测试参数
# =========================
CONTROL_HZ = 10
PRINT_HZ = 5
POSITION_SCALE = 1.0
MAX_TRAJECTORY_POINTS = 2000
AXIS_LENGTH = 0.06
VIEW_MARGIN = 0.12
MIN_VIEW_RANGE = 0.20


# =========================
# Vive Standing world -> NERO Base 标定旋转
# 请与最终 teleop_nero_tracker.py 中的标定矩阵保持一致
# =========================
R_base_tracker = np.array([
    [0.9214312744, -0.0458795358, 0.3858231133],
    [0.3878845517, 0.0509047504, -0.9203011903],
    [0.0225827621, 0.9976491240, 0.0647011922],
])


# =========================
# TCP 标定结果（法兰坐标系下）
# =========================
# TCP 原点：沿当前法兰 +X 偏移 5 cm
# TCP 方向：先绕自身 Y -90°，再绕旋转后的自身 Z +90°
# 在本项目采用的 xyz-RPY 表达下等效为：roll=-90°, pitch=0°, yaw=+90°
TCP_OFFSET = np.array([
    0.05,
    0.0,
    0.0,
    -np.pi / 2,
    0.0,
    np.pi / 2,
], dtype=float)


# =========================
# 位姿工具
# =========================
def pose_to_matrix(pose):
    """[x,y,z,roll,pitch,yaw] -> 4x4；R=Rz(yaw)Ry(pitch)Rx(roll)。"""
    pose = np.asarray(pose, dtype=float)
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    return T


def matrix_to_pose(T):
    """4x4 -> [x,y,z,roll,pitch,yaw]，角度单位 rad。"""
    T = np.asarray(T, dtype=float)
    return np.concatenate((
        T[:3, 3],
        Rotation.from_matrix(T[:3, :3]).as_euler("xyz"),
    ))


def matrix_rpy_deg(T):
    return Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)


def rotation_error_deg(R_a, R_b):
    return np.rad2deg(
        Rotation.from_matrix(R_a @ R_b.T).magnitude()
    )


def validate_mapping():
    if not np.isfinite(R_base_tracker).all():
        raise ValueError("R_base_tracker 含 NaN/Inf。")
    if not np.allclose(R_base_tracker.T @ R_base_tracker, np.eye(3), atol=1e-6):
        raise ValueError("R_base_tracker 不是正交矩阵。")
    if not np.isclose(np.linalg.det(R_base_tracker), 1.0, atol=1e-6):
        raise ValueError("R_base_tracker 的行列式必须为 +1。")


def read_tracker_pose(tracker):
    """读取 OpenVR 3x4 device->Standing world 位姿并过滤异常帧。"""
    pose = tracker.get_pose_matrix()
    if pose is None:
        return None

    T = np.eye(4)
    T[:3, :] = np.asarray(pose)["m"]
    R = T[:3, :3]

    if not np.isfinite(T).all():
        return None
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-3):
        return None
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-3):
        return None
    return T


def wait_for_tracker_pose(tracker, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        T = read_tracker_pose(tracker)
        if T is not None:
            return T
        time.sleep(0.02)
    raise RuntimeError("5 秒内未读取到有效 Vive Tracker 位姿。")


def wait_for_robot_poses(arm, timeout=5.0):
    """仅读取反馈，不使能、不运动。"""
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            tcp = arm.get_tcp_pose()
            flange = arm.get_flange_pose()
            return pose_to_matrix(tcp), pose_to_matrix(flange)
        except RuntimeError as exc:
            last_error = exc
            time.sleep(0.05)
    if last_error is not None:
        raise last_error
    raise RuntimeError("5 秒内未读取到机械臂位姿反馈。")


# =========================
# 与遥操程序一致的映射
# =========================
def tracker_target(T_tracker_0, T_tracker, T_tcp_0):
    """Vive 相对运动 -> NERO Base 下目标 TCP。"""
    target_tcp = T_tcp_0.copy()

    delta_p_tracker = T_tracker[:3, 3] - T_tracker_0[:3, 3]
    delta_p_nero = POSITION_SCALE * (R_base_tracker @ delta_p_tracker)
    target_tcp[:3, 3] += delta_p_nero

    delta_R_tracker = T_tracker[:3, :3] @ T_tracker_0[:3, :3].T
    delta_R_nero = R_base_tracker @ delta_R_tracker @ R_base_tracker.T
    target_tcp[:3, :3] = delta_R_nero @ T_tcp_0[:3, :3]

    return target_tcp, delta_p_tracker, delta_p_nero, delta_R_nero


# =========================
# SDK TCP 自检
# =========================
def check_sdk_tcp_relation(T_tcp_0, T_flange_0):
    """检查 SDK get_tcp_pose() 是否与设置的 TCP_OFFSET 一致。"""
    T_flange_tcp_expected = pose_to_matrix(TCP_OFFSET)
    T_flange_tcp_sdk = np.linalg.inv(T_flange_0) @ T_tcp_0

    pos_err = np.linalg.norm(
        T_flange_tcp_sdk[:3, 3] - T_flange_tcp_expected[:3, 3]
    )
    rot_err = rotation_error_deg(
        T_flange_tcp_sdk[:3, :3],
        T_flange_tcp_expected[:3, :3],
    )

    print("\n========== SDK TCP offset 自检 ==========")
    print("设定 TCP offset [m, rad]:")
    print(np.array2string(TCP_OFFSET, precision=6, suppress_small=True))
    print("SDK反馈反推出的 Flange->TCP [x y z roll pitch yaw]:")
    print(np.array2string(matrix_to_pose(T_flange_tcp_sdk), precision=6, suppress_small=True))
    print(f"位置误差: {pos_err * 1000:.3f} mm")
    print(f"旋转误差: {rot_err:.6f} deg")

    if pos_err < 1e-4 and rot_err < 1e-3:
        print("TCP_OFFSET_CHECK: PASS")
    else:
        print("TCP_OFFSET_CHECK: WARNING")
        print("SDK 的 TCP 欧拉角约定或 get_tcp_pose() 实现可能与当前假设不同，请先不要运动。")

    return T_flange_tcp_sdk, T_flange_tcp_expected, pos_err, rot_err


# =========================
# 终端输出
# =========================
def print_state(
    T_tracker,
    delta_p_tracker,
    delta_p_nero,
    delta_R_nero,
    target_tcp,
    target_flange,
    T_flange_tcp_sdk,
    sdk_conversion_used,
    manual_sdk_pos_err,
    manual_sdk_rot_err,
):
    raw_p = T_tracker[:3, 3]
    raw_rpy = matrix_rpy_deg(T_tracker)
    delta_rpy = Rotation.from_matrix(delta_R_nero).as_euler("xyz", degrees=True)

    tcp_pose_rad = matrix_to_pose(target_tcp)
    tcp_rpy_deg = matrix_rpy_deg(target_tcp)

    flange_pose_rad = matrix_to_pose(target_flange)
    flange_rpy_deg = matrix_rpy_deg(target_flange)

    # 用最终法兰目标重新正向计算 TCP，检查反算是否自洽
    reconstructed_tcp = target_flange @ T_flange_tcp_sdk
    relation_pos_err = np.linalg.norm(
        reconstructed_tcp[:3, 3] - target_tcp[:3, 3]
    )
    relation_rot_err = rotation_error_deg(
        reconstructed_tcp[:3, :3], target_tcp[:3, :3]
    )

    print(
        "\033[2J\033[H"
        "========== Vive -> TCP -> Flange 安全测试 =========="
        "\n*** 本程序没有发送任何运动命令 ***\n\n"
        f"Vive abs [m]\n"
        f"  x={raw_p[0]: .4f}  y={raw_p[1]: .4f}  z={raw_p[2]: .4f}\n"
        f"Vive abs RPY [deg]\n"
        f"  r={raw_rpy[0]: .2f}  p={raw_rpy[1]: .2f}  y={raw_rpy[2]: .2f}\n\n"
        f"Vive delta [m]\n"
        f"  dx={delta_p_tracker[0]: .4f}  dy={delta_p_tracker[1]: .4f}  dz={delta_p_tracker[2]: .4f}\n"
        f"Mapped NERO delta [m]\n"
        f"  dx={delta_p_nero[0]: .4f}  dy={delta_p_nero[1]: .4f}  dz={delta_p_nero[2]: .4f}\n"
        f"Mapped delta RPY [deg]\n"
        f"  r={delta_rpy[0]: .2f}  p={delta_rpy[1]: .2f}  y={delta_rpy[2]: .2f}\n\n"
        "----- 目标 TCP（你希望工具中心到达的位姿）-----\n"
        f"TCP xyz [m] : {tcp_pose_rad[0]: .5f}, {tcp_pose_rad[1]: .5f}, {tcp_pose_rad[2]: .5f}\n"
        f"TCP RPY [deg]: {tcp_rpy_deg[0]: .2f}, {tcp_rpy_deg[1]: .2f}, {tcp_rpy_deg[2]: .2f}\n\n"
        "----- 最终法兰控制量（本应传给 move_pose）-----\n"
        f"move_pose xyz [m] : {flange_pose_rad[0]: .5f}, {flange_pose_rad[1]: .5f}, {flange_pose_rad[2]: .5f}\n"
        f"move_pose RPY [rad]: {flange_pose_rad[3]: .6f}, {flange_pose_rad[4]: .6f}, {flange_pose_rad[5]: .6f}\n"
        f"move_pose RPY [deg]: {flange_rpy_deg[0]: .2f}, {flange_rpy_deg[1]: .2f}, {flange_rpy_deg[2]: .2f}\n\n"
        "----- TCP/法兰反算自检 -----\n"
        f"target_flange @ T_flange_tcp -> target_tcp\n"
        f"position error = {relation_pos_err * 1000:.6f} mm\n"
        f"rotation error = {relation_rot_err:.9f} deg\n\n"
        "----- SDK get_tcp2flange_pose 对照 -----\n"
        f"SDK conversion used = {sdk_conversion_used}\n"
        f"manual vs SDK position error = {manual_sdk_pos_err * 1000:.6f} mm\n"
        f"manual vs SDK rotation error = {manual_sdk_rot_err:.9f} deg\n\n"
        "关闭图窗或 Ctrl+C 退出。\n",
        flush=True,
    )


# =========================
# 可视化
# =========================
class RobotTargetVisualizer:
    """在 NERO Base 下同时画初始/目标 Flange 与 TCP。"""

    def __init__(self, T_flange_0, T_tcp_0):
        plt.ion()
        self.fig = plt.figure("Vive TCP/Flange Control Preview")
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_title("NERO Base: TCP target and final Flange command (NO MOTION)")
        self.ax.set_xlabel("Base X [m]")
        self.ax.set_ylabel("Base Y [m]")
        self.ax.set_zlabel("Base Z [m]")

        self.T_flange_0 = T_flange_0.copy()
        self.T_tcp_0 = T_tcp_0.copy()

        # 初始位置
        self.ax.plot(
            [T_flange_0[0, 3]], [T_flange_0[1, 3]], [T_flange_0[2, 3]],
            marker="s", linestyle="None", label="initial flange",
        )
        self.ax.plot(
            [T_tcp_0[0, 3]], [T_tcp_0[1, 3]], [T_tcp_0[2, 3]],
            marker="o", linestyle="None", label="initial TCP",
        )

        # 当前目标位置
        self.flange_point, = self.ax.plot(
            [T_flange_0[0, 3]], [T_flange_0[1, 3]], [T_flange_0[2, 3]],
            marker="s", linestyle="None", label="target flange / move_pose",
        )
        self.tcp_point, = self.ax.plot(
            [T_tcp_0[0, 3]], [T_tcp_0[1, 3]], [T_tcp_0[2, 3]],
            marker="o", linestyle="None", label="target TCP",
        )

        # 法兰到 TCP 的工具连线
        self.tool_line, = self.ax.plot([], [], [], linewidth=2.0, label="flange -> TCP")

        # TCP 目标轨迹
        self.tcp_trajectory_line, = self.ax.plot([], [], [], linewidth=1.2, label="TCP target trajectory")
        self.tcp_trajectory = []

        # TCP 目标坐标轴：不同线型表示 x/y/z，不依赖特定颜色
        self.tcp_axis_x, = self.ax.plot([], [], [], linestyle="-", linewidth=2.0, label="TCP X")
        self.tcp_axis_y, = self.ax.plot([], [], [], linestyle="--", linewidth=2.0, label="TCP Y")
        self.tcp_axis_z, = self.ax.plot([], [], [], linestyle=":", linewidth=2.0, label="TCP Z")

        # 法兰目标坐标轴
        self.flange_axis_x, = self.ax.plot([], [], [], linestyle="-", linewidth=1.0, label="Flange X")
        self.flange_axis_y, = self.ax.plot([], [], [], linestyle="--", linewidth=1.0, label="Flange Y")
        self.flange_axis_z, = self.ax.plot([], [], [], linestyle=":", linewidth=1.0, label="Flange Z")

        self.text = self.ax.text2D(
            0.02, 0.97, "", transform=self.ax.transAxes, va="top", family="monospace"
        )

        self.ax.legend(loc="upper right", fontsize=8)
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        self._fit_view([T_flange_0[:3, 3], T_tcp_0[:3, 3]])
        plt.show(block=False)

    def is_open(self):
        return plt.fignum_exists(self.fig.number)

    @staticmethod
    def _set_point(line, p):
        line.set_data([p[0]], [p[1]])
        line.set_3d_properties([p[2]])

    @staticmethod
    def _set_segment(line, p0, p1):
        line.set_data([p0[0], p1[0]], [p0[1], p1[1]])
        line.set_3d_properties([p0[2], p1[2]])

    def _set_frame_axes(self, T, lines):
        p = T[:3, 3]
        R = T[:3, :3]
        for i, line in enumerate(lines):
            self._set_segment(line, p, p + AXIS_LENGTH * R[:, i])

    def _fit_view(self, points):
        pts = np.asarray(points, dtype=float)
        p_min = pts.min(axis=0)
        p_max = pts.max(axis=0)
        center = 0.5 * (p_min + p_max)
        half = max(
            MIN_VIEW_RANGE / 2,
            0.5 * float(np.max(p_max - p_min)) + VIEW_MARGIN,
        )
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)

    def update(self, target_tcp, target_flange):
        p_tcp = target_tcp[:3, 3]
        p_flange = target_flange[:3, 3]

        self._set_point(self.tcp_point, p_tcp)
        self._set_point(self.flange_point, p_flange)
        self._set_segment(self.tool_line, p_flange, p_tcp)

        self._set_frame_axes(
            target_tcp,
            (self.tcp_axis_x, self.tcp_axis_y, self.tcp_axis_z),
        )
        self._set_frame_axes(
            target_flange,
            (self.flange_axis_x, self.flange_axis_y, self.flange_axis_z),
        )

        self.tcp_trajectory.append(p_tcp.copy())
        if len(self.tcp_trajectory) > MAX_TRAJECTORY_POINTS:
            self.tcp_trajectory = self.tcp_trajectory[-MAX_TRAJECTORY_POINTS:]
        traj = np.asarray(self.tcp_trajectory)
        self.tcp_trajectory_line.set_data(traj[:, 0], traj[:, 1])
        self.tcp_trajectory_line.set_3d_properties(traj[:, 2])

        tcp_rpy = matrix_rpy_deg(target_tcp)
        flange_pose = matrix_to_pose(target_flange)
        flange_rpy = matrix_rpy_deg(target_flange)
        self.text.set_text(
            "TARGET TCP\n"
            f"xyz = [{p_tcp[0]: .4f}, {p_tcp[1]: .4f}, {p_tcp[2]: .4f}] m\n"
            f"rpy = [{tcp_rpy[0]: .1f}, {tcp_rpy[1]: .1f}, {tcp_rpy[2]: .1f}] deg\n\n"
            "FINAL move_pose (FLANGE)\n"
            f"xyz = [{p_flange[0]: .4f}, {p_flange[1]: .4f}, {p_flange[2]: .4f}] m\n"
            f"rpy = [{flange_pose[3]: .3f}, {flange_pose[4]: .3f}, {flange_pose[5]: .3f}] rad\n"
            f"    = [{flange_rpy[0]: .1f}, {flange_rpy[1]: .1f}, {flange_rpy[2]: .1f}] deg"
        )

        # 将初始点、当前目标和轨迹纳入视野
        points = [
            self.T_flange_0[:3, 3],
            self.T_tcp_0[:3, 3],
            p_flange,
            p_tcp,
        ]
        if len(self.tcp_trajectory) > 0:
            points.extend(self.tcp_trajectory)
        self._fit_view(points)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# =========================
# 主程序
# =========================
def main():
    validate_mapping()

    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(script_dir / "dp_real_0309"))

    from track import ViveTrackerModule
    from config import NERO_CAN_CHANNEL, NERO_CAN_INTERFACE
    from robot_control import NeroArm

    tracker_module = None
    arm = None

    try:
        # ---------- Vive ----------
        tracker_module = ViveTrackerModule()
        tracker_module.print_discovered_objects()
        devices = tracker_module.return_selected_devices("tracker")
        if len(devices) != 1:
            raise RuntimeError(
                f"检测到 {len(devices)} 个 Tracker；本测试要求只连接遥操作所使用的一个 Tracker。"
            )
        tracker = next(iter(devices.values()))

        # ---------- 机械臂：只连接和读取，不 enable ----------
        arm = NeroArm(
            can_interface=NERO_CAN_INTERFACE,
            can_channel=NERO_CAN_CHANNEL,
        )
        arm.connect()
        while not arm._driver.enable():
            arm._driver.set_normal_mode()
            time.sleep(0.01)

        print("CAN mode enabled")
        print("ArmStatus", arm.get_arm_status())

        if not hasattr(arm, "set_tcp_offset"):
            raise AttributeError(
                "当前 NeroArm 封装没有公开 set_tcp_offset()；请先把 SDK 的该接口透传到 NeroArm。"
            )
        
        # 只保存在 SDK/Driver 实例，不向控制器发送运动目标
        arm.set_tcp_offset(TCP_OFFSET.tolist())

        print("机械臂仅已 connect，未 enable，未发送运动命令。")
        print("请保持 Vive 和机械臂静止，读取初始位姿……")
        aa = arm.get_tcp_pose()
        print("tcp位姿为：",aa)

        T_tracker_0 = wait_for_tracker_pose(tracker, timeout=5.0)
        T_tcp_0, T_flange_0 = wait_for_robot_poses(arm, timeout=5.0)

        # 检查 SDK set_tcp_offset/get_tcp_pose 是否采用我们预期的变换约定
        (
            T_flange_tcp_sdk,
            _T_flange_tcp_expected,
            offset_pos_err,
            offset_rot_err,
        ) = check_sdk_tcp_relation(T_tcp_0, T_flange_0)

        # 目标 TCP -> 目标法兰所需固定变换（矩阵手算备用/对照）
        T_tcp_flange = np.linalg.inv(T_tcp_0) @ T_flange_0

        sdk_tcp2flange_available = hasattr(arm, "get_tcp2flange_pose")
        print(f"SDK get_tcp2flange_pose available: {sdk_tcp2flange_available}")
        if not sdk_tcp2flange_available:
            print("当前 NeroArm 未公开 get_tcp2flange_pose()；将使用等价矩阵反算，并继续做几何自检。")

        # 初始时如果 Vive delta = 0，反算出的目标法兰应等于当前法兰
        initial_target_flange = T_tcp_0 @ T_tcp_flange
        init_pos_err = np.linalg.norm(
            initial_target_flange[:3, 3] - T_flange_0[:3, 3]
        )
        init_rot_err = rotation_error_deg(
            initial_target_flange[:3, :3], T_flange_0[:3, :3]
        )
        print("\n========== 初始零运动自检 ==========")
        print(f"Vive delta=0 时法兰位置误差: {init_pos_err * 1000:.9f} mm")
        print(f"Vive delta=0 时法兰姿态误差: {init_rot_err:.9f} deg")
        if init_pos_err < 1e-9 and init_rot_err < 1e-7:
            print("ZERO_MOTION_CHECK: PASS")
        else:
            print("ZERO_MOTION_CHECK: WARNING")

        if offset_pos_err >= 1e-4 or offset_rot_err >= 1e-3:
            print("\n警告：TCP offset 自检未通过。当前程序仍只显示计算结果，但请不要据此启动遥操作。")

        visualizer = RobotTargetVisualizer(T_flange_0, T_tcp_0)

        dt = 1.0 / CONTROL_HZ
        print_dt = 1.0 / PRINT_HZ
        last_print = 0.0

        while visualizer.is_open():
            cycle_start = time.monotonic()
            T_tracker = read_tracker_pose(tracker)

            if T_tracker is not None:
                (
                    target_tcp,
                    delta_p_tracker,
                    delta_p_nero,
                    delta_R_nero,
                ) = tracker_target(T_tracker_0, T_tracker, T_tcp_0)

                # TCP -> 法兰反算：优先使用 SDK 官方 get_tcp2flange_pose()。
                # 同时保留矩阵手算结果做交叉验证。
                target_flange_manual = target_tcp @ T_tcp_flange

                if sdk_tcp2flange_available:
                    target_tcp_pose = matrix_to_pose(target_tcp).tolist()
                    target_flange_sdk_pose = np.asarray(
                        arm.get_tcp2flange_pose(target_tcp_pose), dtype=float
                    )
                    target_flange = pose_to_matrix(target_flange_sdk_pose)
                    manual_sdk_pos_err = np.linalg.norm(
                        target_flange_manual[:3, 3] - target_flange[:3, 3]
                    )
                    manual_sdk_rot_err = rotation_error_deg(
                        target_flange_manual[:3, :3], target_flange[:3, :3]
                    )
                    sdk_conversion_used = True
                else:
                    target_flange = target_flange_manual
                    manual_sdk_pos_err = 0.0
                    manual_sdk_rot_err = 0.0
                    sdk_conversion_used = False

                # 这里故意不调用 arm.move_pose()/move_p()
                visualizer.update(target_tcp, target_flange)

                now = time.monotonic()
                if now - last_print >= print_dt:
                    print_state(
                        T_tracker,
                        delta_p_tracker,
                        delta_p_nero,
                        delta_R_nero,
                        target_tcp,
                        target_flange,
                        T_flange_tcp_sdk,
                        sdk_conversion_used,
                        manual_sdk_pos_err,
                        manual_sdk_rot_err,
                    )
                    last_print = now

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print("\n测试结束。")

    finally:
        # 不调用 emergency_stop：本程序从未发出运动命令。
        if arm is not None:
            try:
                arm.disconnect()
            except Exception as exc:
                print(f"机械臂断开时出现异常: {exc}")

        if tracker_module is not None:
            del tracker_module

        plt.ioff()
        plt.close("all")


if __name__ == "__main__":
    main()
