"""NERO 七轴局部连续 IK；只做数学计算，不导入或连接 SDK。

目标 T_base_flange，末端采用已核验 URDF 的 link7；角度均为 rad。
离线自检：python -m robot_control.nero_ik
"""
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

URDF_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "nero"
    / "nero_description.urdf"
)
MAX_FK_POSITION_ERROR_M = 0.002
MAX_FK_ROTATION_ERROR_DEG = 2.0
DEFAULT_MAX_IK_JOINT_DELTA_RAD = np.deg2rad(10.0)
REGULARIZATION_WEIGHT = 0.05
MAX_NFEV = 80
MAX_SOLVE_SECONDS = 0.075  # 单次求解软预算；当前超过 20 Hz 周期，仅用于实机诊断。


def wrap_angle(angle):
    """数学角差 [-pi, pi)；不能用它绕过有限旋转关节的行程检查。"""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def valid_transform(value):
    try:
        T = np.asarray(value, dtype=float)
        return (T.shape == (4, 4) and np.isfinite(T).all()
                and np.allclose(T[3], [0, 0, 0, 1], atol=1e-6)
                and np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-3)
                and np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-3))
    except (ValueError, TypeError):
        return False


def pose_error(actual, target):
    """返回位置距离 m、相对旋转角 rad，不对欧拉角直接相减。"""
    return (float(np.linalg.norm(actual[:3, 3] - target[:3, 3])),
            float(Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()))


class NeroIK:
    def __init__(
        self,
        urdf_path=URDF_PATH,
        *,
        max_joint_delta_rad=DEFAULT_MAX_IK_JOINT_DELTA_RAD,
    ):
        self.urdf_path = Path(urdf_path).resolve()
        self.max_joint_delta_rad = float(max_joint_delta_rad)
        if (
            not np.isfinite(self.max_joint_delta_rad)
            or self.max_joint_delta_rad < 0
        ):
            raise ValueError('IK max_joint_delta_rad 必须是有限非负数')
        root = ET.parse(self.urdf_path).getroot()
        self.joint_names = [f'joint{i}' for i in range(1, 8)]
        self.base_link, self.flange_link = 'base_link', 'link7'
        self.origins, self.axes, limits = [], [], []
        parent = self.base_link
        for name in self.joint_names:
            matches = root.findall(f"joint[@name='{name}']")
            if len(matches) != 1:
                raise ValueError(f'URDF 缺少或重复关节 {name}')
            joint = matches[0]
            child = f'link{len(self.origins) + 1}'
            if (joint.get('type') != 'revolute'
                    or joint.find('parent') is None or joint.find('child') is None
                    or joint.find('parent').get('link') != parent
                    or joint.find('child').get('link') != child):
                raise ValueError(f'{name}: 必须是 base_link→link7 顺序旋转关节链')
            origin, axis_node, limit = joint.find('origin'), joint.find('axis'), joint.find('limit')
            xyz = np.fromstring(origin.get('xyz', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            rpy = np.fromstring(origin.get('rpy', '0 0 0') if origin is not None else '0 0 0', sep=' ')
            axis = np.fromstring(axis_node.get('xyz', '1 0 0') if axis_node is not None else '1 0 0', sep=' ')
            if (xyz.shape != (3,) or rpy.shape != (3,) or axis.shape != (3,)
                    or not np.isfinite(np.r_[xyz, rpy, axis]).all() or np.linalg.norm(axis) == 0
                    or limit is None):
                raise ValueError(f'{name}: 非法原点、旋转轴或限位')
            T = np.eye(4)
            T[:3, 3], T[:3, :3] = xyz, Rotation.from_euler('xyz', rpy).as_matrix()
            self.origins.append(T)
            self.axes.append(axis / np.linalg.norm(axis))
            limits.append([float(limit.get('lower')), float(limit.get('upper'))])
            parent = child
        self.limits = np.asarray(limits)
        if not np.isfinite(self.limits).all() or np.any(self.limits[:, 0] >= self.limits[:, 1]):
            raise ValueError('非法关节限位')
        self.last_diagnostics = {}

    def in_limits(self, q):
        try:
            q = np.asarray(q, dtype=float)
            return (q.shape == (7,) and np.isfinite(q).all()
                    and np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1]))
        except (ValueError, TypeError):
            return False

    def continuous(self, q, reference):
        if not self.in_limits(q) or not self.in_limits(reference):
            return False
        direct = np.asarray(q) - np.asarray(reference)
        # Nero 七轴均非 continuous。实际发送必须走合法的直接路径。
        return bool(np.max(np.abs(direct)) <= self.max_joint_delta_rad
                    and np.max(np.abs(wrap_angle(direct))) <= self.max_joint_delta_rad)

    def forward(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('FK 要求有限的七轴 rad 数组')
        T = np.eye(4)
        for origin, axis, angle in zip(self.origins, self.axes, q):
            rotation = np.eye(4)
            rotation[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            T = T @ origin @ rotation
        return T

    def solve(self, target_pose, q_seed):
        """从单个 seed 求连续局部解，失败返回 None。

        不把数值失败解释为全局不可达；详细原因放在 last_diagnostics。
        """
        started = time.monotonic()
        diag = self.last_diagnostics = dict(code='IK_FAILED', candidates=[],
                                            ik_success=False, solve_ms=0.0)
        if not valid_transform(target_pose):
            diag['code'] = 'IK_NUMERICAL_ERROR'
            return None
        if not self.in_limits(q_seed):
            try:
                raw = np.asarray(q_seed, dtype=float)
                numeric = raw.shape == (7,) and np.isfinite(raw).all()
            except (TypeError, ValueError):
                numeric = False
            diag['code'] = 'IK_JOINT_LIMIT' if numeric else 'IK_NUMERICAL_ERROR'
            return None
        target = np.asarray(target_pose, dtype=float)
        seed = np.asarray(q_seed, dtype=float).copy()
        p0, r0 = pose_error(self.forward(seed), target)
        # 精确静止目标直接保持 seed；不能用宽 FK 容差形成“永远不动”的死区。
        if p0 < 1e-9 and r0 < 1e-9:
            diag.update(code='IK_SUCCESS', ik_success=True, fk_position_error_mm=p0*1000,
                        fk_rotation_error_deg=np.rad2deg(r0), dq_ik_deg=[0.0]*7)
            return seed

        progress = {
            'residual_evaluations': 0,
            'best_score': (
                (p0 / MAX_FK_POSITION_ERROR_M) ** 2
                + (r0 / np.deg2rad(MAX_FK_ROTATION_ERROR_DEG)) ** 2
            ),
            'best_q': seed.copy(),
            'best_position_error_m': p0,
            'best_rotation_error_rad': r0,
        }

        def residual(candidate):
            if time.monotonic() - started > MAX_SOLVE_SECONDS:
                raise TimeoutError('IK 时间预算耗尽')
            actual = self.forward(candidate)
            position = (actual[:3, 3] - target[:3, 3]) / MAX_FK_POSITION_ERROR_M
            rotation = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
            rotation /= np.deg2rad(MAX_FK_ROTATION_ERROR_DEG)
            progress['residual_evaluations'] += 1
            score = float(position @ position + rotation @ rotation)
            if score < progress['best_score']:
                progress.update(
                    best_score=score,
                    best_q=np.asarray(candidate, dtype=float).copy(),
                    best_position_error_m=float(
                        np.linalg.norm(actual[:3, 3] - target[:3, 3])
                    ),
                    best_rotation_error_rad=float(
                        np.linalg.norm(rotation)
                        * np.deg2rad(MAX_FK_ROTATION_ERROR_DEG)
                    ),
                )
            return np.r_[position, rotation, REGULARIZATION_WEIGHT * wrap_angle(candidate - seed)]

        def record_progress():
            best_q = progress['best_q']
            diag.update(
                residual_evaluations=progress['residual_evaluations'],
                best_intermediate={
                    'fk_position_error_mm': progress['best_position_error_m'] * 1000,
                    'fk_rotation_error_deg': np.rad2deg(
                        progress['best_rotation_error_rad']
                    ),
                    'dq_ik_deg': np.rad2deg(best_q - seed).tolist(),
                    'q_deg': np.rad2deg(best_q).tolist(),
                },
            )

        def assess_candidate(candidate, *, converged, accept_unconverged=False):
            q = np.asarray(candidate, dtype=float)
            info = dict(
                converged=bool(converged),
                q_deg=np.rad2deg(q).tolist(),
            )
            if not np.isfinite(q).all():
                info['code'] = 'IK_NUMERICAL_ERROR'
            elif not self.in_limits(q):
                info['code'] = 'IK_JOINT_LIMIT'
            else:
                p, r = pose_error(self.forward(q), target)
                info.update(
                    fk_position_error_mm=p * 1000,
                    fk_rotation_error_deg=np.rad2deg(r),
                    dq_ik_deg=np.rad2deg(q - seed).tolist(),
                )
                if not converged and not accept_unconverged:
                    info['code'] = 'IK_FAILED'
                elif p >= MAX_FK_POSITION_ERROR_M or r >= np.deg2rad(MAX_FK_ROTATION_ERROR_DEG):
                    info['code'] = 'IK_FK_ERROR'
                elif not self.continuous(q, seed):
                    info['code'] = 'IK_JUMP'
                else:
                    info['code'] = 'IK_SUCCESS'
            return q, info

        try:
            result = least_squares(
                residual,
                seed,
                bounds=(self.limits[:, 0], self.limits[:, 1]),
                max_nfev=MAX_NFEV,
            )
        except TimeoutError:
            record_progress()
            q, info = assess_candidate(
                progress['best_q'], converged=False, accept_unconverged=True
            )
            info['deadline_fallback'] = True
            diag['candidates'].append(info)
            diag.update(
                info,
                budget_exhausted=True,
                solve_ms=(time.monotonic() - started) * 1000,
            )
            if info['code'] != 'IK_SUCCESS':
                return None
            diag['ik_success'] = True
            return q.copy()
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            record_progress()
            info = dict(code='IK_NUMERICAL_ERROR', error=str(exc))
            diag['candidates'].append(info)
            diag.update(info, solve_ms=(time.monotonic() - started) * 1000)
            return None

        record_progress()
        q, info = assess_candidate(
            result.x, converged=bool(result.success)
        )
        info['nfev'] = int(result.nfev)

        diag['candidates'].append(info)
        diag['solve_ms'] = (time.monotonic() - started) * 1000
        diag.update(info)
        if info['code'] != 'IK_SUCCESS':
            return None
        diag['ik_success'] = True
        return q.copy()


def self_test(urdf_path=URDF_PATH):
    """自洽性检查不等于实机标定；使用同一非零 seed，明确打印第七轴。"""
    solver = NeroIK(urdf_path)
    q = np.array([.2, -.3, .15, 1.0, -.1, .2, -.4])
    target = solver.forward(q)
    solution = solver.solve(target, q_seed=q)
    if solution is None:
        raise RuntimeError(solver.last_diagnostics)
    p, r = pose_error(solver.forward(solution), target)
    print('URDF:', solver.urdf_path)
    print('链:', solver.base_link, '→', solver.flange_link, solver.joint_names)
    print('URDF limits rad:\n', solver.limits)
    print('zero FK:\n', solver.forward(np.zeros(7)))
    print('q_input rad:', q, '\nq_ik rad:', solution)
    print('joint_errors deg J1..J7:', np.rad2deg(solution-q))
    print(f'FK position={p*1000:.9f}mm rotation={np.rad2deg(r):.9f}deg')
    if p >= MAX_FK_POSITION_ERROR_M or r >= np.deg2rad(MAX_FK_ROTATION_ERROR_DEG):
        raise RuntimeError('离线自检失败')
    print('离线 IK 自检通过；未连接 CAN 或 SteamVR，不代表实机运动验证。')


if __name__ == '__main__':
    self_test()
