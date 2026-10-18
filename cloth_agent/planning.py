"""Local waypoint planning and TOPPRA retiming; no robot connection."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from robot_control.nero_ik import NeroIK
from .config import effective_arm, joint_vector
from .geometry import interpolate_poses, tcp_to_flange, transform


class PlanningError(ValueError):
    def __init__(self, reason, *, index=None, diagnostics=None):
        super().__init__(reason)
        self.index = index
        self.diagnostics = diagnostics or {}

    def as_dict(self):
        return {"ok": False, "reason": str(self), "index": self.index,
                "diagnostics": self.diagnostics}


@dataclass
class MotionPlan:
    arm: str
    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    mode: str
    _spline: CubicHermiteSpline = field(init=False, repr=False)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float).copy()
        self.positions = np.asarray(self.positions, dtype=float).copy()
        self.velocities = np.asarray(self.velocities, dtype=float).copy()
        if (self.times.ndim != 1 or len(self.times) < 2 or self.times[0] != 0
                or np.any(np.diff(self.times) <= 0)
                or self.positions.shape != (len(self.times), 7)
                or self.velocities.shape != self.positions.shape
                or not all(np.isfinite(x).all() for x in
                           (self.times, self.positions, self.velocities))):
            raise ValueError("Invalid timed joint trajectory.")
        self._spline = CubicHermiteSpline(self.times, self.positions, self.velocities)

    @property
    def duration(self):
        return float(self.times[-1])

    def sample(self, elapsed, derivative=0):
        return self._spline(np.clip(elapsed, 0, self.duration), nu=derivative)

    def as_dict(self):
        return {"arm": self.arm, "times": self.times.tolist(),
                "positions": self.positions.tolist(), "velocities": self.velocities.tolist(),
                "mode": self.mode}

    def peak_rates(self):
        """Exact velocity/acceleration extrema of the stored cubic pieces."""
        c = self._spline.c
        h = np.diff(self.times)[:, None]
        roots = np.divide(-c[1], 3 * c[0], out=np.zeros_like(c[0]), where=np.abs(c[0]) > 1e-15)
        roots = np.clip(roots, 0, h)
        velocities = [c[2], 3 * c[0] * h**2 + 2 * c[1] * h + c[2],
                      3 * c[0] * roots**2 + 2 * c[1] * roots + c[2]]
        accelerations = [2 * c[1], 6 * c[0] * h + 2 * c[1]]
        return (np.max(np.abs(np.concatenate(velocities)), axis=0),
                np.max(np.abs(np.concatenate(accelerations)), axis=0))

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


class Planner:
    def __init__(self, config):
        self.config = config
        self.motion = config["motion"]
        self.arms = {name: effective_arm(config, name) for name in config["arms"]}
        self.ik = {name: NeroIK(max_joint_delta_rad=self.motion["max_ik_step_rad"])
                   for name in self.arms}
        self._lock = Lock()  # Pinocchio model data are mutable.

    def plan(self, arm, seed, *, waypoints=None, mode="cartesian", target_joints=None):
        if arm not in self.arms:
            raise PlanningError(f"Unknown arm: {arm}")
        with self._lock:
            solver = self.ik[arm]
            seed = joint_vector(seed, "seed")
            if not solver.in_limits(seed):
                raise PlanningError("Seed is outside joint limits.")
            points = [seed.copy()]
            if target_joints is not None:
                if waypoints is not None:
                    raise PlanningError("Use target_joints or TCP waypoints, not both.")
                target = joint_vector(target_joints, "target_joints")
                self._append_joint_segment(points, target)
                mode = "joint"
            else:
                if mode not in ("joint", "cartesian") or not waypoints:
                    raise PlanningError("Provide TCP waypoints and joint/cartesian mode.")
                settings = self.arms[arm]
                if settings["task_from_base"] is None or settings["flange_from_tcp"] is None:
                    raise PlanningError("TCP and base calibration are not configured.")
                current_tcp = (transform(settings["task_from_base"]) @ solver.forward(seed)
                               @ transform(settings["flange_from_tcp"]))
                for index, waypoint in enumerate(waypoints):
                    goal = transform(waypoint)
                    poses = ([goal] if mode == "joint" else interpolate_poses(
                        current_tcp, goal, self.motion["position_step_m"],
                        self.motion["rotation_step_rad"]))
                    for pose in poses:
                        flange = tcp_to_flange(pose, settings["task_from_base"],
                                               settings["flange_from_tcp"])
                        q = solver.solve(flange, points[-1])
                        if q is None:
                            raise PlanningError("IK failed.", index=index,
                                                diagnostics=solver.last_diagnostics.copy())
                        if mode == "joint":
                            self._append_joint_segment(points, q)
                        else:
                            points.append(q)
                    current_tcp = goal
            self._check_path(arm, np.asarray(points))
            plan = self._retime(arm, np.asarray(points), mode)
            self._check_path(arm, plan.sample(np.linspace(0, plan.duration,
                                                         max(100, len(plan.times) * 2))))
            return plan

    def _append_joint_segment(self, points, target):
        count = max(1, int(np.ceil(np.max(np.abs(target - points[-1]))
                                   / self.motion["joint_step_rad"])))
        points.extend(np.linspace(points[-1], target, count + 1)[1:])

    def _check_path(self, arm, points):
        solver, settings = self.ik[arm], self.arms[arm]
        for index, q in enumerate(points):
            if not solver.in_limits(q):
                raise PlanningError("Path violates joint limits.", index=index)
            if settings["workspace"] is not None:
                tcp = (transform(settings["task_from_base"]) @ solver.forward(q)
                       @ transform(settings["flange_from_tcp"]))[:3, 3]
                bounds = np.asarray(settings["workspace"], dtype=float)
                if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] >= bounds[1]):
                    raise PlanningError("workspace must be [minimum_xyz, maximum_xyz].")
                if np.any(tcp < bounds[0]) or np.any(tcp > bounds[1]):
                    raise PlanningError("TCP leaves the configured task workspace.", index=index)

    def _retime(self, arm, points, mode):
        import toppra as ta
        from toppra import algorithm, constraint

        keep = np.r_[True, np.max(np.abs(np.diff(points, axis=0)), axis=1) > 1e-10]
        points = points[keep]
        if len(points) == 1:
            return MotionPlan(arm, [0, 1 / self.config["control_hz"]],
                              np.repeat(points, 2, axis=0), np.zeros((2, 7)), mode)
        settings = self.arms[arm]
        velocity = joint_vector(settings["velocity_limits"], "velocity_limits")
        acceleration = joint_vector(settings["acceleration_limits"], "acceleration_limits")
        if np.any(velocity <= 0) or np.any(acceleration <= 0):
            raise PlanningError("Motion limits must be positive.")
        ss = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        ss /= ss[-1]
        path = ta.SplineInterpolator(ss, points)
        instance = algorithm.TOPPRA(
            [constraint.JointVelocityConstraint(velocity),
             constraint.JointAccelerationConstraint(acceleration)], path,
            gridpoints=np.linspace(0, 1, max(100, len(points) * 5)), solver_wrapper="seidel")
        trajectory = instance.compute_trajectory(0, 0)
        if trajectory is None:
            raise PlanningError("TOPPRA could not time-parameterize the joint path.")
        # Leave sampling margin below the configured limits.
        scale = 1.05
        duration = float(trajectory.duration) * scale
        times = np.linspace(0, duration, max(3, int(np.ceil(duration / .01)) + 1))
        plan = MotionPlan(arm, times, trajectory(times / scale),
                          trajectory(times / scale, 1) / scale, mode)
        # Retiming interpolation can introduce small extrema. Slow the final
        # stored cubic by its exact extrema, rather than claiming TOPPRA's
        # constraints also bound a different interpolation automatically.
        peak_v, peak_a = plan.peak_rates()
        correction = max(1., float(np.max(peak_v / velocity)),
                         float(np.sqrt(np.max(peak_a / acceleration)))) * 1.001
        if correction > 1:
            plan = MotionPlan(arm, times * correction, plan.positions,
                              plan.velocities / correction, mode)
        return plan
