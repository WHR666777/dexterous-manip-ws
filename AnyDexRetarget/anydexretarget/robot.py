from typing import List
import time

import numpy as np
import numpy.typing as npt
import pinocchio as pin


class RobotWrapper:
    """Pinocchio robot wrapper for forward kinematics."""

    def __init__(self, urdf_path: str):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        if self.model.nv != self.model.nq:
            raise NotImplementedError("Cannot handle robot with special joint.")
        self._point_cache_key = None
        self._point_cache = None

        # Timing statistics for FK and Jacobian
        self._timing_enabled = False
        self._fk_time_sum = 0.0
        self._jacobian_time_sum = 0.0
        self._fk_call_count = 0
        self._jacobian_call_count = 0

    def enable_timing(self, enabled: bool = True):
        """Enable or disable timing statistics."""
        self._timing_enabled = enabled
        if enabled:
            self.reset_timing()

    def reset_timing(self):
        """Reset timing statistics."""
        self._fk_time_sum = 0.0
        self._jacobian_time_sum = 0.0
        self._fk_call_count = 0
        self._jacobian_call_count = 0

    def get_timing_stats(self):
        """Get timing statistics.

        Returns:
            dict with keys:
                - fk_avg_us: average FK time in microseconds
                - jacobian_avg_us: average Jacobian time in microseconds
                - fk_call_count: number of FK calls
                - jacobian_call_count: number of Jacobian calls
        """
        fk_avg = (self._fk_time_sum / self._fk_call_count * 1e6) if self._fk_call_count > 0 else 0
        jac_avg = (self._jacobian_time_sum / self._jacobian_call_count * 1e6) if self._jacobian_call_count > 0 else 0
        return {
            'fk_avg_us': fk_avg,
            'jacobian_avg_us': jac_avg,
            'fk_call_count': self._fk_call_count,
            'jacobian_call_count': self._jacobian_call_count,
        }

    @property
    def dof_joint_names(self) -> List[str]:
        """Return names of joints with DOF > 0."""
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def joint_limits(self):
        """Return joint limits as (lower, upper) pairs."""
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    def _make_pin_qpos(self, qpos: npt.NDArray) -> npt.NDArray:
        """Create a Pinocchio-owned q vector for eigenpy compatibility."""
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != self.model.nq:
            raise ValueError(
                f"Expected qpos with {self.model.nq} values, got shape {qpos.shape}"
            )
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos contains NaN or infinite values")

        pin_qpos = pin.neutral(self.model)
        pin_qpos[:] = qpos
        return pin_qpos

    def get_link_index(self, name: str) -> int:
        """Get frame index by name."""
        idx = self.model.getFrameId(name, pin.BODY)
        if idx >= self.model.nframes:
            raise ValueError(
                f"Link/frame '{name}' not found in robot model. "
                f"Model has {self.model.nframes} frames."
            )
        return idx

    def compute_forward_kinematics(self, qpos: npt.NDArray):
        """Compute forward kinematics for all links."""
        qpos = self._make_pin_qpos(qpos)
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        """Get link pose as 4x4 homogeneous matrix."""
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return np.array(pose.homogeneous, dtype=np.float64, copy=True)

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        """Compute Jacobian for a single link."""
        qpos = self._make_pin_qpos(qpos)
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return np.array(J, dtype=np.float64, copy=True)

    def compute_all_jacobians_batch(self, qpos: npt.NDArray, link_indices: List[int]) -> npt.NDArray:
        """Batch compute position Jacobians for multiple links.

        This is more efficient than calling compute_single_link_local_jacobian
        multiple times because it uses computeJointJacobians once.

        Args:
            qpos: Joint positions
            link_indices: List of frame indices

        Returns:
            jacobians: (num_links, 3, nq) position Jacobians in world frame
        """
        return self.compute_all_jacobians_batch_with_offsets(qpos, link_indices)

    def _prepare_point_offsets(self, link_indices, local_offsets):
        """Cache fixed offsets and their cross matrices; detect in-place edits.

        Keep the original allclose-to-zero Jacobian convention, including tiny
        offsets. A single-entry cache avoids growing with dynamic callers.
        """
        links = tuple(link_indices)
        if local_offsets is None:
            key = (links, None)
        else:
            local_offsets = np.asarray(local_offsets, dtype=np.float64)
            if local_offsets.shape != (len(links), 3):
                raise ValueError('Point offsets must have shape (num_links, 3)')
            key = (links, local_offsets.tobytes())
        if key != self._point_cache_key:
            offsets = (np.zeros((len(links), 3)) if local_offsets is None
                       else local_offsets.copy())
            cross = np.zeros((len(links), 3, 3))
            x, y, z = offsets.T
            cross[:, 0, 1], cross[:, 0, 2] = -z, y
            cross[:, 1, 0], cross[:, 1, 2] = z, -x
            cross[:, 2, 0], cross[:, 2, 1] = -y, x
            cross[np.all(np.isclose(offsets, 0.0), axis=1)] = 0.0
            self._point_cache_key = key
            self._point_cache = offsets, cross
        return self._point_cache

    def compute_points_batch(
        self,
        qpos: npt.NDArray,
        link_indices: List[int],
        local_offsets: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Batch compute positions for points attached to frames.

        Args:
            qpos: Joint positions
            link_indices: List of frame indices
            local_offsets: (num_links, 3) offsets in each frame's local coordinates.
                If None, uses the frame origins.

        Returns:
            positions: (num_links, 3) world positions
        """
        qpos = self._make_pin_qpos(qpos)
        pin.forwardKinematics(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)

        offsets, _ = self._prepare_point_offsets(link_indices, local_offsets)
        frames = [self.data.oMf[idx] for idx in link_indices]
        translations = np.array([frame.translation for frame in frames], dtype=np.float64)
        rotations = np.array([frame.rotation for frame in frames], dtype=np.float64)
        return translations + (rotations @ offsets[..., None])[..., 0]

    def compute_all_jacobians_batch_with_offsets(
        self,
        qpos: npt.NDArray,
        link_indices: List[int],
        local_offsets: npt.NDArray | None = None,
    ) -> npt.NDArray:
        """Batch compute position Jacobians for points attached to frames.

        Args:
            qpos: Joint positions
            link_indices: List of frame indices
            local_offsets: (num_links, 3) offsets in each frame's local coordinates.
                If None, uses the frame origins.

        Returns:
            jacobians: (num_links, 3, nq) world position Jacobians
        """
        qpos = self._make_pin_qpos(qpos)

        pin.computeJointJacobians(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)

        _, cross = self._prepare_point_offsets(link_indices, local_offsets)
        local = np.array([
            pin.getFrameJacobian(self.model, self.data, idx, pin.LOCAL)
            for idx in link_indices
        ], dtype=np.float64)
        rotations = np.array([self.data.oMf[idx].rotation for idx in link_indices],
                             dtype=np.float64)
        return rotations @ (local[:, :3] - cross @ local[:, 3:])

    def compute_fk_batch(self, qpos: npt.NDArray, link_indices: List[int]) -> npt.NDArray:
        """Batch compute FK positions for multiple links.

        Args:
            qpos: Joint positions
            link_indices: List of frame indices

        Returns:
            positions: (num_links * 3,) flattened positions
        """
        qpos = self._make_pin_qpos(qpos)
        pin.forwardKinematics(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)

        positions = []
        for idx in link_indices:
            pos = np.array(self.data.oMf[idx].translation, dtype=np.float64, copy=True)
            positions.append(pos)

        return np.concatenate(positions)
