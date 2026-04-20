from typing import List, Optional, Tuple
import math
import os

import numpy as np
import pinocchio as pin

from motion_algorithm.models import (
    MotionConfig,
    PoseData,
    PoseErrorReport,
    PlannedTrajectory,
    TrajectoryPointData,
    IKDebugInfo,
)


class MotionPlannerPinocchioCore:
    """
    基础能力层：
    - FK
    - IK
    - 位姿插值
    - 关节三次插值
    - 位姿误差评估
    """

    def __init__(
        self,
        config: MotionConfig,
        joint_names: List[str],
        urdf_path: str,
        dt: float = 0.005,
        end_effector_frame_name: Optional[str] = None,
        ik_max_iters: int = 200,
        ik_eps: float = 1e-4,
        ik_damping: float = 1e-6,
        ik_dt: float = 0.2,
        ik_max_step_norm: float = 0.2,
    ):
        self.config = config
        self.joint_names = joint_names
        self.dt = dt

        self.ik_max_iters = ik_max_iters
        self.ik_eps = ik_eps
        self.ik_damping = ik_damping
        self.ik_dt = ik_dt
        self.ik_max_step_norm = ik_max_step_norm

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self.data = self.model.createData()

        if self.model.nq < len(self.joint_names) or self.model.nv < len(self.joint_names):
            raise ValueError(
                f"Pinocchio模型自由度不足: nq={self.model.nq}, nv={self.model.nv}, "
                f"joint_names={len(self.joint_names)}"
            )

        self.active_dof = len(self.joint_names)
        self.lower_limits = np.array(self.model.lowerPositionLimit[:self.active_dof], dtype=float)
        self.upper_limits = np.array(self.model.upperPositionLimit[:self.active_dof], dtype=float)

        self.end_effector_frame_id, self.end_effector_frame_name = self._resolve_end_effector_frame(
            end_effector_frame_name
        )

    # =========================
    # 模型辅助
    # =========================

    def _resolve_end_effector_frame(self, frame_name: Optional[str]) -> Tuple[int, str]:
        if frame_name is not None:
            frame_names = [frame.name for frame in self.model.frames]
            frame_id = self.model.getFrameId(frame_name)

            if frame_id >= len(self.model.frames) or frame_names[frame_id] != frame_name:
                raise ValueError(
                    f"指定的末端frame不存在: {frame_name}\n"
                    f"当前可用frames:\n" + "\n".join(frame_names)
                )

            return frame_id, frame_name

        candidate_id = None
        candidate_name = None
        for i, frame in enumerate(self.model.frames):
            if getattr(frame, "parentJoint", 0) > 0:
                candidate_id = i
                candidate_name = frame.name

        if candidate_id is None:
            raise ValueError("未能自动识别末端 frame，请手动指定 end_effector_frame_name")

        return candidate_id, candidate_name

    def get_all_joint_names(self) -> List[str]:
        return list(self.model.names)

    def get_all_frame_names(self) -> List[str]:
        return [frame.name for frame in self.model.frames]

    def print_model_info(self):
        print("\n" + "=" * 80)
        print("Pinocchio 模型信息")
        print("=" * 80)
        print(f"URDF路径: {self.urdf_path}")
        print(f"nq={self.model.nq}, nv={self.model.nv}, active_dof={self.active_dof}")
        print(f"当前末端frame: {self.end_effector_frame_name} (id={self.end_effector_frame_id})")

        print("\n---- Joint Names ----")
        for i, name in enumerate(self.model.names):
            print(f"{i:2d}: {name}")

        print("\n---- Frame Names ----")
        for i, frame in enumerate(self.model.frames):
            print(f"{i:2d}: {frame.name}")

        print("=" * 80)

    def _clamp_q(self, q: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=float).copy()
        q[:self.active_dof] = np.clip(q[:self.active_dof], self.lower_limits, self.upper_limits)
        return q

    def _active_to_full_q(self, active_joints: Optional[List[float]]) -> np.ndarray:
        q = pin.neutral(self.model)

        if active_joints is None:
            return self._clamp_q(np.array(q, dtype=float))

        active_np = np.array(active_joints, dtype=float)
        if active_np.shape[0] != self.active_dof:
            raise ValueError(
                f"主动关节长度错误，应为 {self.active_dof}，当前为 {active_np.shape[0]}"
            )

        q = np.array(q, dtype=float)
        q[:self.active_dof] = active_np
        return self._clamp_q(q)

    def _full_q_to_active(self, q: np.ndarray) -> List[float]:
        return np.array(q[:self.active_dof], dtype=float).tolist()

    def _check_hit_joint_limit(self, q: np.ndarray, tol: float = 1e-3) -> bool:
        active_q = np.array(q[:self.active_dof], dtype=float)
        near_lower = np.any(np.abs(active_q - self.lower_limits) < tol)
        near_upper = np.any(np.abs(active_q - self.upper_limits) < tol)
        return bool(near_lower or near_upper)

    # =========================
    # 四元数工具
    # =========================

    def _normalize_quaternion(self, quat: List[float]) -> np.ndarray:
        q = np.array(quat, dtype=float)
        if q.shape != (4,):
            raise ValueError(f"四元数长度必须为4，当前输入: {quat}")

        norm = np.linalg.norm(q)
        if norm < 1e-12:
            raise ValueError("四元数范数过小，无法归一化")

        return q / norm

    def quaternion_to_rotation_matrix(self, quat: List[float]) -> np.ndarray:
        w, x, y, z = self._normalize_quaternion(quat)

        return np.array([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)]
        ], dtype=float)

    def rotation_matrix_to_quaternion(self, R: np.ndarray) -> List[float]:
        R = np.array(R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError(f"旋转矩阵尺寸必须为3x3，当前为: {R.shape}")

        trace = np.trace(R)
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return self._normalize_quaternion([w, x, y, z]).tolist()

    def axis_angle_to_quaternion(self, axis: List[float], angle_deg: float) -> List[float]:
        axis_np = np.array(axis, dtype=float)
        if axis_np.shape != (3,):
            raise ValueError(f"旋转轴长度必须为3，当前输入: {axis}")

        axis_norm = np.linalg.norm(axis_np)
        if axis_norm < 1e-12:
            raise ValueError("旋转轴范数过小，无法构造四元数")

        axis_unit = axis_np / axis_norm
        half_rad = math.radians(angle_deg) / 2.0
        w = math.cos(half_rad)
        xyz = axis_unit * math.sin(half_rad)
        return self._normalize_quaternion([w, xyz[0], xyz[1], xyz[2]]).tolist()

    def quaternion_multiply(self, q1: List[float], q2: List[float]) -> List[float]:
        w1, x1, y1, z1 = self._normalize_quaternion(q1)
        w2, x2, y2, z2 = self._normalize_quaternion(q2)

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        return self._normalize_quaternion([w, x, y, z]).tolist()

    def slerp_quaternion(self, q0: List[float], q1: List[float], s: float) -> np.ndarray:
        if s < 0.0 or s > 1.0:
            raise ValueError("插值系数 s 必须在 [0, 1] 内")

        qa = self._normalize_quaternion(q0)
        qb = self._normalize_quaternion(q1)

        dot = float(np.dot(qa, qb))
        if dot < 0.0:
            qb = -qb
            dot = -dot

        if dot > 0.9995:
            q = qa + s * (qb - qa)
            return q / np.linalg.norm(q)

        theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = math.sin(theta_0)

        theta = theta_0 * s
        sin_theta = math.sin(theta)

        coeff0 = math.sin(theta_0 - theta) / sin_theta_0
        coeff1 = sin_theta / sin_theta_0

        q = coeff0 * qa + coeff1 * qb
        return q / np.linalg.norm(q)

    # =========================
    # FK / 误差
    # =========================

    def _make_target_se3(self, position: List[float], quaternion: List[float]) -> pin.SE3:
        pos = np.array(position, dtype=float)
        if pos.shape != (3,):
            raise ValueError(f"位置长度必须为3，当前输入: {position}")
        R = self.quaternion_to_rotation_matrix(quaternion)
        return pin.SE3(R, pos)

    def forward_kinematics_from_active_joints(self, active_joints: List[float]) -> np.ndarray:
        q = self._active_to_full_q(active_joints)
        pin.framesForwardKinematics(self.model, self.data, q)
        T = self.data.oMf[self.end_effector_frame_id]

        T_mat = np.eye(4, dtype=float)
        T_mat[:3, :3] = np.array(T.rotation, dtype=float)
        T_mat[:3, 3] = np.array(T.translation, dtype=float)
        return T_mat

    def pose_from_active_joints(self, active_joints: List[float]) -> PoseData:
        fk_T = self.forward_kinematics_from_active_joints(active_joints)
        return PoseData(
            position=fk_T[:3, 3].tolist(),
            quaternion=self.rotation_matrix_to_quaternion(fk_T[:3, :3])
        )

    def _rotation_angle_between_matrices_deg(self, R_target: np.ndarray, R_fk: np.ndarray) -> float:
        R_err = R_target.T @ R_fk
        cos_theta = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
        return math.degrees(math.acos(cos_theta))

    def evaluate_pose_error(
        self,
        active_joints: List[float],
        target_position: List[float],
        target_quaternion: List[float]
    ) -> PoseErrorReport:
        fk_T = self.forward_kinematics_from_active_joints(active_joints)
        fk_position = fk_T[:3, 3]
        fk_rotation = fk_T[:3, :3]

        target_position_np = np.array(target_position, dtype=float)
        target_rotation = self.quaternion_to_rotation_matrix(target_quaternion)

        return PoseErrorReport(
            fk_position=fk_position.tolist(),
            fk_quaternion=self.rotation_matrix_to_quaternion(fk_rotation),
            position_error_norm=float(np.linalg.norm(fk_position - target_position_np)),
            orientation_error_deg=float(
                self._rotation_angle_between_matrices_deg(target_rotation, fk_rotation)
            ),
        )

    # =========================
    # IK
    # =========================

    def solve_inverse_kinematics_pose(
        self,
        target_position: List[float],
        target_quaternion: List[float],
        initial_active_joints: Optional[List[float]] = None,
        verbose: bool = False,
    ) -> Optional[List[float]]:
        joints, _ = self.solve_inverse_kinematics_pose_with_debug(
            target_position=target_position,
            target_quaternion=target_quaternion,
            initial_active_joints=initial_active_joints,
            verbose=verbose,
        )
        return joints

    def solve_inverse_kinematics_pose_with_debug(
        self,
        target_position: List[float],
        target_quaternion: List[float],
        initial_active_joints: Optional[List[float]] = None,
        verbose: bool = False,
    ):
        q = self._active_to_full_q(initial_active_joints)
        target_se3 = self._make_target_se3(target_position, target_quaternion)

        success = False
        final_iter = 0
        final_err_norm = float("inf")
        message = "IK未开始"

        for i in range(self.ik_max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.end_effector_frame_id)
            current_se3 = self.data.oMf[self.end_effector_frame_id]

            iMd = current_se3.actInv(target_se3)
            err = pin.log6(iMd).vector
            err_norm = float(np.linalg.norm(err))

            final_iter = i + 1
            final_err_norm = err_norm

            if err_norm < self.ik_eps:
                success = True
                message = "IK收敛成功"
                break

            J = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                self.end_effector_frame_id,
                pin.LOCAL
            )

            try:
                Jlog = pin.Jlog6(iMd.inverse())
                J = -Jlog @ J
            except Exception:
                pass

            try:
                JJt = J @ J.T + self.ik_damping * np.eye(6)
                dq = -J.T @ np.linalg.solve(JJt, err)
            except np.linalg.LinAlgError:
                message = "雅可比求解失败，矩阵可能接近奇异"
                break

            step_norm = float(np.linalg.norm(dq))
            if step_norm > self.ik_max_step_norm:
                dq = dq * (self.ik_max_step_norm / step_norm)

            q = pin.integrate(self.model, q, dq * self.ik_dt)
            q = self._clamp_q(q)

            if verbose and (i % 10 == 0):
                print(f"[IK DEBUG] iter={i:03d}, err_norm={err_norm:.6e}")

        if not success and message == "IK未开始":
            message = f"达到最大迭代次数 {self.ik_max_iters} 后仍未收敛"

        final_active_joints = self._full_q_to_active(q)

        pose_report = self.evaluate_pose_error(
            active_joints=final_active_joints,
            target_position=target_position,
            target_quaternion=target_quaternion
        )

        debug_info = IKDebugInfo(
            success=success,
            iterations=final_iter,
            final_error_norm=final_err_norm,
            final_position_error_norm=pose_report.position_error_norm,
            final_orientation_error_deg=pose_report.orientation_error_deg,
            target_position=list(target_position),
            target_quaternion=list(target_quaternion),
            final_fk_position=pose_report.fk_position,
            final_fk_quaternion=pose_report.fk_quaternion,
            final_active_joints=final_active_joints,
            hit_joint_limit=self._check_hit_joint_limit(q),
            message=message,
        )

        if success:
            return final_active_joints, debug_info

        return None, debug_info

    # =========================
    # 插值
    # =========================

    def interpolate_cartesian_pose_line(
        self,
        start_pose: PoseData,
        end_pose: PoseData,
        num_waypoints: int
    ) -> List[PoseData]:
        if num_waypoints < 2:
            raise ValueError("num_waypoints 至少为 2")

        start_pos_np = np.array(start_pose.position, dtype=float)
        end_pos_np = np.array(end_pose.position, dtype=float)

        points = []
        for i in range(num_waypoints):
            s = i / (num_waypoints - 1)
            pos = (1.0 - s) * start_pos_np + s * end_pos_np
            quat = self.slerp_quaternion(start_pose.quaternion, end_pose.quaternion, s)

            points.append(PoseData(position=pos.tolist(), quaternion=quat.tolist()))

        return points

    def generate_cubic_segment(
        self,
        start_q: List[float],
        end_q: List[float],
        segment_duration: float,
        include_start: bool = True
    ) -> List[TrajectoryPointData]:
        """
        这里虽然函数名仍叫 generate_cubic_segment，
        但实现已经替换为“五次多项式插值”。

        边界条件：
        - 起点位置 = start_q
        - 终点位置 = end_q
        - 起点速度 = 0
        - 终点速度 = 0
        - 起点加速度 = 0
        - 终点加速度 = 0
        """
        if segment_duration <= 0:
            raise ValueError("segment_duration 必须大于 0")

        start_np = np.array(start_q, dtype=float)
        end_np = np.array(end_q, dtype=float)

        if start_np.shape != end_np.shape:
            raise ValueError(
                f"start_q 与 end_q 维度不一致: {start_np.shape} vs {end_np.shape}"
            )

        delta_q = end_np - start_np
        T = float(segment_duration)

        # 五次多项式系数（零速度、零加速度边界）
        # q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        a0 = start_np
        a1 = np.zeros_like(start_np)
        a2 = np.zeros_like(start_np)
        a3 = 10.0 * delta_q / (T ** 3)
        a4 = -15.0 * delta_q / (T ** 4)
        a5 = 6.0 * delta_q / (T ** 5)

        points = []
        num_steps = int(np.floor(T / self.dt))
        start_index = 0 if include_start else 1

        for i in range(start_index, num_steps + 1):
            t = min(i * self.dt, T)
            q = (
                a0
                + a1 * t
                + a2 * (t ** 2)
                + a3 * (t ** 3)
                + a4 * (t ** 4)
                + a5 * (t ** 5)
            )
            points.append(
                TrajectoryPointData(
                    positions=q.tolist(),
                    time_from_start=t
                )
            )

        # 确保最后一个点精确落在终点时刻 T
        if len(points) == 0 or abs(points[-1].time_from_start - T) > 1e-9:
            t = T
            q = (
                a0
                + a1 * t
                + a2 * (t ** 2)
                + a3 * (t ** 3)
                + a4 * (t ** 4)
                + a5 * (t ** 5)
            )
            points.append(
                TrajectoryPointData(
                    positions=q.tolist(),
                    time_from_start=t
                )
            )

        return points

    # =========================
    # 通用辅助
    # =========================

    def build_target_pose_by_offset(
        self,
        base_pose: PoseData,
        delta_position: List[float],
        relative_axis: List[float],
        relative_angle_deg: float
    ) -> PoseData:
        if len(delta_position) != 3:
            raise ValueError("delta_position 长度必须为3")

        end_position = [
            base_pose.position[0] + delta_position[0],
            base_pose.position[1] + delta_position[1],
            base_pose.position[2] + delta_position[2]
        ]

        relative_rotation_quaternion = self.axis_angle_to_quaternion(
            axis=relative_axis,
            angle_deg=relative_angle_deg
        )
        end_quaternion = self.quaternion_multiply(
            relative_rotation_quaternion,
            base_pose.quaternion
        )

        return PoseData(position=end_position, quaternion=end_quaternion)