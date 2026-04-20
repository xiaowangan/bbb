from typing import List, Optional

from motion_algorithm.models import (
    PlannedTrajectory,
    PointToPointPlanRequest,
    PointToPointPlanResult,
    PoseData,
    TrajectoryPointData,
)
from motion_algorithm.planner_core import MotionPlannerPinocchioCore


class PointToPointPoseTemplate:
    """
    单段模板层：
    点到点位姿直线运动模板
    """

    def __init__(self, core: MotionPlannerPinocchioCore):
        self.core = core

    def _plan_joint_trajectory_from_pose_path(
        self,
        pose_path: List[PoseData],
        motion_time: float,
        initial_active_joints: Optional[List[float]],
        ik_verbose: bool,
    ) -> PlannedTrajectory:
        ik_joint_points: List[List[float]] = []
        last_joints = initial_active_joints

        for i, pose in enumerate(pose_path):
            debug_info = None
            
            if i == 0 and initial_active_joints is not None:
                joints = initial_active_joints.copy()
            else:
                joints, debug_info = self.core.solve_inverse_kinematics_pose_with_debug(
                    target_position=pose.position,
                    target_quaternion=pose.quaternion,
                    initial_active_joints=last_joints,
                    verbose=ik_verbose,
                )

            if joints is None:
                raise ValueError(
                    f"第 {i} 个位姿路径点 IK 求解失败\n"
                    f"- 目标位置: {pose.position}\n"
                    f"- 目标四元数: {pose.quaternion}\n"
                    f"- 迭代次数: {debug_info.iterations}\n"
                    f"- 最终总误差: {debug_info.final_error_norm:.6e}\n"
                    f"- 最终位置误差: {debug_info.final_position_error_norm:.6e} m\n"
                    f"- 最终姿态误差: {debug_info.final_orientation_error_deg:.6f} deg\n"
                    f"- 是否接近关节限位: {debug_info.hit_joint_limit}\n"
                    f"- 最终FK位置: {debug_info.final_fk_position}\n"
                    f"- 最终FK四元数: {debug_info.final_fk_quaternion}\n"
                    f"- 最终关节角: {debug_info.final_active_joints}\n"
                    f"- 原因: {debug_info.message}"
                )

            ik_joint_points.append(joints)
            last_joints = joints

        full_points: List[TrajectoryPointData] = []
        segment_duration = motion_time / (len(ik_joint_points) - 1)
        global_time_offset = 0.0

        for seg_idx in range(len(ik_joint_points) - 1):
            q0 = ik_joint_points[seg_idx]
            q1 = ik_joint_points[seg_idx + 1]

            segment_points = self.core.generate_cubic_segment(
                start_q=q0,
                end_q=q1,
                segment_duration=segment_duration,
                include_start=(seg_idx == 0),
            )

            for pt in segment_points:
                full_points.append(
                    TrajectoryPointData(
                        positions=pt.positions,
                        time_from_start=pt.time_from_start + global_time_offset
                    )
                )

            global_time_offset += segment_duration

        if len(full_points) > 0:
            full_points[-1].time_from_start = motion_time

        return PlannedTrajectory(
            joint_names=self.core.joint_names.copy(),
            points=full_points
        )

    def plan(self, request: PointToPointPlanRequest) -> PointToPointPlanResult:
        if request.motion_time <= 0:
            raise ValueError("motion_time 必须大于 0")
        if request.num_waypoints < 2:
            raise ValueError("num_waypoints 至少为 2")

        pose_path = self.core.interpolate_cartesian_pose_line(
            start_pose=request.start_pose,
            end_pose=request.end_pose,
            num_waypoints=request.num_waypoints
        )

        trajectory = self._plan_joint_trajectory_from_pose_path(
            pose_path=pose_path,
            motion_time=request.motion_time,
            initial_active_joints=request.initial_active_joints,
            ik_verbose=request.ik_verbose,
        )

        if len(trajectory.points) == 0:
            raise ValueError("规划结果为空，未生成任何轨迹点。")

        start_joint_est = trajectory.points[0].positions
        end_joint_est = trajectory.points[-1].positions

        start_report = self.core.evaluate_pose_error(
            active_joints=start_joint_est,
            target_position=request.start_pose.position,
            target_quaternion=request.start_pose.quaternion,
        )
        end_report = self.core.evaluate_pose_error(
            active_joints=end_joint_est,
            target_position=request.end_pose.position,
            target_quaternion=request.end_pose.quaternion,
        )

        return PointToPointPlanResult(
            trajectory=trajectory,
            start_pose_report=start_report,
            end_pose_report=end_report,
        )