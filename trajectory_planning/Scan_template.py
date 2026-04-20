from typing import List

from motion_algorithm.models import (
    PlannedTrajectory,
    PointToPointPlanRequest,
    PointToPointPlanResult,
    PoseData,
    TrajectoryPointData,
    ScanPlanRequest,
    ScanPlanResult,
    GridScanPlanRequest,
    GridScanPlanResult,
)
from motion_algorithm.planner_core import MotionPlannerPinocchioCore
from motion_algorithm.PointToPoint_template import PointToPointPoseTemplate


class BackAndForthScanPoseTemplate:
    """
    一维往返扫描模板：
    - 固定姿态
    - 沿指定方向往返运动
    - 本质是多段点到点轨迹拼接
    """

    def __init__(self, core: MotionPlannerPinocchioCore):
        self.core = core
        self.ptp_template = PointToPointPoseTemplate(core)

    def _normalize_direction(self, direction: List[float]) -> List[float]:
        if len(direction) != 3:
            raise ValueError("scan_direction 长度必须为 3")

        norm = sum(v * v for v in direction) ** 0.5
        if norm < 1e-12:
            raise ValueError("scan_direction 不能为零向量")

        return [v / norm for v in direction]

    def _offset_pose(
        self,
        base_pose: PoseData,
        direction: List[float],
        distance: float,
    ) -> PoseData:
        return PoseData(
            position=[
                base_pose.position[0] + direction[0] * distance,
                base_pose.position[1] + direction[1] * distance,
                base_pose.position[2] + direction[2] * distance,
            ],
            quaternion=base_pose.quaternion.copy(),
        )

    def _build_scan_key_poses(self, request: ScanPlanRequest) -> List[PoseData]:
        if request.scan_distance <= 0:
            raise ValueError("scan_distance 必须大于 0")
        if request.scan_count <= 0:
            raise ValueError("scan_count 必须大于 0")

        direction = self._normalize_direction(request.scan_direction)

        key_poses = [request.start_pose]

        current_pose = request.start_pose
        forward = True

        for _ in range(request.scan_count):
            if forward:
                next_pose = self._offset_pose(current_pose, direction, request.scan_distance)
            else:
                next_pose = self._offset_pose(current_pose, direction, -request.scan_distance)

            key_poses.append(next_pose)
            current_pose = next_pose
            forward = not forward

        return key_poses

    def _merge_segment_trajectories(
        self,
        segment_trajectories: List[PlannedTrajectory]
    ) -> PlannedTrajectory:
        if len(segment_trajectories) == 0:
            raise ValueError("没有可拼接的分段轨迹")

        merged_points: List[TrajectoryPointData] = []
        global_time_offset = 0.0

        for seg_idx, traj in enumerate(segment_trajectories):
            if len(traj.points) == 0:
                continue

            for pt_idx, pt in enumerate(traj.points):
                if seg_idx > 0 and pt_idx == 0:
                    # 跳过重复段首点，避免拼接处重复
                    continue

                merged_points.append(
                    TrajectoryPointData(
                        positions=pt.positions.copy(),
                        time_from_start=pt.time_from_start + global_time_offset
                    )
                )

            segment_end_time = traj.points[-1].time_from_start
            global_time_offset += segment_end_time

        return PlannedTrajectory(
            joint_names=segment_trajectories[0].joint_names.copy(),
            points=merged_points
        )

    def plan(self, request: ScanPlanRequest) -> ScanPlanResult:
        if request.motion_time_per_segment <= 0:
            raise ValueError("motion_time_per_segment 必须大于 0")
        if request.num_waypoints_per_segment < 2:
            raise ValueError("num_waypoints_per_segment 至少为 2")

        key_poses = self._build_scan_key_poses(request)

        segment_results: List[PointToPointPlanResult] = []
        segment_trajectories: List[PlannedTrajectory] = []

        last_joints = request.initial_active_joints

        for i in range(len(key_poses) - 1):
            segment_request = PointToPointPlanRequest(
                start_pose=key_poses[i],
                end_pose=key_poses[i + 1],
                initial_active_joints=last_joints,
                motion_time=request.motion_time_per_segment,
                num_waypoints=request.num_waypoints_per_segment,
                ik_verbose=request.ik_verbose,
            )

            result = self.ptp_template.plan(segment_request)
            segment_results.append(result)
            segment_trajectories.append(result.trajectory)

            last_joints = result.trajectory.points[-1].positions

        merged_trajectory = self._merge_segment_trajectories(segment_trajectories)

        return ScanPlanResult(
            trajectory=merged_trajectory,
            segment_results=segment_results,
            key_poses=key_poses,
        )


class GridScanPoseTemplate:
    """
    二维栅格扫描模板：
    - 固定姿态
    - 每一行沿 line_direction 扫描
    - 行与行之间沿 step_direction 步进
    - 相邻行采用蛇形/折返方式连接
    - 本质仍然是多段点到点轨迹拼接
    """

    def __init__(self, core: MotionPlannerPinocchioCore):
        self.core = core
        self.ptp_template = PointToPointPoseTemplate(core)

    def _normalize_direction(self, direction: List[float], name: str) -> List[float]:
        if len(direction) != 3:
            raise ValueError(f"{name} 长度必须为 3")

        norm = sum(v * v for v in direction) ** 0.5
        if norm < 1e-12:
            raise ValueError(f"{name} 不能为零向量")

        return [v / norm for v in direction]

    def _offset_pose(
        self,
        base_pose: PoseData,
        direction: List[float],
        distance: float,
    ) -> PoseData:
        return PoseData(
            position=[
                base_pose.position[0] + direction[0] * distance,
                base_pose.position[1] + direction[1] * distance,
                base_pose.position[2] + direction[2] * distance,
            ],
            quaternion=base_pose.quaternion.copy(),
        )

    def _merge_segment_trajectories(
        self,
        segment_trajectories: List[PlannedTrajectory]
    ) -> PlannedTrajectory:
        if len(segment_trajectories) == 0:
            raise ValueError("没有可拼接的分段轨迹")

        merged_points: List[TrajectoryPointData] = []
        global_time_offset = 0.0

        for seg_idx, traj in enumerate(segment_trajectories):
            if len(traj.points) == 0:
                continue

            for pt_idx, pt in enumerate(traj.points):
                if seg_idx > 0 and pt_idx == 0:
                    continue

                merged_points.append(
                    TrajectoryPointData(
                        positions=pt.positions.copy(),
                        time_from_start=pt.time_from_start + global_time_offset
                    )
                )

            segment_end_time = traj.points[-1].time_from_start
            global_time_offset += segment_end_time

        return PlannedTrajectory(
            joint_names=segment_trajectories[0].joint_names.copy(),
            points=merged_points
        )

    def _build_grid_key_poses(self, request: GridScanPlanRequest) -> List[PoseData]:
        if request.line_length <= 0:
            raise ValueError("line_length 必须大于 0")
        if request.line_count <= 0:
            raise ValueError("line_count 必须大于 0")
        if request.line_count > 1 and request.step_distance <= 0:
            raise ValueError("当 line_count > 1 时，step_distance 必须大于 0")

        line_dir = self._normalize_direction(request.line_direction, "line_direction")
        step_dir = self._normalize_direction(request.step_direction, "step_direction")

        key_poses: List[PoseData] = [request.start_pose]
        current_row_start = request.start_pose

        for row_idx in range(request.line_count):
            line_sign = 1.0 if (row_idx % 2 == 0) else -1.0

            row_end_pose = self._offset_pose(
                current_row_start,
                line_dir,
                line_sign * request.line_length
            )
            key_poses.append(row_end_pose)

            if row_idx == request.line_count - 1:
                break

            next_row_start = self._offset_pose(
                row_end_pose,
                step_dir,
                request.step_distance
            )
            key_poses.append(next_row_start)

            current_row_start = next_row_start

        return key_poses

    def plan(self, request: GridScanPlanRequest) -> GridScanPlanResult:
        if request.motion_time_per_segment <= 0:
            raise ValueError("motion_time_per_segment 必须大于 0")
        if request.num_waypoints_per_segment < 2:
            raise ValueError("num_waypoints_per_segment 至少为 2")

        key_poses = self._build_grid_key_poses(request)

        segment_results: List[PointToPointPlanResult] = []
        segment_trajectories: List[PlannedTrajectory] = []

        last_joints = request.initial_active_joints

        for i in range(len(key_poses) - 1):
            segment_request = PointToPointPlanRequest(
                start_pose=key_poses[i],
                end_pose=key_poses[i + 1],
                initial_active_joints=last_joints,
                motion_time=request.motion_time_per_segment,
                num_waypoints=request.num_waypoints_per_segment,
                ik_verbose=request.ik_verbose,
            )

            result = self.ptp_template.plan(segment_request)
            segment_results.append(result)
            segment_trajectories.append(result.trajectory)

            last_joints = result.trajectory.points[-1].positions

        merged_trajectory = self._merge_segment_trajectories(segment_trajectories)

        return GridScanPlanResult(
            trajectory=merged_trajectory,
            segment_results=segment_results,
            key_poses=key_poses,
        )