from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MotionConfig:
    deadzone: float = 1.0
    gain: float = 0.3
    max_cmd: float = 5.0
    alpha: float = 0.4
    max_step: float = 2.0


@dataclass
class PoseData:
    position: List[float]
    quaternion: List[float]   # [w, x, y, z]


@dataclass
class TrajectoryPointData:
    positions: List[float]
    time_from_start: float


@dataclass
class PoseErrorReport:
    fk_position: List[float]
    fk_quaternion: List[float]
    position_error_norm: float
    orientation_error_deg: float

@dataclass
class IKDebugInfo:
    success: bool
    iterations: int
    final_error_norm: float
    final_position_error_norm: float
    final_orientation_error_deg: float
    target_position: List[float]
    target_quaternion: List[float]
    final_fk_position: List[float]
    final_fk_quaternion: List[float]
    final_active_joints: Optional[List[float]]
    hit_joint_limit: bool
    message: str


@dataclass
class PlannedTrajectory:
    joint_names: List[str]
    points: List[TrajectoryPointData]


@dataclass
class PointToPointPlanRequest:
    start_pose: PoseData
    end_pose: PoseData
    initial_active_joints: Optional[List[float]]
    motion_time: float = 2.0
    num_waypoints: int = 8
    ik_verbose: bool = False


@dataclass
class PointToPointPlanResult:
    trajectory: PlannedTrajectory
    start_pose_report: PoseErrorReport
    end_pose_report: PoseErrorReport


@dataclass
class ScanPlanRequest:
    """
    一维往返扫描模板请求参数
    """
    start_pose: PoseData
    initial_active_joints: List[float]
    scan_direction: List[float]
    scan_distance: float
    scan_count: int
    motion_time_per_segment: float
    num_waypoints_per_segment: int
    ik_verbose: bool = False


@dataclass
class ScanPlanResult:
    """
    一维往返扫描模板输出结果
    """
    trajectory: PlannedTrajectory
    segment_results: List[PointToPointPlanResult]
    key_poses: List[PoseData]


@dataclass
class GridScanPlanRequest:
    """
    二维栅格扫描模板请求参数
    - line_direction: 每一行的扫描方向
    - line_length: 每一行的扫描长度
    - line_count: 扫描总行数
    - step_direction: 行与行之间的步进方向
    - step_distance: 相邻两行之间的间距
    """
    start_pose: PoseData
    initial_active_joints: List[float]
    line_direction: List[float]
    line_length: float
    line_count: int
    step_direction: List[float]
    step_distance: float
    motion_time_per_segment: float
    num_waypoints_per_segment: int
    ik_verbose: bool = False


@dataclass
class GridScanPlanResult:
    """
    二维栅格扫描模板输出结果
    """
    trajectory: PlannedTrajectory
    segment_results: List[PointToPointPlanResult]
    key_poses: List[PoseData]