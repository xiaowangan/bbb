from typing import List, Tuple

from motion_algorithm.planner_core import MotionPlannerPinocchioCore
from motion_algorithm.models import ScanPlanRequest, GridScanPlanRequest
from motion_algorithm.default_configs import get_default_start_joints


# =========================
# 一维往返扫描 demo
# =========================

def build_case_scan_x_3times(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, ScanPlanRequest]:
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    request = ScanPlanRequest(
        start_pose=start_pose,
        initial_active_joints=start_active_joints,
        scan_direction=[1.0, 0.0, 0.0],
        scan_distance=0.015,
        scan_count=3,
        motion_time_per_segment=1.5,
        num_waypoints_per_segment=8,
        ik_verbose=False,
    )
    return "X方向往返扫描3次", request


def build_case_scan_x_5times(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, ScanPlanRequest]:
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    request = ScanPlanRequest(
        start_pose=start_pose,
        initial_active_joints=start_active_joints,
        scan_direction=[1.0, 0.0, 0.0],
        scan_distance=0.015,
        scan_count=5,
        motion_time_per_segment=1.5,
        num_waypoints_per_segment=8,
        ik_verbose=False,
    )
    return "X方向往返扫描5次", request


def build_case_scan_y_3times(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, ScanPlanRequest]:
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    request = ScanPlanRequest(
        start_pose=start_pose,
        initial_active_joints=start_active_joints,
        scan_direction=[0.0, 1.0, 0.0],
        scan_distance=0.01,
        scan_count=3,
        motion_time_per_segment=1.5,
        num_waypoints_per_segment=8,
        ik_verbose=False,
    )
    return "Y方向往返扫描3次", request


def build_case_scan_z_small(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, ScanPlanRequest]:
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    request = ScanPlanRequest(
        start_pose=start_pose,
        initial_active_joints=start_active_joints,
        scan_direction=[0.0, 0.0, 1.0],
        scan_distance=0.005,
        scan_count=3,
        motion_time_per_segment=1.5,
        num_waypoints_per_segment=8,
        ik_verbose=False,
    )
    return "Z方向小幅往返扫描3次", request


def build_case_scan_x_large(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, ScanPlanRequest]:
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    request = ScanPlanRequest(
        start_pose=start_pose,
        initial_active_joints=start_active_joints,
        scan_direction=[1.0, 0.0, 0.0],
        scan_distance=0.02,
        scan_count=3,
        motion_time_per_segment=1.5,
        num_waypoints_per_segment=8,
        ik_verbose=False,
    )
    return "X方向大幅往返扫描3次", request


def build_all_scan_demo_cases(
    core: MotionPlannerPinocchioCore
) -> List[Tuple[str, ScanPlanRequest]]:
    return [
        build_case_scan_x_3times(core),
        build_case_scan_x_5times(core),
        build_case_scan_y_3times(core),
        build_case_scan_z_small(core),
        build_case_scan_x_large(core),
    ]


# =========================
# 二维栅格扫描 demo
# =========================

def build_case_grid_scan_xy_3rows(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, GridScanPlanRequest]:
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    request = GridScanPlanRequest(
        start_pose=start_pose,
        initial_active_joints=start_active_joints,
        line_direction=[1.0, 0.0, 0.0],
        line_length=0.02,
        line_count=3,
        step_direction=[0.0, 1.0, 0.0],
        step_distance=0.005,
        motion_time_per_segment=1.2,
        num_waypoints_per_segment=8,
        ik_verbose=False,
    )
    return "XY平面栅格扫描3行", request


def build_case_grid_scan_xy_4rows(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, GridScanPlanRequest]:
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    request = GridScanPlanRequest(
        start_pose=start_pose,
        initial_active_joints=start_active_joints,
        line_direction=[1.0, 0.0, 0.0],
        line_length=0.018,
        line_count=4,
        step_direction=[0.0, 1.0, 0.0],
        step_distance=0.005,
        motion_time_per_segment=1.0,
        num_waypoints_per_segment=8,
        ik_verbose=False,
    )
    return "XY平面栅格扫描4行", request


def build_case_grid_scan_xz_3rows(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, GridScanPlanRequest]:
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    request = GridScanPlanRequest(
        start_pose=start_pose,
        initial_active_joints=start_active_joints,
        line_direction=[1.0, 0.0, 0.0],
        line_length=0.015,
        line_count=3,
        step_direction=[0.0, 0.0, 1.0],
        step_distance=0.005,
        motion_time_per_segment=1.0,
        num_waypoints_per_segment=8,
        ik_verbose=False,
    )
    return "XZ平面栅格扫描3行", request


def build_all_grid_scan_demo_cases(
    core: MotionPlannerPinocchioCore
) -> List[Tuple[str, GridScanPlanRequest]]:
    return [
        build_case_grid_scan_xy_3rows(core),
        build_case_grid_scan_xy_4rows(core),
        build_case_grid_scan_xz_3rows(core),
    ]