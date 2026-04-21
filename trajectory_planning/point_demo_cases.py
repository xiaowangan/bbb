from typing import List, Tuple

from motion_algorithm.models import PointToPointPlanRequest
from motion_algorithm.planner_core import MotionPlannerPinocchioCore
from motion_algorithm.default_configs import get_default_start_joints


def build_case_translation_only(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, PointToPointPlanRequest]:
    """
    case 0: 纯平移
    - 姿态不变
    - 仅改变末端位置
    """
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    end_pose = core.build_target_pose_by_offset(
        base_pose=start_pose,
        delta_position=[0.02, 0.00, 0.02],
        relative_axis=[0.0, 0.0, 1.0],
        relative_angle_deg=0.0
    )

    request = PointToPointPlanRequest(
        start_pose=start_pose,
        end_pose=end_pose,
        initial_active_joints=start_active_joints,
        motion_time=2.0,
        num_waypoints=8,
        ik_verbose=False
    )
    return "纯平移", request


def build_case_rotation_only(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, PointToPointPlanRequest]:
    """
    case 1: 纯旋转
    - 位置不变
    - 仅改变末端姿态
    """
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    end_pose = core.build_target_pose_by_offset(
        base_pose=start_pose,
        delta_position=[0.0, 0.0, 0.0],
        relative_axis=[0.0, 0.0, 1.0],
        relative_angle_deg=10.0
    )

    request = PointToPointPlanRequest(
        start_pose=start_pose,
        end_pose=end_pose,
        initial_active_joints=start_active_joints,
        motion_time=2.0,
        num_waypoints=8,
        ik_verbose=False
    )
    return "纯旋转", request


def build_case_translation_and_rotation(
    core: MotionPlannerPinocchioCore
) -> Tuple[str, PointToPointPlanRequest]:
    """
    case 2: 平移 + 旋转
    - 同时改变位置和姿态
    """
    start_active_joints = get_default_start_joints()
    start_pose = core.pose_from_active_joints(start_active_joints)

    end_pose = core.build_target_pose_by_offset(
        base_pose=start_pose,
        delta_position=[0.3, 0.3, 0.25],
        relative_axis=[0.0, 0.0, 1.0],
        relative_angle_deg=15.0
    )

    request = PointToPointPlanRequest(
        start_pose=start_pose,
        end_pose=end_pose,
        initial_active_joints=start_active_joints,
        motion_time=2.0,
        num_waypoints=8,
        ik_verbose=False
    )
    return "平移+旋转", request


def build_all_demo_cases(
    core: MotionPlannerPinocchioCore
) -> List[Tuple[str, PointToPointPlanRequest]]:
    """
    返回所有固定实验 case
    """
    return [
        build_case_translation_only(core),
        build_case_rotation_only(core),
        build_case_translation_and_rotation(core),
    ]