from dataclasses import dataclass
from typing import List

from motion_algorithm.models import PoseData, PlannedTrajectory
from motion_algorithm.default_configs import get_default_start_joints
from motion_algorithm.planner_core import MotionPlannerPinocchioCore


@dataclass
class RuntimeMotionState:
    """
    运行时运动状态：
    - current_active_joints: 当前“认为”的机械臂主动关节角
    - current_pose: 由 current_active_joints 通过 FK 计算出的当前末端位姿

    设计目标：
    1. 系统启动时，以 default_configs 中的默认关节角作为初始状态
    2. 每次规划成功后，用轨迹最后一个点更新当前状态
    3. 下一次规划时，直接从当前状态继续，而不是回到默认位姿
    """
    current_active_joints: List[float]
    current_pose: PoseData

    @classmethod
    def build_default(cls, core: MotionPlannerPinocchioCore) -> "RuntimeMotionState":
        joints = list(get_default_start_joints())
        pose = core.pose_from_active_joints(joints)
        return cls(
            current_active_joints=joints,
            current_pose=pose,
        )

    def reset_to_default(self, core: MotionPlannerPinocchioCore) -> None:
        """
        将当前状态重置为系统默认起始关节角对应的位姿
        """
        joints = list(get_default_start_joints())
        pose = core.pose_from_active_joints(joints)

        self.current_active_joints = joints
        self.current_pose = pose

    def update_from_active_joints(
        self,
        core: MotionPlannerPinocchioCore,
        active_joints: List[float]
    ) -> None:
        """
        直接根据一组主动关节角更新当前状态
        """
        joints = list(active_joints)
        pose = core.pose_from_active_joints(joints)

        self.current_active_joints = joints
        self.current_pose = pose

    def update_from_trajectory_end(
        self,
        core: MotionPlannerPinocchioCore,
        trajectory: PlannedTrajectory
    ) -> None:
        """
        用轨迹最后一个点更新当前状态
        """
        if trajectory is None or len(trajectory.points) == 0:
            raise ValueError("trajectory 为空，无法更新运行时状态")

        end_joints = list(trajectory.points[-1].positions)
        self.update_from_active_joints(core, end_joints)

    def summary_dict(self) -> dict:
        return {
            "current_active_joints": [float(v) for v in self.current_active_joints],
            "current_pose_position": [float(v) for v in self.current_pose.position],
            "current_pose_quaternion": [float(v) for v in self.current_pose.quaternion],
        }