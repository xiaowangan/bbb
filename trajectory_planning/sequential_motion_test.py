import argparse

from motion_algorithm.models import MotionConfig, PointToPointPlanRequest
from motion_algorithm.planner_core import MotionPlannerPinocchioCore
from motion_algorithm.PointToPoint_template import PointToPointPoseTemplate
from motion_algorithm.runtime_motion_state import RuntimeMotionState


def build_core():
    config = MotionConfig()
    core = MotionPlannerPinocchioCore(
        config=config,
        joint_names=["j1", "j2", "j3", "j4", "j5", "j6"],
        urdf_path=r"D:\motion_algorithm\trajectory\motion_algorithm_pkg\motion_algorithm\fairino5.urdf",
        dt=0.005,
        end_effector_frame_name="endoscope_tip",
        ik_max_iters=200,
        ik_eps=1e-4,
        ik_damping=1e-6,
        ik_dt=0.2,
        ik_max_step_norm=0.2,
    )
    return core


def print_runtime_state(runtime_state, title):
    print("\n" + "=" * 80)
    print(title)
    print(f"当前主动关节角: {[round(v, 6) for v in runtime_state.current_active_joints]}")
    print(f"当前末端位置  : {[round(v, 6) for v in runtime_state.current_pose.position]}")
    print(f"当前末端四元数: {[round(v, 6) for v in runtime_state.current_pose.quaternion]}")
    print("=" * 80)


def print_trajectory_summary(result):
    print("✅ 轨迹规划成功")
    print(f"轨迹点数: {len(result.trajectory.points)}")
    print(f"终点位置误差: {result.end_pose_report.position_error_norm:.6e} m")
    print(f"终点姿态误差: {result.end_pose_report.orientation_error_deg:.6f} deg")


def plan_relative_motion(
    core,
    ptp_template,
    runtime_state,
    dx,
    dy,
    dz,
    rot_axis=(0.0, 0.0, 1.0),
    rot_deg=0.0,
    motion_time=2.0,
    num_waypoints=8,
):
    start_pose = runtime_state.current_pose
    start_joints = runtime_state.current_active_joints

    end_pose = core.build_target_pose_by_offset(
        base_pose=start_pose,
        delta_position=[dx, dy, dz],
        relative_axis=list(rot_axis),
        relative_angle_deg=rot_deg,
    )

    request = PointToPointPlanRequest(
        start_pose=start_pose,
        end_pose=end_pose,
        initial_active_joints=start_joints,
        motion_time=motion_time,
        num_waypoints=num_waypoints,
        ik_verbose=False,
    )

    result = ptp_template.plan(request)
    runtime_state.update_from_trajectory_end(core, result.trajectory)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx1", type=float, default=0.05)
    parser.add_argument("--dy1", type=float, default=0.00)
    parser.add_argument("--dz1", type=float, default=0.05)

    parser.add_argument("--dx2", type=float, default=0.05)
    parser.add_argument("--dy2", type=float, default=0.05)
    parser.add_argument("--dz2", type=float, default=0.00)

    parser.add_argument("--dx3", type=float, default=-0.05)
    parser.add_argument("--dy3", type=float, default=0.00)
    parser.add_argument("--dz3", type=float, default=0.05)

    parser.add_argument("--rot1", type=float, default=0.0)
    parser.add_argument("--rot2", type=float, default=0.0)
    parser.add_argument("--rot3", type=float, default=0.0)

    parser.add_argument("--motion_time", type=float, default=2.0)
    parser.add_argument("--num_waypoints", type=int, default=8)

    args = parser.parse_args()

    core = build_core()
    ptp_template = PointToPointPoseTemplate(core)
    runtime_state = RuntimeMotionState.build_default(core)

    print_runtime_state(runtime_state, "系统启动后的默认状态")

    motions = [
        ("第1次运动", args.dx1, args.dy1, args.dz1, args.rot1),
        ("第2次运动", args.dx2, args.dy2, args.dz2, args.rot2),
        ("第3次运动", args.dx3, args.dy3, args.dz3, args.rot3),
    ]

    for name, dx, dy, dz, rot_deg in motions:
        print(f"\n{name}")
        print(f"输入增量: dx={dx}, dy={dy}, dz={dz}, rot_deg={rot_deg}")

        print_runtime_state(runtime_state, f"{name} 规划前状态")

        try:
            result = plan_relative_motion(
                core=core,
                ptp_template=ptp_template,
                runtime_state=runtime_state,
                dx=dx,
                dy=dy,
                dz=dz,
                rot_deg=rot_deg,
                motion_time=args.motion_time,
                num_waypoints=args.num_waypoints,
            )
            print_trajectory_summary(result)
            print_runtime_state(runtime_state, f"{name} 规划后状态")

        except Exception as e:
            print(f"❌ {name} 规划失败")
            print("错误信息：", e)
            break

    print_runtime_state(runtime_state, "全部连续运动结束后的最终状态")


if __name__ == "__main__":
    main()