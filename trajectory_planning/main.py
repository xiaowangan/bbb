import argparse

from motion_algorithm.models import (
    MotionConfig,
    PointToPointPlanRequest,
    ScanPlanRequest,
    GridScanPlanRequest,
)
from motion_algorithm.planner_core import MotionPlannerPinocchioCore
from motion_algorithm.PointToPoint_template import PointToPointPoseTemplate
from motion_algorithm.Scan_template import (
    BackAndForthScanPoseTemplate,
    GridScanPoseTemplate,
)
from motion_algorithm.point_demo_cases import build_all_demo_cases
from motion_algorithm.scan_demo_cases import (
    build_all_scan_demo_cases,
    build_all_grid_scan_demo_cases,
)
from motion_algorithm.runtime_motion_state import RuntimeMotionState


def print_trajectory_summary(trajectory):
    print("✅ 轨迹规划成功！点数：", len(trajectory.points))
    print("=" * 60)
    print("📌 完整轨迹点列表：")
    for i, point in enumerate(trajectory.points):
        t = point.time_from_start
        positions_fmt = ", ".join(f"{v:.3f}" for v in point.positions)
        print(f"点 [{i:2d}] | 时间: {t:.2f}s | 关节角: [{positions_fmt}]")
    print("=" * 60)


def print_validation_summary(request, result):
    def fmt_vec(vec):
        return "[" + ", ".join(f"{v:.6f}" for v in vec) + "]"

    print("🔍 开始进行 FK 回代验证...")

    print("-" * 60)
    print("【起点 FK 验证】")
    print(f"目标位置: {fmt_vec(request.start_pose.position)}")
    print(f"FK位置  : {fmt_vec(result.start_pose_report.fk_position)}")
    print(f"目标四元数: {fmt_vec(request.start_pose.quaternion)}")
    print(f"FK四元数  : {fmt_vec(result.start_pose_report.fk_quaternion)}")
    print(f"位置误差范数: {result.start_pose_report.position_error_norm:.6e} m")
    print(f"姿态误差角  : {result.start_pose_report.orientation_error_deg:.6f} deg")

    print("-" * 60)
    print("【终点 FK 验证】")
    print(f"目标位置: {fmt_vec(request.end_pose.position)}")
    print(f"FK位置  : {fmt_vec(result.end_pose_report.fk_position)}")
    print(f"目标四元数: {fmt_vec(request.end_pose.quaternion)}")
    print(f"FK四元数  : {fmt_vec(result.end_pose_report.fk_quaternion)}")
    print(f"位置误差范数: {result.end_pose_report.position_error_norm:.6e} m")
    print(f"姿态误差角  : {result.end_pose_report.orientation_error_deg:.6f} deg")

    print("=" * 60)


def print_case_header(case_name: str):
    print("\n" + "=" * 80)
    print(f"开始实验：{case_name}")
    print("=" * 80)


def print_case_result_summary(case_name: str, result):
    end_report = result.end_pose_report
    print(f"【{case_name}】结果摘要：")
    print(f"- 轨迹点数: {len(result.trajectory.points)}")
    print(f"- 终点位置误差: {end_report.position_error_norm:.6e} m")
    print(f"- 终点姿态误差: {end_report.orientation_error_deg:.6f} deg")
    print("-" * 80)


def print_scan_result_summary(case_name: str, result):
    print(f"【{case_name}】扫描结果摘要：")
    print(f"- 关键位姿点数: {len(result.key_poses)}")
    print(f"- 分段数量: {len(result.segment_results)}")
    print(f"- 总轨迹点数: {len(result.trajectory.points)}")
    print("-" * 80)


def print_scan_segment_validation_summary(result):
    print("🔍 开始进行扫描分段 FK 回代验证...")

    for i, seg_result in enumerate(result.segment_results):
        start_report = seg_result.start_pose_report
        end_report = seg_result.end_pose_report

        print("-" * 80)
        print(f"【扫描第 {i + 1} 段】")

        print("起点验证：")
        print(f"  - 位置误差: {start_report.position_error_norm:.6e} m")
        print(f"  - 姿态误差: {start_report.orientation_error_deg:.6f} deg")

        print("终点验证：")
        print(f"  - 位置误差: {end_report.position_error_norm:.6e} m")
        print(f"  - 姿态误差: {end_report.orientation_error_deg:.6f} deg")

    print("=" * 80)


def print_runtime_state(runtime_state: RuntimeMotionState, title: str):
    print("\n" + "-" * 80)
    print(title)
    print(f"当前主动关节角: {[round(v, 6) for v in runtime_state.current_active_joints]}")
    print(f"当前末端位置  : {[round(v, 6) for v in runtime_state.current_pose.position]}")
    print(f"当前末端四元数: {[round(v, 6) for v in runtime_state.current_pose.quaternion]}")
    print("-" * 80)


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
    core.print_model_info()
    return core


def select_cases(cases, case_index):
    if case_index is None:
        return cases

    if not (0 <= case_index < len(cases)):
        raise IndexError(f"case_index 越界，应在 [0, {len(cases)-1}]")

    return [cases[case_index]]


def list_cases_by_mode(core, mode):
    if mode == "ptp":
        cases = build_all_demo_cases(core)
        print("\n" + "=" * 80)
        print("[ptp] 点到点位姿轨迹")
        print("=" * 80)
        for i, (case_name, _) in enumerate(cases):
            print(f"  {i}: {case_name}")
        print("=" * 80)
        return

    if mode == "scan":
        cases = build_all_scan_demo_cases(core)
        print("\n" + "=" * 80)
        print("[scan] 一维往返扫描")
        print("=" * 80)
        for i, (case_name, _) in enumerate(cases):
            print(f"  {i}: {case_name}")
        print("=" * 80)
        return

    if mode == "grid":
        cases = build_all_grid_scan_demo_cases(core)
        print("\n" + "=" * 80)
        print("[grid] 二维栅格扫描")
        print("=" * 80)
        for i, (case_name, _) in enumerate(cases):
            print(f"  {i}: {case_name}")
        print("=" * 80)
        return

    if mode == "all":
        print("\n" + "=" * 80)
        print("可用测试用例列表")
        print("=" * 80)

        ptp_cases = build_all_demo_cases(core)
        scan_cases = build_all_scan_demo_cases(core)
        grid_cases = build_all_grid_scan_demo_cases(core)

        print("\n[ptp] 点到点位姿轨迹")
        for i, (case_name, _) in enumerate(ptp_cases):
            print(f"  {i}: {case_name}")

        print("\n[scan] 一维往返扫描")
        for i, (case_name, _) in enumerate(scan_cases):
            print(f"  {i}: {case_name}")

        print("\n[grid] 二维栅格扫描")
        for i, (case_name, _) in enumerate(grid_cases):
            print(f"  {i}: {case_name}")

        print("\n" + "=" * 80)
        return

    raise ValueError(f"未知 mode: {mode}")


def rebuild_ptp_request_from_runtime_state(runtime_state, original_request):
    """
    将 demo case 的“相对运动目标”保留，
    但起点改为 runtime_state 的当前状态。

    做法：
    - 保留原 case 的相对位移 delta_position
    - 保留原 case 的相对姿态变化（通过起终点四元数关系近似复用）
    - 新起点 = 当前运行时位姿
    - IK 初值 = 当前运行时关节角
    """
    start_pose = runtime_state.current_pose

    old_start = original_request.start_pose
    old_end = original_request.end_pose

    delta_position = [
        old_end.position[0] - old_start.position[0],
        old_end.position[1] - old_start.position[1],
        old_end.position[2] - old_start.position[2],
    ]

    core = rebuild_ptp_request_from_runtime_state.core

    start_q = old_start.quaternion
    end_q = old_end.quaternion

    # 这里只保留你原 case 的“绕 z 轴的相对小角度变化”的使用习惯
    # 如果以后状态机给的是明确终点四元数，就直接用真实目标姿态，不需要这层重建
    # 这里通过旋转矩阵差值估计一个相对旋转
    start_R = core.quaternion_to_rotation_matrix(start_q)
    end_R = core.quaternion_to_rotation_matrix(end_q)
    relative_R = end_R @ start_R.T
    relative_quat = core.rotation_matrix_to_quaternion(relative_R)
    new_end_quat = core.quaternion_multiply(relative_quat, start_pose.quaternion)

    end_pose = type(start_pose)(
        position=[
            start_pose.position[0] + delta_position[0],
            start_pose.position[1] + delta_position[1],
            start_pose.position[2] + delta_position[2],
        ],
        quaternion=new_end_quat,
    )

    return PointToPointPlanRequest(
        start_pose=start_pose,
        end_pose=end_pose,
        initial_active_joints=runtime_state.current_active_joints,
        motion_time=original_request.motion_time,
        num_waypoints=original_request.num_waypoints,
        ik_verbose=original_request.ik_verbose,
    )


def rebuild_scan_request_from_runtime_state(runtime_state, original_request):
    return ScanPlanRequest(
        start_pose=runtime_state.current_pose,
        initial_active_joints=runtime_state.current_active_joints,
        scan_direction=original_request.scan_direction,
        scan_distance=original_request.scan_distance,
        scan_count=original_request.scan_count,
        motion_time_per_segment=original_request.motion_time_per_segment,
        num_waypoints_per_segment=original_request.num_waypoints_per_segment,
        ik_verbose=original_request.ik_verbose,
    )


def rebuild_grid_request_from_runtime_state(runtime_state, original_request):
    return GridScanPlanRequest(
        start_pose=runtime_state.current_pose,
        initial_active_joints=runtime_state.current_active_joints,
        line_direction=original_request.line_direction,
        line_length=original_request.line_length,
        line_count=original_request.line_count,
        step_direction=original_request.step_direction,
        step_distance=original_request.step_distance,
        motion_time_per_segment=original_request.motion_time_per_segment,
        num_waypoints_per_segment=original_request.num_waypoints_per_segment,
        ik_verbose=original_request.ik_verbose,
    )


def run_ptp_tests(core, runtime_state: RuntimeMotionState, case_index=None):
    print("\n" + "#" * 80)
    print("开始：点到点位姿轨迹测试")
    print("#" * 80)

    ptp_template = PointToPointPoseTemplate(core)
    ptp_cases = build_all_demo_cases(core)
    selected_cases = select_cases(ptp_cases, case_index)

    success_count = 0
    total_count = len(selected_cases)

    rebuild_ptp_request_from_runtime_state.core = core

    for case_name, original_request in selected_cases:
        print_case_header(case_name)
        print_runtime_state(runtime_state, "本次规划前的运行时状态")

        request = rebuild_ptp_request_from_runtime_state(runtime_state, original_request)

        try:
            print("起点位置:", [round(v, 6) for v in request.start_pose.position])
            print("起点四元数:", [round(v, 6) for v in request.start_pose.quaternion])
            print("终点位置:", [round(v, 6) for v in request.end_pose.position])
            print("终点四元数:", [round(v, 6) for v in request.end_pose.quaternion])

            result = ptp_template.plan(request)

            print_trajectory_summary(result.trajectory)
            print_validation_summary(request, result)
            print_case_result_summary(case_name, result)

            runtime_state.update_from_trajectory_end(core, result.trajectory)
            print_runtime_state(runtime_state, "本次规划完成后更新的运行时状态")

            success_count += 1

        except Exception as e:
            print(f"❌ 实验失败：{case_name}")
            print("错误信息：", e)
            print("-" * 80)

    print("\n" + "=" * 80)
    print("点到点实验结束")
    print(f"成功数量: {success_count}/{total_count}")
    print("=" * 80)


def run_scan_tests(core, runtime_state: RuntimeMotionState, case_index=None):
    print("\n" + "#" * 80)
    print("开始：扫描轨迹模板测试（一维往返）")
    print("#" * 80)

    scan_template = BackAndForthScanPoseTemplate(core)
    scan_cases = build_all_scan_demo_cases(core)
    selected_cases = select_cases(scan_cases, case_index)

    success_count = 0
    total_count = len(selected_cases)

    for case_name, original_request in selected_cases:
        print_case_header(case_name)
        print_runtime_state(runtime_state, "本次规划前的运行时状态")

        request = rebuild_scan_request_from_runtime_state(runtime_state, original_request)

        try:
            print("扫描起点位置:", [round(v, 6) for v in request.start_pose.position])
            print("扫描起点四元数:", [round(v, 6) for v in request.start_pose.quaternion])
            print("扫描方向:", [round(v, 6) for v in request.scan_direction])
            print("扫描距离:", request.scan_distance)
            print("扫描次数:", request.scan_count)

            result = scan_template.plan(request)

            print("关键位姿点列表：")
            for i, pose in enumerate(result.key_poses):
                print(
                    f"关键点 [{i:2d}] | 位置: {[round(v, 6) for v in pose.position]} | "
                    f"四元数: {[round(v, 6) for v in pose.quaternion]}"
                )

            print_trajectory_summary(result.trajectory)
            print_scan_segment_validation_summary(result)
            print_scan_result_summary(case_name, result)

            runtime_state.update_from_trajectory_end(core, result.trajectory)
            print_runtime_state(runtime_state, "本次规划完成后更新的运行时状态")

            success_count += 1

        except Exception as e:
            print(f"❌ 扫描实验失败：{case_name}")
            print("错误信息：", e)
            print("-" * 80)

    print("\n" + "=" * 80)
    print("一维扫描实验结束")
    print(f"成功数量: {success_count}/{total_count}")
    print("=" * 80)


def run_grid_tests(core, runtime_state: RuntimeMotionState, case_index=None):
    print("\n" + "#" * 80)
    print("开始：二维栅格扫描轨迹模板测试")
    print("#" * 80)

    grid_scan_template = GridScanPoseTemplate(core)
    grid_scan_cases = build_all_grid_scan_demo_cases(core)
    selected_cases = select_cases(grid_scan_cases, case_index)

    success_count = 0
    total_count = len(selected_cases)

    for case_name, original_request in selected_cases:
        print_case_header(case_name)
        print_runtime_state(runtime_state, "本次规划前的运行时状态")

        request = rebuild_grid_request_from_runtime_state(runtime_state, original_request)

        try:
            print("栅格起点位置:", [round(v, 6) for v in request.start_pose.position])
            print("栅格起点四元数:", [round(v, 6) for v in request.start_pose.quaternion])
            print("行扫描方向:", [round(v, 6) for v in request.line_direction])
            print("每行长度:", request.line_length)
            print("扫描行数:", request.line_count)
            print("步进方向:", [round(v, 6) for v in request.step_direction])
            print("步进距离:", request.step_distance)

            result = grid_scan_template.plan(request)

            print("关键位姿点列表：")
            for i, pose in enumerate(result.key_poses):
                print(
                    f"关键点 [{i:2d}] | 位置: {[round(v, 6) for v in pose.position]} | "
                    f"四元数: {[round(v, 6) for v in pose.quaternion]}"
                )

            print_trajectory_summary(result.trajectory)
            print_scan_segment_validation_summary(result)
            print_scan_result_summary(case_name, result)

            runtime_state.update_from_trajectory_end(core, result.trajectory)
            print_runtime_state(runtime_state, "本次规划完成后更新的运行时状态")

            success_count += 1

        except Exception as e:
            print(f"❌ 栅格扫描实验失败：{case_name}")
            print("错误信息：", e)
            print("-" * 80)

    print("\n" + "=" * 80)
    print("二维栅格扫描实验结束")
    print(f"成功数量: {success_count}/{total_count}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "ptp", "scan", "grid"],
        help="选择测试模式"
    )
    parser.add_argument(
        "--case",
        type=int,
        default=None,
        help="指定要运行的 case 索引，不填则运行该模式下全部 case"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用测试用例，不执行规划"
    )
    parser.add_argument(
        "--reset_before_each_mode",
        action="store_true",
        help="每种模式开始前都重置为默认位姿；不加此参数则全程连续衔接"
    )
    args = parser.parse_args()

    print("========= Pinocchio 位姿轨迹模板启动 ==========")

    core = build_core()
    runtime_state = RuntimeMotionState.build_default(core)

    print_runtime_state(runtime_state, "系统启动后的默认运行时状态")

    if args.list:
        list_cases_by_mode(core, args.mode)
        return

    if args.mode in ["all", "ptp"]:
        if args.reset_before_each_mode and args.mode == "all":
            runtime_state.reset_to_default(core)
            print_runtime_state(runtime_state, "进入 ptp 模式前已重置默认状态")
        run_ptp_tests(core, runtime_state, args.case if args.mode == "ptp" else None)

    if args.mode in ["all", "scan"]:
        if args.reset_before_each_mode and args.mode == "all":
            runtime_state.reset_to_default(core)
            print_runtime_state(runtime_state, "进入 scan 模式前已重置默认状态")
        run_scan_tests(core, runtime_state, args.case if args.mode == "scan" else None)

    if args.mode in ["all", "grid"]:
        if args.reset_before_each_mode and args.mode == "all":
            runtime_state.reset_to_default(core)
            print_runtime_state(runtime_state, "进入 grid 模式前已重置默认状态")
        run_grid_tests(core, runtime_state, args.case if args.mode == "scan" else None)

    print("\n" + "=" * 80)
    print("全部实验结束")
    print("=" * 80)
    print_runtime_state(runtime_state, "全部实验结束后的最终运行时状态")


if __name__ == "__main__":
    main()