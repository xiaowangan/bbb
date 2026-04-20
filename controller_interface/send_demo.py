import argparse
import importlib
import inspect
from pathlib import Path

from motion_algorithm.main import build_core
from motion_algorithm.PointToPoint_template import PointToPointPoseTemplate
from motion_algorithm.point_demo_cases import build_all_demo_cases as build_point_demo_cases

from motion_algorithm.controller_interface.trajectory_exporter import (
    trajectory_to_dict,
    save_trajectory_json,
)
from motion_algorithm.controller_interface.trajectory_validator import (
    validate_trajectory,
    print_validation_result,
)
from motion_algorithm.controller_interface.tcp_sender import send_trajectory_payload


def print_case_list(cases):
    print("\n可用 case 列表：")
    print("=" * 80)
    for idx, case in enumerate(cases):
        print(f"[{idx}] {case['name']}")
    print("=" * 80)


def select_case(cases, case_index=None, case_name=None):
    if case_name is not None:
        for idx, case in enumerate(cases):
            if case["name"] == case_name:
                return idx, case
        raise ValueError(f"未找到 case_name='{case_name}'")

    if case_index is None:
        case_index = 0

    if case_index < 0 or case_index >= len(cases):
        raise IndexError(f"case_index 越界: {case_index}，有效范围 0 ~ {len(cases)-1}")

    return case_index, cases[case_index]


def parse_args():
    parser = argparse.ArgumentParser(description="轨迹规划并通过 TCP 发送给对方")
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅打印所有可用 case，不执行规划和发送",
    )
    parser.add_argument(
        "--case-index",
        type=int,
        default=None,
        help="按索引选择 case，例如 --case-index 0",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default=None,
        help='按名称选择 case，例如 --case-name "[scan] X方向往返扫描3次"',
    )
    parser.add_argument(
        "--host",
        type=str,
        default="192.168.112.95",
        help="TCP 接收端 IP",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9001,
        help="TCP 接收端端口",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="TCP 超时时间（秒）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="轨迹 JSON 导出目录",
    )
    parser.add_argument(
        "--angle-unit",
        choices=["rad", "deg"],
        default="rad",
        help="发送给对方的角度单位",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.3,
        help="轨迹校验时允许的最大相邻关节跳变",
    )
    return parser.parse_args()


def try_import_scan_case_builder():
    """
    自动尝试导入扫动 case 构造函数。
    返回:
        builder_func 或 None
    """
    candidates = [
        ("motion_algorithm.scan_demo_cases", "build_all_scan_demo_cases"),
        ("motion_algorithm.scan_demo_cases", "build_all_demo_cases"),
        ("motion_algorithm.scan_demo_cases", "build_scan_demo_cases"),
        ("motion_algorithm.scan_cases", "build_all_scan_demo_cases"),
        ("motion_algorithm.scan_cases", "build_all_demo_cases"),
        ("motion_algorithm.scan_cases", "build_scan_demo_cases"),
    ]

    for module_name, func_name in candidates:
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name, None)
            if callable(func):
                print(f"已加载扫动 case 构造函数: {module_name}.{func_name}")
                return func
        except Exception:
            pass

    print("未找到扫动 case 构造函数，将只保留点到点 case。")
    return None


def try_import_scan_template(core):
    """
    自动尝试导入扫动模板类并实例化。
    优先从 motion_algorithm.Scan_template 中找。
    返回:
        template_instance 或 None
    """
    module_candidates = [
        "motion_algorithm.Scan_template",
        "motion_algorithm.scan_template",
    ]

    preferred_class_names = [
        "ScanTrajectoryTemplate",
        "ScanTemplate",
        "ScanPoseTemplate",
        "ScanPlannerTemplate",
    ]

    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            print(f"已导入扫动模板模块: {module_name}")

            # 先按常见类名找
            for class_name in preferred_class_names:
                cls = getattr(module, class_name, None)
                if inspect.isclass(cls):
                    try:
                        instance = cls(core)
                        if hasattr(instance, "plan") and callable(instance.plan):
                            print(f"已加载扫动模板类: {module_name}.{class_name}")
                            return instance
                    except Exception:
                        pass

            # 再自动扫描模块内所有类，找带 plan() 的 Scan 相关类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if not inspect.isclass(attr):
                    continue

                if attr.__module__ != module.__name__:
                    continue

                name_lower = attr.__name__.lower()
                if "scan" not in name_lower:
                    continue

                try:
                    instance = attr(core)
                    if hasattr(instance, "plan") and callable(instance.plan):
                        print(f"已自动匹配扫动模板类: {module_name}.{attr.__name__}")
                        return instance
                except Exception:
                    continue

        except Exception as e:
            print(f"尝试导入扫动模板模块失败: {module_name} | {e}")

    print("未找到扫动模板类，扫动 case 将无法执行。")
    return None


def build_all_cases(core):
    all_cases = []

    # 点到点 case
    try:
        point_cases = build_point_demo_cases(core)
        for name, request in point_cases:
            all_cases.append({
                "name": f"[point] {name}",
                "request": request,
                "planner_type": "point",
            })
    except Exception as e:
        print(f"加载点到点 case 失败: {e}")

    # 扫动 case
    scan_case_builder = try_import_scan_case_builder()
    if scan_case_builder is not None:
        try:
            scan_cases = scan_case_builder(core)
            for name, request in scan_cases:
                all_cases.append({
                    "name": f"[scan] {name}",
                    "request": request,
                    "planner_type": "scan",
                })
        except Exception as e:
            print(f"加载扫动 case 失败: {e}")

    return all_cases


def execute_case(ptp_template, scan_template, planner_type, request):
    if planner_type == "point":
        return ptp_template.plan(request)

    if planner_type == "scan":
        if scan_template is None:
            raise RuntimeError("未找到扫动模板类，无法执行 scan case。")
        return scan_template.plan(request)

    raise ValueError(f"未知 planner_type: {planner_type}")


def main():
    args = parse_args()

    print("========= send_demo 启动 ==========")

    core = build_core()
    ptp_template = PointToPointPoseTemplate(core)
    scan_template = try_import_scan_template(core)

    cases = build_all_cases(core)

    if len(cases) == 0:
        print("未找到任何可用 case。")
        return

    if args.list:
        print_case_list(cases)
        return

    print_case_list(cases)

    try:
        case_index, case_item = select_case(
            cases=cases,
            case_index=args.case_index,
            case_name=args.case_name,
        )
    except Exception as e:
        print(f"选择 case 失败: {e}")
        return

    case_name = case_item["name"]
    request = case_item["request"]
    planner_type = case_item["planner_type"]

    print(f"\n当前执行 case: [{case_index}] {case_name}")
    print(f"规划类型: {planner_type}")
    print(f"request 类型: {type(request).__name__}")

    try:
        result = execute_case(
            ptp_template=ptp_template,
            scan_template=scan_template,
            planner_type=planner_type,
            request=request,
        )
    except Exception as e:
        print(f"轨迹规划失败: {e}")
        return

    validation = validate_trajectory(
        trajectory=result.trajectory,
        expected_joint_dim=6,
        max_joint_step=args.max_joint_step,
    )
    print_validation_result(validation)

    if not validation.is_valid:
        print("轨迹检查不通过，取消后续导出与发送。")
        return

    convert_to_degrees = args.angle_unit == "deg"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_case_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in case_name
    )

    save_path = save_trajectory_json(
        trajectory=result.trajectory,
        save_path=str(output_dir / f"case_{case_index}_{safe_case_name}.json"),
        case_name=case_name,
        angle_unit=args.angle_unit,
        convert_to_degrees=convert_to_degrees,
    )
    print(f"轨迹已保存到: {save_path}")

    payload = trajectory_to_dict(
        trajectory=result.trajectory,
        case_name=case_name,
        angle_unit=args.angle_unit,
        convert_to_degrees=convert_to_degrees,
    )

    print("\n轨迹 payload 构造完成。")
    print(f"case_name: {payload['case_name']}")
    print(f"point_count: {payload['point_count']}")
    print(f"duration: {payload['duration']:.3f} s")
    print(f"angle_unit: {payload['angle_unit']}")
    print(f"发送目标: {args.host}:{args.port}")

    try:
        send_trajectory_payload(
            host=args.host,
            port=args.port,
            trajectory_payload=payload,
            timeout=args.timeout,
        )
        print("TCP 发送完成。")
    except Exception as e:
        print(f"TCP 发送失败: {e}")
        return


if __name__ == "__main__":
    main()