import math
from dataclasses import dataclass, field
from typing import List, Optional

from motion_algorithm.models import PlannedTrajectory


@dataclass
class TrajectoryValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    point_count: int = 0
    duration: float = 0.0
    max_joint_step: float = 0.0


def _is_finite_number(value: float) -> bool:
    return math.isfinite(float(value))


def validate_trajectory(
    trajectory: PlannedTrajectory,
    expected_joint_dim: Optional[int] = 6,
    max_joint_step: float = 0.2,
    require_strict_time_increase: bool = True,
) -> TrajectoryValidationResult:
    """
    对轨迹做基础合法性检查。

    参数：
    - expected_joint_dim: 每个点关节维度，默认 6
    - max_joint_step: 相邻点最大允许跳变（按当前单位判断）
    - require_strict_time_increase: 是否要求时间严格递增

    返回：
    - TrajectoryValidationResult
    """
    result = TrajectoryValidationResult(is_valid=True)

    if trajectory is None:
        result.is_valid = False
        result.errors.append("trajectory 为 None")
        return result

    result.point_count = len(trajectory.points)

    if len(trajectory.points) == 0:
        result.is_valid = False
        result.errors.append("轨迹为空，没有任何点")
        return result

    if expected_joint_dim is not None and len(trajectory.joint_names) != expected_joint_dim:
        result.warnings.append(
            f"joint_names 数量为 {len(trajectory.joint_names)}，预期为 {expected_joint_dim}"
        )

    prev_t = None
    prev_q = None
    max_step_observed = 0.0

    for idx, pt in enumerate(trajectory.points):
        t = float(pt.time_from_start)
        q = [float(v) for v in pt.positions]

        if not _is_finite_number(t):
            result.is_valid = False
            result.errors.append(f"点 {idx} 的时间不是有限数: {t}")

        if expected_joint_dim is not None and len(q) != expected_joint_dim:
            result.is_valid = False
            result.errors.append(
                f"点 {idx} 的关节维度错误: {len(q)}，预期 {expected_joint_dim}"
            )

        for j, v in enumerate(q):
            if not _is_finite_number(v):
                result.is_valid = False
                result.errors.append(f"点 {idx} 第 {j} 个关节值不是有限数: {v}")

        if prev_t is not None:
            if require_strict_time_increase:
                if t <= prev_t:
                    result.is_valid = False
                    result.errors.append(
                        f"时间未严格递增: 点 {idx-1} 时间 {prev_t:.6f}, 点 {idx} 时间 {t:.6f}"
                    )
            else:
                if t < prev_t:
                    result.is_valid = False
                    result.errors.append(
                        f"时间倒退: 点 {idx-1} 时间 {prev_t:.6f}, 点 {idx} 时间 {t:.6f}"
                    )

        if prev_q is not None and len(q) == len(prev_q):
            local_max_step = max(abs(a - b) for a, b in zip(q, prev_q))
            max_step_observed = max(max_step_observed, local_max_step)
            if local_max_step > max_joint_step:
                result.warnings.append(
                    f"点 {idx-1}->{idx} 相邻关节跳变较大: {local_max_step:.6f} > {max_joint_step:.6f}"
                )

        prev_t = t
        prev_q = q

    result.duration = float(trajectory.points[-1].time_from_start)
    result.max_joint_step = max_step_observed

    return result


def print_validation_result(result: TrajectoryValidationResult) -> None:
    print("\n" + "=" * 60)
    print("轨迹检查结果")
    print("=" * 60)
    print(f"是否通过: {result.is_valid}")
    print(f"点数: {result.point_count}")
    print(f"总时长: {result.duration:.6f} s")
    print(f"最大相邻关节跳变: {result.max_joint_step:.6f}")

    if result.errors:
        print("\n错误：")
        for err in result.errors:
            print(f"  - {err}")

    if result.warnings:
        print("\n警告：")
        for warn in result.warnings:
            print(f"  - {warn}")

    print("=" * 60)