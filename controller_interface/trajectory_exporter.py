import json
import math
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from motion_algorithm.models import PlannedTrajectory


def rad_to_deg(value: float) -> float:
    return value * 180.0 / math.pi


def trajectory_to_dict(
    trajectory: PlannedTrajectory,
    case_name: str = "unknown_case",
    trajectory_id: Optional[str] = None,
    angle_unit: str = "rad",
    convert_to_degrees: bool = False,
) -> Dict[str, Any]:
    """
    将 PlannedTrajectory 导出为标准字典结构。

    参数：
    - trajectory: 规划结果
    - case_name: 用例名
    - trajectory_id: 轨迹唯一ID，不传则自动生成
    - angle_unit: 输出单位说明，建议 "rad" 或 "deg"
    - convert_to_degrees: 是否将关节角从 rad 转为 deg

    返回：
    - 标准 dict，可继续转 JSON 或通过 TCP 发送
    """
    if trajectory_id is None:
        trajectory_id = str(uuid.uuid4())

    if len(trajectory.points) == 0:
        duration = 0.0
    else:
        duration = float(trajectory.points[-1].time_from_start)

    exported_points = []
    for pt in trajectory.points:
        q = [float(v) for v in pt.positions]
        if convert_to_degrees:
            q = [rad_to_deg(v) for v in q]

        exported_points.append(
            {
                "t": float(pt.time_from_start),
                "q": q,
            }
        )

    result = {
        "type": "trajectory",
        "trajectory_id": trajectory_id,
        "case_name": case_name,
        "joint_names": list(trajectory.joint_names),
        "angle_unit": angle_unit,
        "time_unit": "s",
        "point_count": len(trajectory.points),
        "duration": duration,
        "points": exported_points,
    }
    return result


def trajectory_to_json(
    trajectory: PlannedTrajectory,
    case_name: str = "unknown_case",
    trajectory_id: Optional[str] = None,
    angle_unit: str = "rad",
    convert_to_degrees: bool = False,
    indent: int = 2,
) -> str:
    """
    将轨迹导出为 JSON 字符串。
    """
    data = trajectory_to_dict(
        trajectory=trajectory,
        case_name=case_name,
        trajectory_id=trajectory_id,
        angle_unit=angle_unit,
        convert_to_degrees=convert_to_degrees,
    )
    return json.dumps(data, ensure_ascii=False, indent=indent)


def save_trajectory_json(
    trajectory: PlannedTrajectory,
    save_path: str,
    case_name: str = "unknown_case",
    trajectory_id: Optional[str] = None,
    angle_unit: str = "rad",
    convert_to_degrees: bool = False,
    indent: int = 2,
) -> str:
    """
    保存轨迹为 JSON 文件。
    返回保存后的绝对路径。
    """
    json_text = trajectory_to_json(
        trajectory=trajectory,
        case_name=case_name,
        trajectory_id=trajectory_id,
        angle_unit=angle_unit,
        convert_to_degrees=convert_to_degrees,
        indent=indent,
    )

    path = Path(save_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text, encoding="utf-8")
    return str(path)