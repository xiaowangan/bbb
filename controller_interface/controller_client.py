import uuid
from typing import Any, Dict, List, Optional

from .tcp_sender import send_json_message
from .trajectory_validator import validate_trajectory


class ControllerTCPClient:
    """
    真实运控客户端：
    1. 接收上层传入的简化轨迹格式: [{"t": ..., "q": [...]}, ...]
    2. 在本地封装为运控协议 payload
    3. 做轨迹校验
    4. 通过 TCP 发送给运控
    """

    def __init__(
        self,
        host: str = "192.168.112.95",
        port: int = 9001,
        timeout: float = 5.0,
        joint_names: Optional[List[str]] = None,
        angle_unit: str = "rad",
        max_joint_step: float = 0.3,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.joint_names = joint_names or ["j1", "j2", "j3", "j4", "j5", "j6"]
        self.angle_unit = angle_unit
        self.max_joint_step = max_joint_step

    def send_trajectory(
        self,
        task_id: str,
        trajectory: list,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not trajectory:
            return {
                "success": False,
                "message": "trajectory is empty",
                "task_id": task_id,
            }

        validation = self._validate_serialized_trajectory(trajectory)
        if not validation["is_valid"]:
            return {
                "success": False,
                "message": "trajectory validation failed",
                "task_id": task_id,
                "validation": validation,
            }

        payload = self._build_payload(
            task_id=task_id,
            trajectory=trajectory,
            summary=summary or {},
        )

        ok, ack = send_json_message(
            host=self.host,
            port=self.port,
            payload=payload,
            timeout=self.timeout,
        )

        return {
            "success": bool(ok),
            "message": ack.get("message", "ok" if ok else "controller ack failed"),
            "task_id": task_id,
            "controller_host": self.host,
            "controller_port": self.port,
            "payload_summary": {
                "trajectory_id": payload["trajectory_id"],
                "point_count": payload["point_count"],
                "duration": payload["duration"],
                "angle_unit": payload["angle_unit"],
            },
            "ack": ack,
            "validation": validation,
        }

    def _build_payload(
        self,
        task_id: str,
        trajectory: list,
        summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        duration = float(trajectory[-1]["t"]) if trajectory else 0.0

        return {
            "type": "trajectory",
            "trajectory_id": str(uuid.uuid4()),
            "task_id": task_id,
            "case_name": summary.get("plan_type", "upper_machine_auto_plan"),
            "joint_names": list(self.joint_names),
            "angle_unit": self.angle_unit,
            "time_unit": "s",
            "point_count": len(trajectory),
            "duration": duration,
            "points": trajectory,
            "summary": summary,
        }

    def _validate_serialized_trajectory(self, trajectory: list) -> Dict[str, Any]:
        class _Point:
            def __init__(self, t: float, q: list) -> None:
                self.time_from_start = t
                self.positions = q

        class _Trajectory:
            def __init__(self, points: list, joint_names: list) -> None:
                self.points = points
                self.joint_names = joint_names

        points = []
        for pt in trajectory:
            t = float(pt["t"])
            q = [float(v) for v in pt["q"]]
            points.append(_Point(t, q))

        dummy_traj = _Trajectory(points, self.joint_names)

        result = validate_trajectory(
            trajectory=dummy_traj,
            expected_joint_dim=len(self.joint_names),
            max_joint_step=self.max_joint_step,
            require_strict_time_increase=True,
        )

        return {
            "is_valid": result.is_valid,
            "errors": list(result.errors),
            "warnings": list(result.warnings),
            "point_count": result.point_count,
            "duration": result.duration,
            "max_joint_step": result.max_joint_step,
        }