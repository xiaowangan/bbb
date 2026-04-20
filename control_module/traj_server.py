#!/usr/bin/env python3
"""
traj_server.py — 轨迹接收服务模块

职责：
  · 作为 TCP 服务端，监听轨迹模块（tcp_sender）的连接
  · 收到轨迹后立即回 ack（收到即回，不等执行完）
  · 通过回调把轨迹数据交给 FR5Controller.execute_trajectory 异步执行
  · 所有执行结果的上报、日志、状态机处理均由 execute_trajectory 自己负责
    本模块不做二次上报，避免重复

协议（与 tcp_sender.py 对应）：
  发：4字节大端无符号整数（正文字节数）+ UTF-8 JSON 正文
  收：同格式的 JSON ACK

ACK 格式：
  成功收到（格式校验通过）:
    {"ok": true,  "message": "已接收", "trajectory_id": "xxx"}
  格式校验失败:
    {"ok": false, "message": "原因",   "trajectory_id": "xxx"}

轨迹 payload 预期字段（由 tcp_sender 发来）:
  {
    "type":          "trajectory",
    "trajectory_id": "uuid-xxx",
    "case_name":     "sweep_A",
    "joint_names":   ["j1","j2","j3","j4","j5","j6"],
    "angle_unit":    "deg",
    "point_count":   N,
    "duration":      3.5,
    "points": [
      {"t": 0.0,  "q": [37.0, -158.0, 97.0, 48.0, -61.0, -0.3]},
      {"t": 0.05, "q": [...] },
      ...
    ]
  }

使用方式（在主代码里）：

    from traj_server import TrajectoryServer

    traj_srv = TrajectoryServer(
        listen_port   = 9001,
        on_trajectory = ctrl.execute_trajectory,
    )
    traj_srv.start()

    # 退出时
    traj_srv.stop()
"""

import json
import socket
import struct
import threading
from typing import Any, Callable, Dict, Optional


# ──────────────────────────────────────────────
# 配置常量
# ──────────────────────────────────────────────
LISTEN_IP    = "0.0.0.0"
RECV_TIMEOUT = 10.0


# ──────────────────────────────────────────────
# 底层收发工具
# ──────────────────────────────────────────────

def _recv_exact(conn: socket.socket, num_bytes: int) -> bytes:
    chunks, received = [], 0
    while received < num_bytes:
        chunk = conn.recv(num_bytes - received)
        if not chunk:
            raise ConnectionError("连接已关闭，未收到完整数据")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _send_json(conn: socket.socket, payload: Dict[str, Any]) -> None:
    body   = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!I", len(body))
    conn.sendall(header)
    conn.sendall(body)


def _recv_json(conn: socket.socket) -> Dict[str, Any]:
    header   = _recv_exact(conn, 4)
    body_len = struct.unpack("!I", header)[0]
    body     = _recv_exact(conn, body_len)
    return json.loads(body.decode("utf-8"))


# ──────────────────────────────────────────────
# 格式校验
# ──────────────────────────────────────────────

def _validate_trajectory(payload: Dict[str, Any]) -> Optional[str]:
    """返回 None 表示格式合法；返回字符串表示失败原因。"""
    if payload.get("type") != "trajectory":
        return "payload.type 不是 trajectory"

    joint_names = payload.get("joint_names", [])
    points      = payload.get("points", [])

    if not isinstance(joint_names, list) or len(joint_names) == 0:
        return "joint_names 非法或为空"

    if not isinstance(points, list) or len(points) == 0:
        return "points 非法或为空"

    prev_t = None
    for i, pt in enumerate(points):
        if "t" not in pt or "q" not in pt:
            return f"点 {i} 缺少 t 或 q 字段"
        q = pt["q"]
        if not isinstance(q, list):
            return f"点 {i} 的 q 不是 list"
        if len(q) != len(joint_names):
            return (f"点 {i} 维度 {len(q)} "
                    f"与 joint_names 数量 {len(joint_names)} 不一致")
        t = float(pt["t"])
        if prev_t is not None and t <= prev_t:
            return f"点 {i} 时间未严格递增"
        prev_t = t

    return None


# ══════════════════════════════════════════════
# TrajectoryServer 主类
# ══════════════════════════════════════════════

class TrajectoryServer:
    """
    TCP 服务端：接收轨迹模块推来的轨迹，立即回 ack，
    然后在独立线程里调用 on_trajectory（即 FR5Controller.execute_trajectory）执行。
    执行结果的上报、日志、状态机处理全部由 execute_trajectory 自己负责，本类不介入。
    """

    def __init__(
        self,
        listen_port  : int,
        on_trajectory: Callable[[Dict[str, Any]], bool],
        listen_ip    : str = LISTEN_IP,
    ):
        self._port          = listen_port
        self._ip            = listen_ip
        self._on_trajectory = on_trajectory

        self._server_sock  : Optional[socket.socket] = None
        self._running      = False
        self._listen_thread: Optional[threading.Thread] = None

    # ──────────────────────────────────────────
    # 启动 / 停止
    # ──────────────────────────────────────────

    def start(self):
        self._running     = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self._ip, self._port))
        self._server_sock.listen(5)
        self._server_sock.settimeout(1.0)

        self._listen_thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="TrajServer-listen"
        )
        self._listen_thread.start()
        print(f"  [TrajServer] 已启动，监听 {self._ip}:{self._port}")

    def stop(self):
        self._running = False
        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
        print("  [TrajServer] 已停止")

    # ──────────────────────────────────────────
    # 内部：监听主循环
    # ──────────────────────────────────────────

    def _listen_loop(self):
        while self._running:
            try:
                conn, addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            threading.Thread(
                target=self._handle_connection,
                args=(conn, addr),
                daemon=True,
                name=f"TrajServer-conn-{addr[1]}",
            ).start()

    # ──────────────────────────────────────────
    # 内部：处理单次连接
    # ──────────────────────────────────────────

    def _handle_connection(self, conn: socket.socket, addr):
        traj_id = "unknown"
        try:
            conn.settimeout(RECV_TIMEOUT)

            # 1. 接收轨迹
            payload = _recv_json(conn)
            traj_id = payload.get("trajectory_id", "unknown")
            n_points = len(payload.get("points", []))
            print(f"\n  [TrajServer] 收到轨迹 id={traj_id}  "
                  f"来自={addr}  点数={n_points}")

            # 2. 格式校验
            err_msg = _validate_trajectory(payload)
            if err_msg:
                ack = {"ok": False, "message": err_msg, "trajectory_id": traj_id}
                _send_json(conn, ack)
                print(f"  [TrajServer] 格式校验失败，已拒绝: {err_msg}")
                return

            # 3. 立即回"已收到" ack（不等执行完）
            ack = {"ok": True, "message": "已接收", "trajectory_id": traj_id}
            _send_json(conn, ack)
            print(f"  [TrajServer] 已回 ack，启动异步执行 id={traj_id}")

            # 4. 异步执行，后续全部由 execute_trajectory 负责
            threading.Thread(
                target=self._on_trajectory,
                args=(payload,),
                daemon=True,
                name=f"TrajServer-exec-{traj_id}",
            ).start()

        except (ConnectionError, OSError, json.JSONDecodeError) as e:
            print(f"  [TrajServer] 连接处理异常 {addr}: {e}")
            try:
                ack = {"ok": False, "message": f"服务端异常: {e}",
                       "trajectory_id": traj_id}
                _send_json(conn, ack)
            except Exception:
                pass
        finally:
            conn.close()
