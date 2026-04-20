#!/usr/bin/env python3
"""
host_comm.py — 运动控制节点与上位机通信模块

职责：
  · get_system_state()       主动拉：定时从上位机拉取系统状态机状态（每秒一次）
  · send_execution_result()  主动推：运动执行完毕后向上位机上报结果
  · on_state_change()        内部钩子：本地状态机变化时同步通知上位机

使用方式（在 fr5_controller_sdk_v10.py 里）：

    from host_comm import HostComm

    # 1. 启动通信（连接上位机）
    comm = HostComm(host_ip="192.168.58.1", host_port=9000)
    comm.start()

    # 2. 把 comm 注入 FR5Controller，让它在合适时机调用
    ctrl = FR5Controller(robot_ip=ROBOT_IP, comm=comm)

    # 3. 退出时关闭
    comm.stop()

消息格式（均为 JSON + 换行符分隔）：

  拉取请求（你→上位机）:
    {"type": "get_system_state"}

  上位机回复:
    {"type": "system_state", "state": "AUTO", "t": 1720000000.123}

  推送执行结果（你→上位机）:
    {
      "type":        "execution_result",
      "cmd":         "RECENTER",
      "ok":          true,
      "max_err_deg": 0.12,
      "reason":      "",
      "t":           1720000000.456
    }

  本地状态变化通知（你→上位机，on_state_change 触发）:
    {"type": "state_change", "old": "AUTO", "new": "ERROR", "reason": "MoveJ错误码:-1", "t": ...}
"""

import json
import socket
import threading
import time
from typing import Optional


# ──────────────────────────────────────────────
# 配置常量
# ──────────────────────────────────────────────
POLL_INTERVAL   = 1.0    # get_system_state 轮询间隔（秒）
CONNECT_TIMEOUT = 5.0    # TCP 连接超时（秒）
RECV_TIMEOUT    = 3.0    # 单次接收超时（秒）
RECONNECT_DELAY = 3.0    # 断线后重连等待（秒）
RECV_BUFSIZE    = 4096


# ══════════════════════════════════════════════
# HostComm 主类
# ══════════════════════════════════════════════

class HostComm:
    """
    与上位机的 TCP 通信类。
    独立于 FR5Controller，通过注入方式使用。
    """

    def __init__(self, host_ip: str, host_port: int):
        self._host_ip   = host_ip
        self._host_port = host_port

        self._sock      : Optional[socket.socket] = None
        self._lock      = threading.Lock()          # 保护 socket 的并发读写
        self._running   = False

        # 缓存：最近一次从上位机拉到的系统状态
        self._last_system_state : Optional[str]   = None
        self._last_state_time   : Optional[float] = None

        # 后台线程
        self._poll_thread   : Optional[threading.Thread] = None

    # ──────────────────────────────────────────
    # 启动 / 停止
    # ──────────────────────────────────────────

    def start(self):
        """启动通信模块：建立连接，启动轮询线程。"""
        self._running = True
        self._connect()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="HostComm-poll"
        )
        self._poll_thread.start()
        print(f"  [HostComm] 已启动，目标: {self._host_ip}:{self._host_port}")

    def stop(self):
        """停止通信模块，关闭 socket。"""
        self._running = False
        self._close_socket()
        print("  [HostComm] 已停止")

    # ──────────────────────────────────────────
    # 对外接口 1：get_system_state（主动拉）
    # ──────────────────────────────────────────

    def get_system_state(self) -> Optional[str]:
        """
        向上位机发送拉取请求，等待回复，返回状态字符串。
        由轮询线程定期调用，外部也可直接调用。
        返回值示例: "AUTO" / "MANUAL" / None（通信失败）
        """
        req = {"type": "get_system_state"}
        raw = self._send_and_recv(req)
        if raw is None:
            return None

        try:
            msg = json.loads(raw)
            if msg.get("type") == "system_state":
                state = msg.get("state")
                self._last_system_state = state
                self._last_state_time   = time.time()
                print(f"  [HostComm] 上位机系统状态: {state}")
                return state
            else:
                print(f"  [HostComm] get_system_state 收到非预期消息: {msg}")
                return None
        except json.JSONDecodeError as e:
            print(f"  [HostComm] get_system_state JSON解析失败: {e}  原始: {raw!r}")
            return None

    # ──────────────────────────────────────────
    # 对外接口 2：send_execution_result（主动推）
    # ──────────────────────────────────────────

    def send_execution_result(
        self,
        cmd        : str,
        ok         : bool,
        max_err_deg: float = 0.0,
        reason     : str   = "",
    ) -> bool:
        """
        向上位机上报一次运动执行结果（只推，不等回复）。

        参数:
            cmd         指令标识，如 "RECENTER" / "SWEEP_A" / "PRESET_0"
            ok          是否执行成功
            max_err_deg 最大关节误差（度），ok=True 时有意义
            reason      失败原因字符串，ok=False 时填写
        """
        msg = {
            "type"       : "execution_result",
            "cmd"        : cmd,
            "ok"         : ok,
            "max_err_deg": round(max_err_deg, 4),
            "reason"     : reason,
            "t"          : round(time.time(), 3),
        }
        success = self._send(msg)
        if success:
            status = "成功" if ok else f"失败({reason})"
            print(f"  [HostComm] 已上报执行结果: {cmd} → {status}")
        return success

    # ──────────────────────────────────────────
    # 对外接口 3：on_state_change（本地状态变化钩子）
    # ──────────────────────────────────────────

    def on_state_change(self, old: str, new: str, reason: str) -> bool:
        """
        本地状态机发生转移时调用，通知上位机（只推，不等回复）。
        在 FR5Controller._on_state_change 里调用此方法。

        示例调用:
            def _on_state_change(self, old, new, reason):
                if self.comm:
                    self.comm.on_state_change(old, new, reason)
        """
        msg = {
            "type"  : "state_change",
            "old"   : old,
            "new"   : new,
            "reason": reason,
            "t"     : round(time.time(), 3),
        }
        return self._send(msg)

    # ──────────────────────────────────────────
    # 属性：读取缓存的上位机状态（不发网络请求）
    # ──────────────────────────────────────────

    @property
    def last_system_state(self) -> Optional[str]:
        """最近一次轮询拿到的上位机系统状态（可能是 None，表示尚未拿到）。"""
        return self._last_system_state

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    # ──────────────────────────────────────────
    # 内部：轮询线程（每秒调一次 get_system_state）
    # ──────────────────────────────────────────

    def _poll_loop(self):
        """后台轮询线程，每隔 POLL_INTERVAL 秒拉一次上位机系统状态。"""
        while self._running:
            if self._sock is None:
                # 未连接，尝试重连
                self._connect()
            else:
                self.get_system_state()
            time.sleep(POLL_INTERVAL)

    # ──────────────────────────────────────────
    # 内部：TCP 连接管理
    # ──────────────────────────────────────────

    def _connect(self):
        """尝试连接上位机，失败则打印警告（不抛异常，等待下次重试）。"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CONNECT_TIMEOUT)
            sock.connect((self._host_ip, self._host_port))
            sock.settimeout(RECV_TIMEOUT)
            with self._lock:
                self._sock = sock
            print(f"  [HostComm] 已连接上位机 {self._host_ip}:{self._host_port}")
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            print(f"  [HostComm] 连接失败: {e}，{RECONNECT_DELAY}s 后重试")
            time.sleep(RECONNECT_DELAY)

    def _close_socket(self):
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    # ──────────────────────────────────────────
    # 内部：发送（不等回复）
    # ──────────────────────────────────────────

    def _send(self, msg: dict) -> bool:
        """
        将 msg 序列化为 JSON 并通过 TCP 发送（用 \\n 分隔消息）。
        发送失败时自动断开连接，返回 False。
        """
        with self._lock:
            if self._sock is None:
                print(f"  [HostComm] 未连接，无法发送: {msg.get('type')}")
                return False
            try:
                data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
                self._sock.sendall(data)
                return True
            except (OSError, BrokenPipeError) as e:
                print(f"  [HostComm] 发送失败: {e}")
                self._sock = None          # 标记断线，轮询线程下次重连
                return False

    # ──────────────────────────────────────────
    # 内部：发送并等待一条回复
    # ──────────────────────────────────────────

    def _send_and_recv(self, msg: dict) -> Optional[str]:
        """
        发送 msg，然后等待上位机回复一行 JSON。
        超时或断线返回 None。
        """
        with self._lock:
            if self._sock is None:
                return None
            try:
                data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
                self._sock.sendall(data)

                # 按行读取（简单协议：每条消息一行，以 \n 结束）
                buf = b""
                while b"\n" not in buf:
                    chunk = self._sock.recv(RECV_BUFSIZE)
                    if not chunk:
                        raise ConnectionResetError("上位机关闭连接")
                    buf += chunk
                line, _ = buf.split(b"\n", 1)
                return line.decode("utf-8").strip()

            except (socket.timeout, OSError, ConnectionResetError) as e:
                print(f"  [HostComm] 收发异常: {e}")
                self._sock = None
                return None


# ══════════════════════════════════════════════
# 独立测试入口（不依赖机械臂，直接跑验证通信）
# ══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    HOST_IP   = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    HOST_PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 9000

    print(f"[测试] 连接 {HOST_IP}:{HOST_PORT}")
    comm = HostComm(host_ip=HOST_IP, host_port=HOST_PORT)
    comm.start()

    try:
        time.sleep(1.5)   # 等连接建立

        # 模拟：本地状态变化
        comm.on_state_change("MANUAL", "AUTO", "切换自动模式")
        time.sleep(0.5)

        # 模拟：运动成功
        comm.send_execution_result("RECENTER", ok=True, max_err_deg=0.08)
        time.sleep(0.5)

        # 模拟：运动失败
        comm.send_execution_result("PRESET_1", ok=False, reason="MoveJ错误码:-1")
        time.sleep(0.5)

        # 让轮询跑几次
        print("[测试] 等待轮询 3 次...")
        time.sleep(3.5)

    except KeyboardInterrupt:
        pass
    finally:
        comm.stop()
        print("[测试] 结束")
