#!/usr/bin/env python3
"""
mock_host_server.py — 模拟上位机，用于联调测试

在你自己电脑上跑这个脚本，它会：
  · 监听 TCP 端口，等你的 fr5_controller_sdk_v11.py 来连
  · 收到 get_system_state 请求时，自动回复当前模拟状态
  · 收到 execution_result / state_change 时，打印出来
  · 支持键盘输入，手动切换模拟的系统状态

用法：
  python3 mock_host_server.py              # 默认监听 0.0.0.0:9000
  python3 mock_host_server.py 9001         # 指定端口

测试流程：
  1. 先跑这个脚本
  2. 再跑 fr5_controller_sdk_v11.py（或者直接跑 host_comm.py 的测试入口）
  3. 看两边的打印，确认消息收发正常
"""

import json
import socket
import threading
import time
import sys


# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
LISTEN_IP   = "0.0.0.0"
LISTEN_PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 9000

# 模拟的系统状态机状态（键盘输入可切换）
MOCK_STATES = ["MANUAL", "ASSIST", "AUTO", "HOLD", "ERROR", "ESTOP"]
_current_state = "AUTO"
_state_lock    = threading.Lock()


def get_mock_state() -> str:
    with _state_lock:
        return _current_state

def set_mock_state(s: str):
    global _current_state
    with _state_lock:
        _current_state = s


# ──────────────────────────────────────────────
# 处理单个客户端连接
# ──────────────────────────────────────────────

def handle_client(conn: socket.socket, addr):
    print(f"\n  [Server] 客户端已连接: {addr}")
    buf = b""
    try:
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk

            # 按 \n 拆分消息
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8"))
                    _dispatch(conn, msg)
                except json.JSONDecodeError as e:
                    print(f"  [Server] JSON解析失败: {e}  原始: {line!r}")

    except (ConnectionResetError, OSError) as e:
        print(f"  [Server] 连接断开: {e}")
    finally:
        conn.close()
        print(f"  [Server] 客户端已断开: {addr}")


def _dispatch(conn: socket.socket, msg: dict):
    """根据消息类型分发处理。"""
    t = msg.get("type", "unknown")

    if t == "get_system_state":
        # 收到拉取请求 → 回复当前模拟状态
        state = get_mock_state()
        reply = {
            "type" : "system_state",
            "state": state,
            "t"    : round(time.time(), 3),
        }
        _send(conn, reply)
        print(f"  [Server] ← get_system_state  回复: {state}")

    elif t == "execution_result":
        ok     = msg.get("ok", False)
        cmd    = msg.get("cmd", "?")
        err    = msg.get("max_err_deg", 0)
        reason = msg.get("reason", "")
        status = f"成功 误差={err}°" if ok else f"失败 原因={reason}"
        print(f"  [Server] ← execution_result  [{cmd}] {status}")

    elif t == "state_change":
        old    = msg.get("old", "?")
        new    = msg.get("new", "?")
        reason = msg.get("reason", "")
        print(f"  [Server] ← state_change  {old} → {new}  ({reason})")

    else:
        print(f"  [Server] ← 未知消息类型: {msg}")


def _send(conn: socket.socket, msg: dict):
    try:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        conn.sendall(data)
    except OSError as e:
        print(f"  [Server] 发送失败: {e}")


# ──────────────────────────────────────────────
# 键盘输入线程（手动切换模拟状态）
# ──────────────────────────────────────────────

def keyboard_loop():
    print("\n  键盘指令：")
    print("  直接输入状态名切换，如: AUTO  MANUAL  ERROR  ESTOP")
    print(f"  可选: {MOCK_STATES}")
    print("  q → 退出\n")
    while True:
        try:
            cmd = input("  mock> ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd == "Q":
            break
        if cmd in MOCK_STATES:
            set_mock_state(cmd)
            print(f"  [Server] 模拟状态已切换为: {cmd}")
        else:
            print(f"  [Server] 未知状态: {cmd}  可选: {MOCK_STATES}")


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_IP, LISTEN_PORT))
    server.listen(5)
    server.settimeout(1.0)   # 让 accept 可被 KeyboardInterrupt 打断

    print("=" * 55)
    print("  模拟上位机 (mock_host_server)")
    print(f"  监听: {LISTEN_IP}:{LISTEN_PORT}")
    print(f"  初始模拟状态: {get_mock_state()}")
    print("=" * 55)

    # 键盘输入线程
    kb_thread = threading.Thread(target=keyboard_loop, daemon=True)
    kb_thread.start()

    try:
        while True:
            try:
                conn, addr = server.accept()
                t = threading.Thread(
                    target=handle_client, args=(conn, addr), daemon=True
                )
                t.start()
            except socket.timeout:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        print("\n  [Server] 已关闭")


if __name__ == "__main__":
    main()
