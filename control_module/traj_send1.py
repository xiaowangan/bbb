#!/usr/bin/env python3
"""
traj_send.py — 轨迹生成 + TCP 发送一体脚本

用法：
    python3 traj_send.py                          # 默认发到 127.0.0.1:9001
    python3 traj_send.py --host 192.168.58.1      # 指定运控 IP
    python3 traj_send.py --dry-run                # 只生成轨迹，不发送（验证用）

整体流程：
    1. 用 pinocchio 做 IK，生成笛卡尔三轴扫动轨迹（rad，JSON格式）
    2. 通过 TCP 发给 fr5_controller_sdk_v12.py 里的 traj_server（端口9001）
    3. 协议：4字节大端长度头 + UTF-8 JSON 正文，收到 ACK 后退出

依赖：
    conda install pinocchio -c conda-forge
"""

import argparse
import json
import socket
import struct
import uuid
import numpy as np
import pinocchio as pin


# ══════════════════════════════════════════════
# 参数配置（按需修改）
# ══════════════════════════════════════════════

URDF_PATH      = "fr5.urdf"          # URDF 路径，与本脚本同目录
EE_FRAME       = "wrist3_link"       # 末端执行器 frame 名
Q_INIT_DEG     = [37.029, -158.192, 97.107, 48.646, -61.312, -0.279]  # 起始关节角（°）
SWEEP_DIST     = 0.05                # 每轴扫动距离（m）
SWEEP_DURATION = 2.0                 # 每轴单程时长（s）
DT             = 0.05                # 轨迹采样间隔（s）

DEFAULT_HOST   = "127.0.0.1"
DEFAULT_PORT   = 9001
TCP_TIMEOUT    = 60.0                # 等待 ACK 超时（s，轨迹执行期间保持连接）


# ══════════════════════════════════════════════
# 五次样条（平滑插值）
# ══════════════════════════════════════════════

def quintic(s: float) -> float:
    s = float(np.clip(s, 0.0, 1.0))
    return 10*s**3 - 15*s**4 + 6*s**5


# ══════════════════════════════════════════════
# IK 求解
# ══════════════════════════════════════════════

def ik(model, data, frame_id, p_target, q_prev):
    q = q_prev.copy()
    for _ in range(2000):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        err = p_target - data.oMf[frame_id].translation
        if np.linalg.norm(err) < 1e-5:
            return q, True
        J  = pin.computeFrameJacobian(
                model, data, q, frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3, :]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
        q  = np.clip(
                pin.integrate(model, q, 0.1 * dq),
                model.lowerPositionLimit,
                model.upperPositionLimit)
    return q, False


# ══════════════════════════════════════════════
# 单轴来回扫动
# ══════════════════════════════════════════════

def sweep_one_axis(model, data, frame_id, q_start, axis_idx, distance, duration, dt):
    pin.forwardKinematics(model, data, q_start)
    pin.updateFramePlacements(model, data)
    p0     = data.oMf[frame_id].translation.copy()
    axis_w = data.oMf[frame_id].rotation[:, axis_idx]
    p_fwd  = p0 + distance * axis_w
    p_back = p0.copy()

    points   = []
    t_offset = 0.0
    q_prev   = q_start.copy()
    fail_cnt = 0

    for p_a, p_b in [(p0, p_fwd), (p_fwd, p_back)]:
        times = np.arange(0.0, duration + dt * 0.5, dt)
        for t in times:
            alpha     = quintic(t / duration)
            p_target  = p_a + alpha * (p_b - p_a)
            q_sol, ok = ik(model, data, frame_id, p_target, q_prev)
            if not ok:
                fail_cnt += 1
            q_prev = q_sol
            points.append({
                "t": round(float(t + t_offset), 6),
                "q": [round(float(v), 8) for v in q_sol],
            })
        t_offset += float(times[-1]) + dt

    axis_name = ["x", "y", "z"][axis_idx]
    if fail_cnt > 0:
        print(f"    [警告] {axis_name} 轴有 {fail_cnt} 个点未完全收敛")
    else:
        print(f"    {axis_name} 轴全部收敛 ✓")

    return points


# ══════════════════════════════════════════════
# 生成完整三轴扫动轨迹
# ══════════════════════════════════════════════

def generate_trajectory():
    print("\n[1/3] 加载 URDF 并生成轨迹...")

    model    = pin.buildModelFromUrdf(URDF_PATH)
    data     = model.createData()
    frame_id = model.getFrameId(EE_FRAME)

    if frame_id >= model.nframes:
        raise RuntimeError(f"找不到 frame '{EE_FRAME}'，请检查 URDF_PATH 和 EE_FRAME 配置")

    q_init = np.deg2rad(Q_INIT_DEG)
    pin.forwardKinematics(model, data, q_init)
    pin.updateFramePlacements(model, data)
    print(f"  末端初始位置: {np.round(data.oMf[frame_id].translation, 4)}")

    all_points = []
    t_base     = 0.0

    for axis_idx in range(3):
        q_cur = np.array(all_points[-1]["q"]) if all_points else q_init
        pts   = sweep_one_axis(
            model, data, frame_id,
            q_start  = q_cur,
            axis_idx = axis_idx,
            distance = SWEEP_DIST,
            duration = SWEEP_DURATION,
            dt       = DT,
        )
        for pt in pts:
            all_points.append({
                "t": round(pt["t"] + t_base, 6),
                "q": pt["q"],
            })
        t_base = all_points[-1]["t"] + DT

    payload = {
        "type":          "trajectory",
        "trajectory_id": str(uuid.uuid4()),
        "case_name":     "[scan] EE三轴扫动",
        "joint_names":   ["j1", "j2", "j3", "j4", "j5", "j6"],
        "angle_unit":    "rad",
        "point_count":   len(all_points),
        "duration":      round(all_points[-1]["t"], 3),
        "points":        all_points,
    }

    print(f"  生成完成：{payload['point_count']} 个点，总时长 {payload['duration']}s")
    return payload


# ══════════════════════════════════════════════
# TCP 发送（4字节头 + JSON，等 ACK）
# ══════════════════════════════════════════════

def _recv_exact(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("连接关闭，未收到完整数据")
        buf += chunk
    return buf


def send_trajectory(payload, host, port):
    print(f"\n[2/3] 连接运控 {host}:{port} ...")

    body   = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    header = struct.pack("!I", len(body))

    with socket.create_connection((host, port), timeout=TCP_TIMEOUT) as sock:
        sock.settimeout(TCP_TIMEOUT)
        sock.sendall(header + body)
        print(f"  已发送 {len(body)} 字节，等待 ACK...")

        ack_header = _recv_exact(sock, 4)
        ack_len    = struct.unpack("!I", ack_header)[0]
        ack_body   = _recv_exact(sock, ack_len)
        ack        = json.loads(ack_body.decode("utf-8"))

    return ack


# ══════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="轨迹生成 + 发送给运控")
    parser.add_argument("--host",    default=DEFAULT_HOST, help="运控 IP（默认 127.0.0.1）")
    parser.add_argument("--port",    default=DEFAULT_PORT, type=int, help="运控轨迹端口（默认 9001）")
    parser.add_argument("--dry-run", action="store_true",  help="只生成轨迹，不发送")
    parser.add_argument("--save",    default="traj_sweep.json", help="保存轨迹 JSON 到文件（默认 traj_sweep.json）")
    args = parser.parse_args()

    print("=" * 55)
    print("  FR5 笛卡尔扫动轨迹生成 + 发送")
    print(f"  起始角(°): {Q_INIT_DEG}")
    print(f"  扫动距离: ±{SWEEP_DIST*100:.1f}cm  每轴时长: {SWEEP_DURATION*2}s  dt: {DT}s")
    print("=" * 55)

    # 生成轨迹
    payload = generate_trajectory()

    # 保存 JSON
    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  轨迹已保存到 {args.save}")

    if args.dry_run:
        print("\n[dry-run 模式] 不发送，退出。")
        return

    # 发送
    try:
        ack = send_trajectory(payload, args.host, args.port)
        print(f"\n[3/3] ACK 收到:")
        print(f"  ok      = {ack.get('ok')}")
        print(f"  message = {ack.get('message')}")
        print(f"  traj_id = {ack.get('trajectory_id')}")
        if ack.get("ok"):
            print(f"\n  ✓ 运控已接收轨迹，正在执行（共 {payload['duration']}s）")
        else:
            print(f"\n  ✗ 运控拒绝了轨迹，原因：{ack.get('message')}")
    except ConnectionRefusedError:
        print(f"\n  [错误] 连接 {args.host}:{args.port} 被拒绝")
        print("  请确认 fr5_controller_sdk_v12.py 已启动，且状态机处于 AUTO 状态")
    except Exception as e:
        print(f"\n  [错误] 发送失败: {e}")

    print("=" * 55)


if __name__ == "__main__":
    main()
