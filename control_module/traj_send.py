#!/usr/bin/env python3
"""
traj_send.py — FR5 笛卡尔扫动轨迹生成 + TCP 发送

规划逻辑与 trajectory_planning 完全对齐：
  1. 笛卡尔空间按 num_waypoints 插出关键路径点（线性位置 + SLERP 姿态）
  2. 对每个关键路径点只算一次 IK
  3. 相邻两个 IK 结果之间做五次多项式关节插值（零速度+零加速度边界）
  这三步保证轨迹在加速度层面连续，与 trajectory_planning 平滑效果一致。

用法：
    python3 traj_send.py                     # 默认发到 127.0.0.1:9001
    python3 traj_send.py --host 192.168.58.1
    python3 traj_send.py --dry-run           # 只生成，不发送

依赖：conda install pinocchio -c conda-forge
"""

import argparse
import json
import socket
import struct
import uuid
import math
import numpy as np
import pinocchio as pin


# ══════════════════════════════════════════════
# 参数配置（按需修改）
# ══════════════════════════════════════════════

URDF_PATH      = "fr5.urdf"
EE_FRAME       = "wrist3_link"
Q_INIT_DEG     = [37.029, -158.192, 97.107, 48.646, -61.312, -0.279]

SWEEP_DIST     = 0.05    # 每轴扫动距离（m）
SWEEP_DURATION = 2.0     # 每轴单程时长（s）

NUM_WAYPOINTS  = 8       # 每段关键路径点数（与 trajectory_planning 默认值对齐）
DT             = 0.005   # 五次多项式插值采样间隔（s），与 planner_core dt 对齐

# IK 参数（与 planner_core 对齐）
IK_MAX_ITERS   = 200
IK_EPS         = 1e-4
IK_DAMPING     = 1e-6
IK_DT          = 0.2
IK_MAX_STEP    = 0.2

DEFAULT_HOST   = "127.0.0.1"
DEFAULT_PORT   = 9001
TCP_TIMEOUT    = 60.0


# ══════════════════════════════════════════════
# SLERP 四元数插值（与 planner_core.slerp_quaternion 对齐）
# ══════════════════════════════════════════════

def slerp_quaternion(q0, q1, t):
    """q = [w, x, y, z]"""
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return q0 + t * (q1 - q0)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1


# ══════════════════════════════════════════════
# IK（位置+姿态，与 planner_core 参数完全对齐）
# ══════════════════════════════════════════════

def solve_ik(model, data, frame_id, p_target, q_target_wxyz, q_prev):
    q = np.array(q_prev, dtype=float)

    w, x, y, z = q_target_wxyz
    R_target = np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])
    T_target = pin.SE3(R_target, np.array(p_target))

    for _ in range(IK_MAX_ITERS):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T_cur   = data.oMf[frame_id]
        err_se3 = pin.log6(T_cur.inverse() * T_target)
        err_vec = err_se3.vector

        if np.linalg.norm(err_vec) < IK_EPS:
            return q, True

        J = pin.computeFrameJacobian(
            model, data, q, frame_id,
            pin.ReferenceFrame.LOCAL)
        dq   = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), err_vec)
        step = IK_DT * dq
        if np.linalg.norm(step) > IK_MAX_STEP:
            step = step * IK_MAX_STEP / np.linalg.norm(step)
        q = np.clip(
            pin.integrate(model, q, step),
            model.lowerPositionLimit,
            model.upperPositionLimit)

    return q, False


# ══════════════════════════════════════════════
# 五次多项式关节插值（与 generate_cubic_segment 完全一致）
# 边界：零速度 + 零加速度
# ══════════════════════════════════════════════

def quintic_segment(q0, q1, duration, dt, include_start=True):
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    dq = q1 - q0
    T  = float(duration)

    a0 = q0
    a3 = 10.0 * dq / T**3
    a4 = -15.0 * dq / T**4
    a5 =   6.0 * dq / T**5

    points = []
    num_steps = int(math.floor(T / dt))
    start_idx = 0 if include_start else 1

    for i in range(start_idx, num_steps + 1):
        t = min(i * dt, T)
        q_val = a0 + a3*t**3 + a4*t**4 + a5*t**5
        points.append((t, q_val.tolist()))

    if not points or abs(points[-1][0] - T) > 1e-9:
        q_val = a0 + a3*T**3 + a4*T**4 + a5*T**5
        points.append((T, q_val.tolist()))

    return points


# ══════════════════════════════════════════════
# 单段点到点规划
# 笛卡尔路径点 → IK → 五次多项式关节插值
# 对应 PointToPointPoseTemplate.plan()
# ══════════════════════════════════════════════

def plan_segment(model, data, frame_id,
                 start_pos, start_quat,
                 end_pos, end_quat,
                 q_init, motion_time):

    # 1. 笛卡尔空间插出关键路径点
    waypoints = []
    for i in range(NUM_WAYPOINTS):
        s    = i / (NUM_WAYPOINTS - 1)
        pos  = [(1-s)*start_pos[j] + s*end_pos[j] for j in range(3)]
        quat = slerp_quaternion(start_quat, end_quat, s).tolist()
        waypoints.append((pos, quat))

    # 2. 每个路径点算一次 IK
    ik_joints = []
    q_prev = np.array(q_init, dtype=float)
    for i, (pos, quat) in enumerate(waypoints):
        if i == 0:
            ik_joints.append(q_prev.tolist())
        else:
            q_sol, ok = solve_ik(model, data, frame_id, pos, quat, q_prev)
            if not ok:
                print(f"    [警告] 第 {i} 个路径点 IK 未完全收敛")
            ik_joints.append(q_sol.tolist())
            q_prev = q_sol

    # 3. 相邻 IK 结果之间做五次多项式插值
    seg_dur   = motion_time / (len(ik_joints) - 1)
    all_pts   = []
    t_offset  = 0.0

    for seg in range(len(ik_joints) - 1):
        pts = quintic_segment(
            ik_joints[seg], ik_joints[seg+1],
            seg_dur, DT,
            include_start=(seg == 0)
        )
        for t_rel, q in pts:
            all_pts.append((round(t_rel + t_offset, 6), q))
        t_offset += seg_dur

    if all_pts:
        all_pts[-1] = (round(motion_time, 6), all_pts[-1][1])

    return all_pts, ik_joints[-1]


# ══════════════════════════════════════════════
# 单轴来回扫动（对应 BackAndForthScanPoseTemplate）
# ══════════════════════════════════════════════

def sweep_one_axis(model, data, frame_id,
                   start_pos, start_quat, q_start,
                   axis_idx, distance, duration):
    q_np = np.array(q_start, dtype=float)
    pin.forwardKinematics(model, data, q_np)
    pin.updateFramePlacements(model, data)
    axis_w = data.oMf[frame_id].rotation[:, axis_idx].tolist()

    fwd_pos  = [start_pos[j] + distance * axis_w[j] for j in range(3)]
    back_pos = start_pos[:]

    all_points = []
    t_base     = 0.0
    q_cur      = q_start[:]
    axis_name  = ["x", "y", "z"][axis_idx]

    for seg_label, p_a, q_a, p_b, q_b in [
        ("去程", start_pos, start_quat, fwd_pos,  start_quat),
        ("返程", fwd_pos,  start_quat, back_pos, start_quat),
    ]:
        pts, q_cur = plan_segment(
            model, data, frame_id,
            p_a, q_a, p_b, q_b,
            q_cur, duration,
        )
        print(f"    {axis_name} 轴{seg_label}完成，{len(pts)} 个点")
        for t_rel, q in pts:
            all_points.append((round(t_rel + t_base, 6), q))
        t_base = all_points[-1][0] + DT

    return all_points, q_cur


# ══════════════════════════════════════════════
# 生成三轴完整扫动轨迹
# ══════════════════════════════════════════════

def generate_trajectory():
    print("\n[1/3] 加载 URDF，生成三轴扫动轨迹...")

    model    = pin.buildModelFromUrdf(URDF_PATH)
    data     = model.createData()
    frame_id = model.getFrameId(EE_FRAME)

    if frame_id >= model.nframes:
        raise RuntimeError(f"找不到 frame '{EE_FRAME}'，请检查 URDF_PATH 和 EE_FRAME 配置")

    q_init = np.deg2rad(Q_INIT_DEG)
    pin.forwardKinematics(model, data, q_init)
    pin.updateFramePlacements(model, data)

    T0         = data.oMf[frame_id]
    start_pos  = T0.translation.tolist()
    quat_obj   = pin.Quaternion(T0.rotation)
    start_quat = [quat_obj.w, quat_obj.x, quat_obj.y, quat_obj.z]

    print(f"  末端初始位置: {np.round(T0.translation, 4)}")
    print(f"  末端初始四元数(wxyz): {[round(v,4) for v in start_quat]}")

    all_points = []
    t_base     = 0.0
    q_cur      = q_init.tolist()
    cur_pos    = start_pos[:]
    cur_quat   = start_quat[:]

    for axis_idx in range(3):
        print(f"\n  生成 {'xyz'[axis_idx]} 轴扫动...")
        pts, q_cur = sweep_one_axis(
            model, data, frame_id,
            cur_pos, cur_quat, q_cur,
            axis_idx, SWEEP_DIST, SWEEP_DURATION,
        )
        for t_rel, q in pts:
            all_points.append({"t": round(t_rel + t_base, 6), "q": q})
        t_base = all_points[-1]["t"] + DT

        # 更新下一轴起点
        q_np = np.array(q_cur, dtype=float)
        pin.forwardKinematics(model, data, q_np)
        pin.updateFramePlacements(model, data)
        T_cur    = data.oMf[frame_id]
        cur_pos  = T_cur.translation.tolist()
        qo       = pin.Quaternion(T_cur.rotation)
        cur_quat = [qo.w, qo.x, qo.y, qo.z]

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

    print(f"\n  生成完成：{payload['point_count']} 个点，总时长 {payload['duration']}s")
    return payload


# ══════════════════════════════════════════════
# TCP 发送
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
        ack        = json.loads(_recv_exact(sock, ack_len).decode("utf-8"))

    return ack


# ══════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="三轴扫动轨迹生成 + 发送给运控")
    parser.add_argument("--host",    default=DEFAULT_HOST)
    parser.add_argument("--port",    default=DEFAULT_PORT, type=int)
    parser.add_argument("--dry-run", action="store_true", help="只生成，不发送")
    parser.add_argument("--save",    default="traj_sweep.json")
    args = parser.parse_args()

    print("=" * 55)
    print("  FR5 笛卡尔三轴扫动轨迹（五次多项式平滑）")
    print(f"  起始角(°): {Q_INIT_DEG}")
    print(f"  扫动距离: ±{SWEEP_DIST*100:.1f}cm  每轴单程: {SWEEP_DURATION}s")
    print(f"  关键路径点: {NUM_WAYPOINTS}  插值dt: {DT}s")
    print("=" * 55)

    payload = generate_trajectory()

    with open(args.save, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  轨迹已保存到 {args.save}")

    if args.dry_run:
        print("\n[dry-run] 不发送，退出。")
        return

    try:
        ack = send_trajectory(payload, args.host, args.port)
        print(f"\n[3/3] ACK:")
        print(f"  ok      = {ack.get('ok')}")
        print(f"  message = {ack.get('message')}")
        if ack.get("ok"):
            print(f"  ✓ 运控已接收，正在执行（共 {payload['duration']}s）")
        else:
            print(f"  ✗ 运控拒绝：{ack.get('message')}")
    except ConnectionRefusedError:
        print(f"\n  [错误] 连接 {args.host}:{args.port} 被拒绝")
        print("  请确认 fr5_controller_sdk_v12.py 已启动，状态机处于 AUTO")
    except Exception as e:
        print(f"\n  [错误] 发送失败: {e}")

    print("=" * 55)


if __name__ == "__main__":
    main()
