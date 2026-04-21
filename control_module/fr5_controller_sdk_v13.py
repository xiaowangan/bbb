#!/usr/bin/env python3
"""
FR5 运动控制节点 V13 —— ServoJ 伺服版本
fairino SDK + ServoJ伺服模式 + SystemStateMachine + 回中 + HostComm + TrajectoryServer

核心改动（V12 → V13）：
  · _movej() 替换为 _servoj()，使用 ServoMoveStart / ServoJ / ServoMoveEnd
  · blendT 参数（BLEND_MID / BLEND_END）完全废弃
  · 平滑过渡依赖 S 曲线插值 + ServoJ 高频发送，天然平滑
  · SERVO_CMD_T（指令周期）替代 blendT 成为核心平滑参数（建议 0.008s）
  · _execute_move 和 execute_trajectory 均改为 ServoJ 模式
  · stop_motion / trigger_estop 额外调用 ServoMoveEnd 确保伺服模式正确退出
  · execute_trajectory 直接按轨迹模块给的 t/q 逐点 ServoJ 推送，按时间戳控制节拍

指令说明：
  0/1/2        → 运动到预设点位（需处于 AUTO/MANUAL/ASSIST 状态）
  m <方向>     → 小幅微调
    up/down    → 末端上抬/下压（j5）
    left/right → 底座左转/右转（j1）
    fw/bk      → 前推/后退（j2）
    tilt+/-    → 末端旋转（j6）
    示例: m up      (默认3°)
          m left 5  (左转5°)
  setcenter    → 记录当前位置为中心点（写入 center_config.json）
  recenter     → 回到已记录的中心点
  r <点位>     → 10次重复可复现测试
  c <点位>     → 原始 vs 平滑对比测试
  hold         → 锁定（→ HOLD 状态），拒绝新运动指令
  resume       → 解锁（HOLD → 恢复之前的状态）
  stop         → 停止（中断当前运动，状态机保持当前功能态）
  recover      → 从 ERROR / ERROR_END 恢复 → MANUAL
  estop        → 急停（→ ESTOP）
  reset        → 急停复位（ESTOP → INIT → START_READY → MANUAL）
  status       → 查看系统状态机状态
  joints       → 查看当前关节角度
  center       → 查看当前记录的中心点
  mode <模式>  → 手动切换工作模式 manual / assist / auto
  help         → 显示帮助
  q            → 退出
"""

import time
import math
import threading
import csv
import json
import os

# ── 导入通信模块 ──────────────────────────────
from host_comm import HostComm
from traj_server import TrajectoryServer

# ── 导入新状态机模块 ──────────────────────────
from system_state_machine import SystemStateMachine, SysState

try:
    from fairino import Robot
except ImportError:
    print("=" * 60)
    print("  [错误] 找不到 fairino 模块！")
    print("  请把SDK里的 fairino 文件夹复制到本文件同目录")
    print("=" * 60)
    exit(1)

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────
ROBOT_IP  = "192.168.58.2"
SDK_SPEED = 20      # 速度百分比 1~100

# 上位机通信配置
HOST_IP   = "192.168.112.95"   # 上位机 IP
HOST_PORT = 9000              # 上位机监听端口

# 轨迹接收配置
TRAJ_PORT = 9001              # 本机监听，等轨迹模块来连

# ── ServoJ 核心参数 ──────────────────────────
# blendT 在 ServoJ 模式下已完全废弃。
# SERVO_CMD_T 控制每帧发送间隔，承担原 blendT 的平滑作用。
SERVO_CMD_T  = 0.008   # 指令下发周期（s），建议 0.001~0.016
SERVO_FILTER = 0.0     # 滤波时间（暂不开放），保持 0.0
SERVO_GAIN   = 0.0     # 增益（暂不开放），保持 0.0

# ──────────────────────────────────────────────
# 预设位置（角度°，j1~j6）
# ──────────────────────────────────────────────
PRESETS = {
    "0": {
        "name": "位置A（实测）",
        "joints": [37.029, -158.192, 97.107, 48.646, -61.312, -0.279],
    },
    "1": {
        "name": "位置B（实测）",
        "joints": [52.691, -160.206, 95.976, 53.479, -52.447, -0.272],
    },
    "2": {
        "name": "位置C（A和B中间）",
        "joints": [44.860, -159.199, 96.542, 51.063, -56.880, -0.276],
    },
}

# ──────────────────────────────────────────────
# 小幅微调配置
# ──────────────────────────────────────────────
MICRO_STEP_DEFAULT = 3.0
MICRO_STEP_MAX     = 10.0

MICRO_DIRS = {
    "up":    (4,  1, "末端上抬（j5+）"),
    "down":  (4, -1, "末端下压（j5-）"),
    "left":  (0,  1, "底座左转（j1+）"),
    "right": (0, -1, "底座右转（j1-）"),
    "fw":    (1,  1, "前推（j2+）"),
    "bk":    (1, -1, "后退（j2-）"),
    "tilt+": (5,  1, "末端顺时针旋转（j6+）"),
    "tilt-": (5, -1, "末端逆时针旋转（j6-）"),
}

# ──────────────────────────────────────────────
# 运动参数
# ──────────────────────────────────────────────
MAX_JOINT_VEL = 45.0
MAX_JOINT_ACC = 22.9
MIN_DURATION  = 0.5
INTERP_STEPS  = 20

ERROR_TIMEOUT = 30.0     # 原 FAULT_TIMEOUT，改名对应新状态
ERROR_MAX_ERR = 2.87     # 原 FAULT_MAX_ERR

LOG_FILE          = "motion_log.csv"
CENTER_CONFIG     = "center_config.json"   # 中心点持久化文件
JOINT_NAMES       = ["j1", "j2", "j3", "j4", "j5", "j6"]

JOINT_LIMITS = [
    (-170.0, 170.0),
    (-265.0,  85.0),
    (-160.0, 160.0),
    (-265.0,  85.0),
    (-170.0, 170.0),
    (-360.0, 360.0),
]


# ══════════════════════════════════════════════
# 工具函数（不变）
# ══════════════════════════════════════════════

def calc_duration(current: list, target: list,
                  min_dur: float = MIN_DURATION) -> float:
    t_min = min_dur
    for c, t in zip(current, target):
        delta = abs(t - c)
        if delta == 0:
            continue
        t_min = max(t_min,
                    delta / MAX_JOINT_VEL,
                    math.sqrt(2.0 * delta / MAX_JOINT_ACC))
    return round(t_min, 3)


def s_curve_interp(current: list, target: list,
                   steps: int = INTERP_STEPS) -> list:
    waypoints = []
    for i in range(1, steps + 1):
        s = i / steps
        alpha = s - math.sin(2 * math.pi * s) / (2 * math.pi)
        point = [round(c + (t - c) * alpha, 4)
                 for c, t in zip(current, target)]
        waypoints.append(point)
    return waypoints


def clamp_joints(joints: list) -> tuple:
    clamped, hit = [], False
    for j, (lo, hi) in zip(joints, JOINT_LIMITS):
        if j < lo:
            clamped.append(lo); hit = True
        elif j > hi:
            clamped.append(hi); hit = True
        else:
            clamped.append(j)
    return clamped, hit


# ══════════════════════════════════════════════
# 主控制类
# ══════════════════════════════════════════════

class FR5Controller:

    def __init__(self, robot_ip: str, comm: "HostComm | None" = None):
        # ── 实例化新状态机 ───────────────────
        self.sm = SystemStateMachine(verbose=True)
        self.sm.register_callback(self._on_state_change)

        # ── 上位机通信（可选，None 表示不通信）──
        self.comm: "HostComm | None" = comm

        # ── 运动内部标志（不再作为状态机主状态） ──
        self._is_moving   = False           # 是否正在执行运动指令
        self._paused      = False           # PauseMotion 暂停中
        self._stop_event  = threading.Event()  # stop/estop 触发后 set()，运动循环立即感知
        self._move_lock   = threading.Lock()

        # ── 回中：中心点存储 ──────────────────
        self._center_joints: list | None = None
        self._load_center()

        print(f"  [连接] 正在连接控制箱 {robot_ip} ...")
        try:
            self.robot = Robot.RPC(robot_ip)
            print("  [连接] 连接成功")
        except Exception as e:
            print(f"  [连接] 连接失败: {e}")
            exit(1)
        # PauseMotion / ResumeMotion / StopMove 均通过 send_message 走独立 TCP
        # 到 8080 端口，无需第二条 RPC 连接

        self._init_robot()

        self._csv_file = open(LOG_FILE, "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow([
            "timestamp", "event", "command", "sys_state", "smooth_mode",
            "duration_planned_s", "duration_actual_s",
            "j1_target","j2_target","j3_target","j4_target","j5_target","j6_target",
            "j1_actual","j2_actual","j3_actual","j4_actual","j5_actual","j6_actual",
            "j1_error","j2_error","j3_error","j4_error","j5_error","j6_error",
            "max_error_deg", "error_reason",
        ])

    # ──────────────────────────────────────────
    # 状态机回调（可扩展：联动上层 ROS 话题等）
    # ──────────────────────────────────────────
    def _on_state_change(self, old: str, new: str, reason: str):
        """本地状态机发生转移时自动触发，通过 comm 通知上位机。"""
        if self.comm:
            self.comm.on_state_change(old, new, reason)

    # ──────────────────────────────────────────
    # 回中：中心点持久化（读 / 写）
    # ──────────────────────────────────────────
    def _load_center(self):
        """启动时从 center_config.json 加载中心点，文件不存在则跳过。"""
        if not os.path.exists(CENTER_CONFIG):
            print("  [回中] 未找到 center_config.json，中心点未设置")
            return
        try:
            with open(CENTER_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
            joints = data.get("center_joints")
            if joints and len(joints) == 6:
                self._center_joints = [float(v) for v in joints]
                print(f"  [回中] 已加载中心点: {[round(v,3) for v in self._center_joints]}")
            else:
                print("  [回中] center_config.json 格式异常，中心点未加载")
        except Exception as e:
            print(f"  [回中] 加载中心点失败: {e}")

    def _save_center(self):
        """将当前中心点写入 center_config.json。"""
        try:
            data = {
                "center_joints": [round(v, 4) for v in self._center_joints],
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(CENTER_CONFIG, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  [回中] 中心点已保存到 {CENTER_CONFIG}")
        except Exception as e:
            print(f"  [回中] 保存中心点失败: {e}")

    # ──────────────────────────────────────────
    # 回中：对外接口
    # ──────────────────────────────────────────
    def set_center(self) -> bool:
        """
        将当前关节角记录为中心点，同时持久化到 center_config.json。
        可在任意状态下调用（不需要 can_move）。
        """
        joints = self._get_current_joints()
        if joints is None:
            print("  [回中] 无法读取当前关节角，中心点未记录")
            return False
        self._center_joints = joints
        print(f"  [回中] 中心点已记录: {[round(v,3) for v in joints]}")
        self._save_center()
        return True

    def recenter(self) -> bool:
        """
        回到已记录的中心点（平滑运动）。
        未记录时提示用户先执行 setcenter。
        受状态机 can_move() 约束：HOLD/ERROR/ESTOP 下拒绝执行。
        """
        with self._move_lock:
            if not self._check_can_move():
                return False

        if self._center_joints is None:
            print("  [回中] 尚未记录中心点，请先执行 setcenter 命令")
            return False

        print(f"\n  [回中] 正在回中...")
        print(f"    目标中心点: {[round(v,3) for v in self._center_joints]}")
        return self._execute_move(
            self._center_joints,
            smooth=True,
            command_tag="RECENTER",
        )

    def show_center(self):
        """打印当前记录的中心点信息。"""
        if self._center_joints is None:
            print("  [回中] 中心点未设置，请先执行 setcenter")
        else:
            print(f"  [回中] 当前中心点(°): {[round(v,3) for v in self._center_joints]}")

    # ──────────────────────────────────────────
    # 初始化：INIT → START_READY → AUTO（当前版本直接进 AUTO）
    # ──────────────────────────────────────────
    def _init_robot(self):
        try:
            self.robot.ResetAllError()
            state = self.robot.GetRobotState()
            print(f"  [初始化] 机械臂状态: {state}")
        except Exception as e:
            print(f"  [初始化] 警告: {e}")

        # 状态机启动流程
        self.sm.startup()           # INIT → START_READY
        self.sm.to_auto()           # START_READY → MANUAL → AUTO_READY → AUTO
        # 当前版本以 AUTO 作为常规运行模式
        print(f"  [初始化] 系统状态机: {self.sm.state}")

    # ──────────────────────────────────────────
    # 读取关节角度
    # ──────────────────────────────────────────
    def _get_current_joints(self) -> list:
        try:
            ret = self.robot.GetActualJointPosDegree()
            if isinstance(ret, (list, tuple)) and len(ret) == 2:
                code, joints = ret
                if code == 0:
                    return list(joints)
            print(f"  [警告] 读取关节角度失败: {ret}")
            return None
        except Exception as e:
            print(f"  [警告] 读取关节角度异常: {e}")
            return None

    # ──────────────────────────────────────────
    # ★ 核心替换：_servoj 替代 _movej
    # ──────────────────────────────────────────
    def _servoj(self, joint_list: list) -> int:
        try:
            jp = list(map(float, joint_list))
            ret = self.robot.ServoJ(joint_pos=jp, axisPos=[0, 0, 0, 0])
            return ret if ret is not None else 0
        except Exception as e:
            raise e

    # ──────────────────────────────────────────
    # 运动前检查（使用新状态机）
    # ──────────────────────────────────────────
    def _check_can_move(self) -> bool:
        """检查当前状态机是否允许运动"""
        if self._is_moving:
            print("  [警告] 机械臂正在运动，请等待")
            return False
        return self.sm.can_move()

    # ──────────────────────────────────────────
    # 锁定（HOLD）
    # ──────────────────────────────────────────
    def hold(self):
        """
        锁定：进入 HOLD 状态，拒绝新运动指令。
        仅切换状态机，不主动停止正在进行的运动
        （若需同时停止运动，先 stop 再 hold，或使用 p 暂停）。
        """
        if self.sm.state == SysState.HOLD:
            print("  [HOLD] 已处于锁定状态")
            return
        ok = self.sm.hold()
        if ok:
            print("  [HOLD] 已锁定，新运动指令将被拒绝。输入 resume 解锁")

    def resume_from_hold(self):
        """解锁：HOLD → 恢复之前的工作状态，可重新接受运动指令。"""
        ok = self.sm.resume_from_hold()
        if ok:
            print(f"  [解锁] 已恢复到 {self.sm.state}，可接受新指令")

    # ──────────────────────────────────────────
    # 运动暂停 / 续跑（PauseMotion / ResumeMotion）
    # ──────────────────────────────────────────
    def toggle_pause(self):
        """
        p 指令：暂停 / 续跑正在进行的运动。
        PauseMotion / ResumeMotion 走 send_message 独立 TCP（8080端口），
        可真正抢占 MoveJ，原地暂停后可续跑剩余路径。
        不改变状态机状态。
        """
        if not self._is_moving:
            print("  [暂停] 当前没有正在进行的运动")
            return
        if not self._paused:
            try:
                self.robot.PauseMotion()
                self._paused = True
                print("  [暂停] 机械臂已暂停，输入 p 继续")
            except Exception as e:
                print(f"  [暂停] PauseMotion 异常: {e}")
        else:
            try:
                self.robot.ResumeMotion()
                self._paused = False
                print("  [续跑] 机械臂继续运动")
            except Exception as e:
                print(f"  [续跑] ResumeMotion 异常: {e}")

    # ──────────────────────────────────────────
    # 停止（丢弃路径，不切换状态机大状态）
    # ──────────────────────────────────────────
    def stop_motion(self):
        """
        立即停止当前运动。
        ServoJ 模式下必须先 ServoMoveEnd，再 StopMove，
        否则控制器可能保持伺服激活状态导致下次 ServoMoveStart 失败。
        """
        if not self._is_moving:
            print("  [停止] 当前没有正在进行的运动")
            return
        print("  [停止] 正在发送停止指令...")
        self._stop_event.set()
        try:
            self.robot.ServoMoveEnd()   # ★ 退出伺服模式
        except Exception as e:
            print(f"  [停止] ServoMoveEnd 异常: {e}")
        try:
            self.robot.StopMove()
            print("  [停止] 硬件停止指令已送达")
        except Exception as e:
            print(f"  [停止] StopMove 异常: {e}")

    # ──────────────────────────────────────────
    # 错误恢复（原 recover_from_fault）
    # ──────────────────────────────────────────
    def recover_error(self):
        """ERROR / ERROR_END → MANUAL"""
        if self.sm.state not in (SysState.ERROR, SysState.ERROR_END):
            print(f"  [警告] 当前不在 ERROR 状态（当前: {self.sm.state}）")
            return
        try:
            self.robot.ResetAllError()
        except Exception:
            pass
        ok = self.sm.recover_error()
        if ok:
            print("  [恢复] 已恢复到 MANUAL，可重新切换工作模式")

    # ──────────────────────────────────────────
    # 急停 / 复位
    # ──────────────────────────────────────────
    def trigger_estop(self, reason: str = "操作员急停"):
        """任意状态 → ESTOP，立即停止运动"""
        self._stop_event.set()
        self.sm.estop(reason)
        try:
            self.robot.ServoMoveEnd()   # ★ 确保退出伺服模式
        except Exception:
            pass
        try:
            self.robot.StopMove()
        except Exception:
            pass

    def reset_estop(self):
        """ESTOP → INIT → START_READY → MANUAL（完整复位流程）"""
        if not self.sm.reset_estop():
            return
        try:
            self.robot.ResetAllError()
        except Exception:
            pass
        self.sm.startup()    # INIT → START_READY
        self.sm.to_manual()  # START_READY → MANUAL
        print("  [复位] 急停已复位，当前 MANUAL，请确认安全后切换工作模式")

    # ──────────────────────────────────────────
    # 工作模式切换（手动调用）
    # ──────────────────────────────────────────
    def switch_mode(self, mode: str):
        mode = mode.lower().strip()
        if mode == "manual":
            ok = self.sm.to_manual()
        elif mode == "assist":
            ok = self.sm.to_assist()
        elif mode == "auto":
            ok = self.sm.to_auto()
        else:
            print(f"  [模式] 未知模式: {mode}  可选: manual / assist / auto")
            return
        if ok:
            print(f"  [模式] 已切换到 {self.sm.state}")

    # ──────────────────────────────────────────
    # 通用运动执行（内部）
    # ──────────────────────────────────────────
    def _execute_move(self, target: list, smooth: bool,
                      command_tag: str,
                      min_dur: float = MIN_DURATION) -> bool:
        current = self._get_current_joints()
        if current is None:
            print("  [错误] 无法读取当前关节角度")
            return False

        duration = calc_duration(current, target, min_dur)

        # smooth=True  → S曲线插值，步数多，过渡柔顺
        # smooth=False → S曲线插值，步数少，到位更直接
        # 两者都通过 ServoJ 高频推送，不再有 blendT
        if smooth:
            steps    = max(10, int(duration / SERVO_CMD_T * 0.8))
            mode_str = "ServoJ-平滑(S曲线)"
        else:
            steps    = max(5,  int(duration / SERVO_CMD_T * 0.4))
            mode_str = "ServoJ-直达(少插值)"
        waypoints = s_curve_interp(current, target, steps)

        print(f"    目标(°): {[round(v,3) for v in target]}")
        print(f"    当前(°): {[round(v,3) for v in current]}")
        print(f"    预计时间: {duration:.2f}s  模式:{mode_str}  插值步数:{steps}")

        self._is_moving = True
        self._stop_event.clear()
        t_start      = time.time()
        ok           = False
        error_reason = ""

        try:
            # ── 开启伺服模式 ──────────────────────
            ret_start = self.robot.ServoMoveStart()
            if ret_start != 0:
                error_reason = f"ServoMoveStart 失败，错误码:{ret_start}"
                raise RuntimeError(error_reason)

            for i, wp in enumerate(waypoints):
                if self._stop_event.is_set():
                    error_reason = "stop指令中断"
                    break
                if self.sm.state in (SysState.ESTOP, SysState.ERROR):
                    error_reason = f"状态机中断({self.sm.state})"
                    break

                ret = self._servoj(wp)
                if ret != 0:
                    if self._stop_event.is_set():
                        error_reason = "stop指令中断"
                    else:
                        error_reason = f"ServoJ错误码:{ret} (第{i+1}帧)"
                    break

                time.sleep(SERVO_CMD_T)

            else:
                # 所有插值帧已发完，轮询等待机械臂到位
                settle_timeout = duration + 2.0
                t_settle = time.time()
                while time.time() - t_settle < settle_timeout:
                    if self._stop_event.is_set():
                        error_reason = "stop指令中断"
                        break
                    cur = self._get_current_joints()
                    if cur:
                        err = max(abs(a - t) for a, t in zip(cur, target))
                        if err < 1.0:
                            break
                    time.sleep(0.05)
                else:
                    if not error_reason:
                        error_reason = "到位等待超时"
                if not error_reason:
                    ok = True

        except Exception as e:
            error_reason = f"异常: {e}"

        finally:
            # ── 无论成功还是中断，都必须关闭伺服模式 ──
            try:
                self.robot.ServoMoveEnd()
            except Exception as e:
                print(f"  [警告] ServoMoveEnd 异常: {e}")

            self._is_moving = False
            self._paused    = False
            if self._stop_event.is_set() and self.sm.state not in (SysState.ESTOP, SysState.ERROR):
                try:
                    time.sleep(0.15)
                    self.robot.ResetAllError()
                    print("  [停止] 机械臂已制动，可接受新指令")
                except Exception as e:
                    print(f"  [停止] ResetAllError 异常: {e}")

        actual_duration = time.time() - t_start
        if not ok and not error_reason:
            if actual_duration > ERROR_TIMEOUT + duration:
                error_reason = f"运动超时({actual_duration:.1f}s)"

        actual  = self._get_current_joints() or [0.0] * 6
        errors  = [round(a - t, 4) for a, t in zip(actual, target)]
        max_err = max(abs(e) for e in errors)

        if ok:
            status = "OK" if max_err <= ERROR_MAX_ERR else f"WARNING(误差{max_err:.2f}°)"
            print(f"  [完成] 耗时:{actual_duration:.2f}s  最大误差:{max_err:.3f}°  [{status}]")
            self._log_row("MOVE_DONE", command_tag, smooth, duration,
                          actual_duration, target, actual, "")
            if self.comm:
                self.comm.send_execution_result(command_tag, ok=True, max_err_deg=max_err)
        else:
            if "stop指令" in error_reason:
                print(f"  [停止] 运动已中断，当前位置: {[round(v,3) for v in actual]}")
                self._log_row("MOVE_STOPPED", command_tag, smooth, duration,
                              actual_duration, target, actual, error_reason)
                if self.comm:
                    self.comm.send_execution_result(command_tag, ok=False, reason=error_reason)
            else:
                print(f"  [失败] {error_reason}")
                self.sm.trigger_error(error_reason)
                self._log_row("ERROR", command_tag, smooth, duration,
                              actual_duration, target, actual, error_reason)
                if self.comm:
                    self.comm.send_execution_result(command_tag, ok=False, reason=error_reason)
        return ok

    # ──────────────────────────────────────────
    # 轨迹执行（由 TrajectoryServer 回调触发）
    # ──────────────────────────────────────────
    def execute_trajectory(self, payload: dict) -> bool:
        """
        执行轨迹模块推送过来的轨迹（ServoJ 版本）。
        直接按轨迹模块给的 t/q 逐点 ServoJ 推送，
        用相邻点的时间差控制每帧发送间隔，忠实还原轨迹模块的时序规划。
        逻辑链与 _execute_move 完全对齐。
        """
        with self._move_lock:
            if not self._check_can_move():
                return False

        points          = payload.get("points", [])
        traj_id         = payload.get("trajectory_id", "?")
        cmd_tag         = payload.get("case_name") or f"TRAJ_{traj_id[:8]}"
        target          = [math.degrees(float(v)) for v in points[-1]["q"]]   # rad→deg
        duration_planned = float(points[-1].get("t", 0.0))

        print(f"\n  [轨迹] 开始执行 {cmd_tag}  id={traj_id}  共 {len(points)} 个点")
        print(f"  [轨迹] 末点(°): {[round(v, 3) for v in target]}")

        current = self._get_current_joints()
        if current is None:
            print("  [错误] 无法读取当前关节角度")
            return False

        self._is_moving = True
        self._stop_event.clear()
        t_start      = time.time()
        ok           = False
        error_reason = ""

        try:
            # ── 开启伺服模式 ──────────────────────
            ret_start = self.robot.ServoMoveStart()
            if ret_start != 0:
                error_reason = f"ServoMoveStart 失败，错误码:{ret_start}"
                raise RuntimeError(error_reason)

            t_traj_start = time.time()   # 轨迹绝对起始时刻，用于对齐每帧发送时间

            for i, pt in enumerate(points):
                if self._stop_event.is_set():
                    error_reason = "stop指令中断"
                    break
                if self.sm.state in (SysState.ESTOP, SysState.ERROR):
                    error_reason = f"状态机中断({self.sm.state})"
                    break

                q = [math.degrees(float(v)) for v in pt["q"]]   # rad→deg
                ret = self._servoj(q)
                if ret != 0:
                    if self._stop_event.is_set():
                        error_reason = "stop指令中断"
                    else:
                        error_reason = f"ServoJ错误码:{ret} (第{i+1}点)"
                    break

                # ── 按轨迹绝对时间戳对齐节拍 ──────────
                # 用 t_traj_start + pt["t"] 计算本帧应发送的绝对时刻，
                # 扣掉 ServoJ 本身耗时，避免累积误差
                if i + 1 < len(points):
                    t_next = t_traj_start + float(points[i + 1]["t"])
                    sleep_t = t_next - time.time()
                    if sleep_t > 0:
                        time.sleep(sleep_t)

            else:
                # 所有点发完，轮询等待末点真正到位
                settle_timeout = duration_planned + 2.0
                t_settle = time.time()
                while time.time() - t_settle < settle_timeout:
                    if self._stop_event.is_set():
                        error_reason = "stop指令中断"
                        break
                    cur = self._get_current_joints()
                    if cur:
                        err = max(abs(a - t) for a, t in zip(cur, target))
                        if err < 1.0:
                            break
                    time.sleep(0.05)
                else:
                    if not error_reason:
                        error_reason = "到位等待超时"
                if not error_reason:
                    ok = True

        except Exception as e:
            error_reason = f"异常: {e}"

        finally:
            # ── 无论成功还是中断，都必须关闭伺服模式 ──
            try:
                self.robot.ServoMoveEnd()
            except Exception as e:
                print(f"  [警告] ServoMoveEnd 异常: {e}")

            self._is_moving = False
            self._paused    = False
            if self._stop_event.is_set() and self.sm.state not in (SysState.ESTOP, SysState.ERROR):
                try:
                    time.sleep(0.15)
                    self.robot.ResetAllError()
                    print("  [停止] 机械臂已制动，可接受新指令")
                except Exception as e:
                    print(f"  [停止] ResetAllError 异常: {e}")

        actual_duration = time.time() - t_start
        if not ok and not error_reason:
            if actual_duration > ERROR_TIMEOUT + duration_planned:
                error_reason = f"运动超时({actual_duration:.1f}s)"

        actual  = self._get_current_joints() or [0.0] * 6
        errors  = [round(a - t, 4) for a, t in zip(actual, target)]
        max_err = max(abs(e) for e in errors)

        if ok:
            status = "OK" if max_err <= ERROR_MAX_ERR else f"WARNING(误差{max_err:.2f}°)"
            print(f"  [完成] 轨迹 {cmd_tag}  耗时:{actual_duration:.2f}s  "
                  f"末点最大误差:{max_err:.3f}°  [{status}]")
            self._log_row("TRAJ_DONE", cmd_tag, False,
                          duration_planned, actual_duration, target, actual, "")
            if self.comm:
                self.comm.send_execution_result(cmd_tag, ok=True, max_err_deg=max_err)
        else:
            if "stop指令" in error_reason:
                print(f"  [停止] 轨迹已中断 {cmd_tag}，"
                      f"当前位置: {[round(v, 3) for v in actual]}")
                self._log_row("TRAJ_STOPPED", cmd_tag, False,
                              duration_planned, actual_duration, target, actual, error_reason)
                if self.comm:
                    self.comm.send_execution_result(cmd_tag, ok=False, reason=error_reason)
            else:
                print(f"  [失败] 轨迹 {cmd_tag}  原因: {error_reason}")
                self.sm.trigger_error(error_reason)
                self._log_row("TRAJ_ERROR", cmd_tag, False,
                              duration_planned, actual_duration, target, actual, error_reason)
                if self.comm:
                    self.comm.send_execution_result(cmd_tag, ok=False, reason=error_reason)

        return ok

    # ──────────────────────────────────────────
    # 运动到预设点位
    # ──────────────────────────────────────────
    def move_to(self, preset_key: str, smooth: bool = True,
                command_tag: str = "AUTO") -> bool:
        with self._move_lock:
            if not self._check_can_move():
                return False
        preset = PRESETS[preset_key]
        print(f"\n  [运动] → {preset['name']}  [模式:{self.sm.state}]")
        return self._execute_move(preset["joints"], smooth, command_tag)

    # ──────────────────────────────────────────
    # 小幅微调
    # ──────────────────────────────────────────
    def micro_adjust(self, direction: str,
                     step: float = MICRO_STEP_DEFAULT) -> bool:
        with self._move_lock:
            if not self._check_can_move():
                return False

        if direction not in MICRO_DIRS:
            print(f"  [错误] 未知方向: {direction}")
            print(f"  可用: {list(MICRO_DIRS.keys())}")
            return False

        step      = min(abs(step), MICRO_STEP_MAX)
        joint_idx, sign, desc = MICRO_DIRS[direction]

        current = self._get_current_joints()
        if current is None:
            print("  [错误] 无法读取当前关节角度")
            return False

        target = current[:]
        target[joint_idx] += sign * step

        target_clamped, hit_limit = clamp_joints(target)
        if hit_limit:
            actual_step = abs(target_clamped[joint_idx] - current[joint_idx])
            print(f"  [限位] 步长压缩至{actual_step:.1f}°")
            target = target_clamped
            if actual_step < 0.1:
                print("  [限位] 已在边界，无法继续")
                return False

        print(f"\n  [微调] {desc}  步长:{step:.1f}°")
        print(f"  j{joint_idx+1}: {current[joint_idx]:.3f}° → {target[joint_idx]:.3f}°")

        return self._execute_move(
            target, smooth=True,
            command_tag=f"MICRO_{direction.upper()}_{step}deg",
            min_dur=0.3,
        )

    # ──────────────────────────────────────────
    # CSV 日志
    # ──────────────────────────────────────────
    def _log_row(self, event, command, smooth, dur_plan, dur_actual,
                 target, actual, error_reason):
        errors  = [round(a - t, 4) for a, t in zip(actual, target)]
        max_err = round(max(abs(e) for e in errors), 4)
        self._csv.writerow([
            round(time.time(), 3), event, command, self.sm.state,
            "smooth" if smooth else "raw",
            round(dur_plan, 3), round(dur_actual, 3),
            *[round(v, 3) for v in target],
            *[round(v, 3) for v in actual],
            *errors, max_err, error_reason,
        ])
        self._csv_file.flush()

    # ──────────────────────────────────────────
    # 10次重复测试
    # ──────────────────────────────────────────
    def repeat_test(self, preset_key: str, reps: int = 10,
                    smooth: bool = True):
        if preset_key not in PRESETS or preset_key == "0":
            print("  [错误] 请选择点位 1 或 2（0为home基准）")
            return

        mode_str = "平滑" if smooth else "原始"
        print(f"\n{'='*55}")
        print(f"  10次重复测试  目标:{PRESETS[preset_key]['name']}  [{mode_str}]")
        print(f"  路径: 位置A → {preset_key} → 位置A  ×{reps}")
        print(f"{'='*55}")

        durations, max_errors, success_count = [], [], 0

        for i in range(1, reps + 1):
            print(f"\n── 第{i}/{reps}次 ────────────────────────")
            ok1 = self.move_to("0", smooth=smooth,
                               command_tag=f"REPEAT_{i}_HOME")
            if not ok1:
                if self.sm.state == SysState.ERROR:
                    self.recover_error()
                continue
            time.sleep(0.5)

            t0  = time.time()
            ok2 = self.move_to(preset_key, smooth=smooth,
                               command_tag=f"REPEAT_{i}_TARGET")
            elapsed = time.time() - t0

            if not ok2:
                if self.sm.state == SysState.ERROR:
                    self.recover_error()
                continue

            actual  = self._get_current_joints() or [0.0]*6
            tgt     = PRESETS[preset_key]["joints"]
            max_err = max(abs(a - t) for a, t in zip(actual, tgt))
            durations.append(elapsed)
            max_errors.append(max_err)
            success_count += 1
            time.sleep(0.5)

        print(f"\n{'='*55}")
        print(f"  完成：{success_count}/{reps} 次成功")
        if durations:
            print(f"  平均耗时    : {sum(durations)/len(durations):.3f} s")
            print(f"  平均最大误差: {sum(max_errors)/len(max_errors):.3f} °")
            print(f"  最大单次误差: {max(max_errors):.3f} °")
            verdict = "✓ 可复现" if max(max_errors) < ERROR_MAX_ERR else "⚠ 误差偏大"
            print(f"  复现评估    : {verdict}")
        print(f"{'='*55}\n")

    # ──────────────────────────────────────────
    # 对比测试：原始 vs 平滑
    # ──────────────────────────────────────────
    def compare_test(self, preset_key: str, reps: int = 3):
        if preset_key not in PRESETS:
            print(f"  [错误] 未知点位: {preset_key}")
            return

        print(f"\n{'='*55}")
        print(f"  对比测试  目标:{PRESETS[preset_key]['name']}  各{reps}次")
        print(f"{'='*55}")

        for mode_label, smooth_flag, out_file in [
            ("原始(raw)",    False, "motion_log_raw.csv"),
            ("平滑(smooth)", True,  "motion_log_smooth.csv"),
        ]:
            print(f"\n── {mode_label} ────────────────────────")
            old_file, old_csv = self._csv_file, self._csv
            self._csv_file = open(out_file, "w", newline="", encoding="utf-8")
            self._csv = csv.writer(self._csv_file)
            self._csv.writerow([
                "timestamp","event","command","sys_state","smooth_mode",
                "duration_planned_s","duration_actual_s",
                "j1_target","j2_target","j3_target","j4_target","j5_target","j6_target",
                "j1_actual","j2_actual","j3_actual","j4_actual","j5_actual","j6_actual",
                "j1_error","j2_error","j3_error","j4_error","j5_error","j6_error",
                "max_error_deg","error_reason",
            ])
            for i in range(1, reps + 1):
                print(f"  第{i}/{reps}次...")
                self.move_to("0",        smooth=smooth_flag,
                             command_tag=f"CMP_{i}_HOME")
                time.sleep(0.5)
                self.move_to(preset_key, smooth=smooth_flag,
                             command_tag=f"CMP_{i}_TARGET")
                time.sleep(0.5)

            self._csv_file.close()
            self._csv_file, self._csv = old_file, old_csv
            print(f"  → 已保存：{out_file}")

        print(f"\n  [对比测试完成]\n")

    def cleanup(self):
        self._csv_file.close()
        print(f"\n  [日志] 已保存到 {LOG_FILE}")


# ══════════════════════════════════════════════
# 命令行
# ══════════════════════════════════════════════

def print_help():
    print("\n" + "="*60)
    print("  FR5 运动控制节点 V13  (ServoJ 伺服版本)")
    print("="*60)
    print("  【预设点位】")
    for k, v in PRESETS.items():
        print(f"    {k}  → {v['name']}")
        print(f"       {[round(x,2) for x in v['joints']]}")
    print()
    print("  【回中】")
    print("    setcenter  → 记录当前位置为中心点（写入 center_config.json）")
    print("    recenter   → 回到已记录的中心点（平滑运动）")
    print("    center     → 查看当前记录的中心点")
    print()
    print("  【小幅微调】  m <方向> [步长°]  默认步长3°，最大10°")
    for d, (idx, sign, desc) in MICRO_DIRS.items():
        print(f"    m {d:<8} → {desc}")
    print("    示例: m up      m left 5      m tilt+ 2")
    print()
    print("  【测试】")
    print("    r <点位>   → 10次重复测试     (如: r 1)")
    print("    c <点位>   → 原始vs平滑对比   (如: c 1)")
    print()
    print("  【状态机控制】")
    print("    hold       → 锁定（→ HOLD），拒绝新运动指令")
    print("    resume     → 解锁（HOLD → 恢复之前状态）")
    print("    p          → 暂停 / 续跑当前运动（PauseMotion/ResumeMotion）")
    print("    stop       → 停止并丢弃路径（不改变功能模式）")
    print("    recover    → 从 ERROR/ERROR_END 恢复 → MANUAL")
    print("    estop      → 急停（→ ESTOP）")
    print("    reset      → 急停复位（ESTOP → INIT → MANUAL）")
    print("    mode <模式>→ 手动切换模式  manual / assist / auto")
    print()
    print("  【查询】")
    print("    status     → 查看系统状态机状态")
    print("    joints     → 查看当前关节角度")
    print("    center     → 查看当前记录的中心点")
    print("    help / q")
    print()
    print(f"  当前ServoJ配置: cmdT={SERVO_CMD_T}s  filterT={SERVO_FILTER}  gain={SERVO_GAIN}")
    print("  ★ blendT 已废弃，平滑由 S曲线插值+ServoJ高频推送实现")
    print("="*60)


def input_loop(ctrl: FR5Controller):
    time.sleep(0.5)
    print_help()

    while True:
        try:
            cmd = input("\n  指令> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        parts = cmd.split()
        if not parts:
            continue
        p0 = parts[0].lower()

        if p0 == "q":
            break

        elif p0 == "help":
            print_help()

        # ── 状态机控制 ──────────────────────
        elif p0 == "hold":
            ctrl.hold()

        elif p0 == "resume":
            ctrl.resume_from_hold()

        elif p0 == "p":
            ctrl.toggle_pause()

        elif p0 == "stop":
            ctrl.stop_motion()

        elif p0 == "recover":
            ctrl.recover_error()

        elif p0 == "estop":
            reason = " ".join(parts[1:]) if len(parts) > 1 else "操作员急停"
            ctrl.trigger_estop(reason)

        elif p0 == "reset":
            ctrl.reset_estop()

        elif p0 == "mode":
            if len(parts) < 2:
                print("  用法: mode <manual|assist|auto>")
            else:
                ctrl.switch_mode(parts[1])

        # ── 查询 ────────────────────────────
        elif p0 == "status":
            print(f"  {ctrl.sm.status()}")
            print(f"  运动中: {'是' if ctrl._is_moving else '否'}")

        elif p0 == "joints":
            j = ctrl._get_current_joints()
            if j:
                print(f"  当前关节(°): {[round(v,3) for v in j]}")
            else:
                print("  读取失败")

        # ── 回中 ────────────────────────────
        elif p0 == "setcenter":
            ctrl.set_center()

        elif p0 == "recenter":
            threading.Thread(
                target=ctrl.recenter, daemon=True,
            ).start()

        elif p0 == "center":
            ctrl.show_center()

        # ── 微调 ────────────────────────────
        elif p0 == "m":
            if len(parts) < 2:
                print(f"  用法: m <方向> [步长°]  方向: {list(MICRO_DIRS.keys())}")
            else:
                direction = parts[1].lower()
                step = MICRO_STEP_DEFAULT
                if len(parts) >= 3:
                    try:
                        step = float(parts[2])
                    except ValueError:
                        print("  [错误] 步长必须是数字")
                        continue
                threading.Thread(
                    target=ctrl.micro_adjust,
                    args=(direction, step), daemon=True,
                ).start()

        # ── 重复测试 ────────────────────────
        elif p0 == "r":
            if len(parts) < 2 or parts[1] not in PRESETS:
                print(f"  用法: r <点位>  可选: {list(PRESETS.keys())}")
            else:
                threading.Thread(
                    target=ctrl.repeat_test,
                    args=(parts[1],), daemon=True,
                ).start()

        # ── 对比测试 ────────────────────────
        elif p0 == "c":
            if len(parts) < 2 or parts[1] not in PRESETS:
                print(f"  用法: c <点位>  可选: {list(PRESETS.keys())}")
            else:
                threading.Thread(
                    target=ctrl.compare_test,
                    args=(parts[1],), daemon=True,
                ).start()

        # ── 预设点位运动 ────────────────────
        elif p0 in PRESETS:
            threading.Thread(
                target=ctrl.move_to,
                args=(p0,), daemon=True,
            ).start()

        else:
            print(f"  未知指令: {cmd}  输入 help 查看帮助")


def main():
    print("\n" + "="*60)
    print("  FR5 运动控制节点 V13 —— ServoJ 伺服版本")
    print(f"  控制箱: {ROBOT_IP}  上位机: {HOST_IP}:{HOST_PORT}  轨迹端口: {TRAJ_PORT}")
    print("="*60)

    # ── 启动上位机通信 ──────────────────────
    comm = HostComm(host_ip=HOST_IP, host_port=HOST_PORT)
    comm.start()

    # ── 启动机械臂控制器（注入 comm）──────────
    ctrl = FR5Controller(ROBOT_IP, comm=comm)

    # ── 启动轨迹接收服务 ────────────────────
    traj_srv = TrajectoryServer(
        listen_port   = TRAJ_PORT,
        on_trajectory = ctrl.execute_trajectory,
    )
    traj_srv.start()

    try:
        input_loop(ctrl)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.cleanup()
        traj_srv.stop()
        comm.stop()
        print("  [退出]")


if __name__ == "__main__":
    main()
