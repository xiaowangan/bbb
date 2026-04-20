#!/usr/bin/env python3
"""
扶镜臂系统状态机模块
system_state_machine.py

状态定义：
  INIT        → 初始状态，刚上电
  START_READY → 系统设备和连接正常，准备开始
  MANUAL      → 手动状态
  ASSIST      → 辅助状态
  AUTO_READY  → 目标位置良好，位于视野中心，准备自动
  AUTO        → 自动状态（当前版本运行模式）
  HOLD        → 保持状态（锁定当前位置，拒绝新运动指令）
  ERROR       → 错误状态，当数据异常或设备异常时进入
  ERROR_END   → 从错误状态脱离，数据和设备正常时进入，可重新进入手动模式
  ESTOP       → 急停状态，急停按钮、安全链路触发、严重故障

合法状态转移表：
  INIT        → START_READY
  START_READY → MANUAL, ESTOP, ERROR
  MANUAL      → ASSIST, AUTO_READY, HOLD, ERROR, ESTOP
  ASSIST      → MANUAL, AUTO_READY, HOLD, ERROR, ESTOP
  AUTO_READY  → AUTO, MANUAL, ASSIST, HOLD, ERROR, ESTOP
  AUTO        → AUTO_READY, MANUAL, HOLD, ERROR, ESTOP
  HOLD        → MANUAL, ASSIST, AUTO_READY, AUTO, ERROR, ESTOP
  ERROR       → ERROR_END, ESTOP
  ERROR_END   → MANUAL, ESTOP
  ESTOP       → INIT  （复位后重新初始化）
"""

import threading
import time
from typing import Optional, Callable


# ──────────────────────────────────────────────
# 状态枚举（字符串常量）
# ──────────────────────────────────────────────
class SysState:
    INIT        = "INIT"
    START_READY = "START_READY"
    MANUAL      = "MANUAL"
    ASSIST      = "ASSIST"
    AUTO_READY  = "AUTO_READY"
    AUTO        = "AUTO"
    HOLD        = "HOLD"
    ERROR       = "ERROR"
    ERROR_END   = "ERROR_END"
    ESTOP       = "ESTOP"
#字符串的赋值

# ──────────────────────────────────────────────
# 合法状态转移表
# ──────────────────────────────────────────────
VALID_TRANSITIONS: dict[str, list[str]] = {
    SysState.INIT:        [SysState.START_READY],
    SysState.START_READY: [SysState.MANUAL,     SysState.ERROR,     SysState.ESTOP],
    SysState.MANUAL:      [SysState.ASSIST,     SysState.AUTO_READY, SysState.HOLD,
                           SysState.ERROR,      SysState.ESTOP],
    SysState.ASSIST:      [SysState.MANUAL,     SysState.AUTO_READY, SysState.HOLD,
                           SysState.ERROR,      SysState.ESTOP],
    SysState.AUTO_READY:  [SysState.AUTO,       SysState.MANUAL,    SysState.ASSIST,
                           SysState.HOLD,       SysState.ERROR,     SysState.ESTOP],
    SysState.AUTO:        [SysState.AUTO_READY, SysState.MANUAL,    SysState.HOLD,
                           SysState.ERROR,      SysState.ESTOP],
    SysState.HOLD:        [SysState.MANUAL,     SysState.ASSIST,    SysState.AUTO_READY,
                           SysState.AUTO,       SysState.ERROR,     SysState.ESTOP],
    SysState.ERROR:       [SysState.ERROR_END,  SysState.ESTOP],
    SysState.ERROR_END:   [SysState.MANUAL,     SysState.ESTOP],
    SysState.ESTOP:       [SysState.INIT],
}#key-value类型，里面都是字符串

# 任何状态都可以直接进入 ESTOP / ERROR（安全优先），万一缺了没写，这里是直接补充上去了
for _s in list(VALID_TRANSITIONS.keys()):
    for _e in (SysState.ESTOP, SysState.ERROR):
        if _e not in VALID_TRANSITIONS[_s]:
            VALID_TRANSITIONS[_s].append(_e)


# ──────────────────────────────────────────────
# 工作模式分类（便于业务层判断）
# ──────────────────────────────────────────────
FUNCTIONAL_STATES = {SysState.MANUAL, SysState.ASSIST, SysState.AUTO}
READY_STATES      = {SysState.START_READY, SysState.AUTO_READY}
ERROR_STATES      = {SysState.ERROR, SysState.ERROR_END, SysState.ESTOP}


# ══════════════════════════════════════════════
# 状态机主类
# ══════════════════════════════════════════════

class SystemStateMachine:
    """
    扶镜臂系统状态机

    用法示例：
        from system_state_machine import SystemStateMachine, SysState

        sm = SystemStateMachine()
        sm.register_callback(my_on_state_change)   # 可选
        sm.startup()           # INIT → START_READY
        sm.to_manual()         # START_READY → MANUAL
        sm.to_auto()           # MANUAL → AUTO_READY → AUTO（两步）
        sm.hold()              # AUTO → HOLD
        sm.resume_from_hold()  # HOLD → AUTO（回到锁定前的状态）
        sm.trigger_error("传感器断线")  # → ERROR
        sm.recover_error()     # ERROR → ERROR_END → MANUAL
        sm.estop()             # → ESTOP
        sm.reset_estop()       # ESTOP → INIT
    """

    def __init__(self, verbose: bool = True):
        self._state        : str            = SysState.INIT
        self._prev_state   : Optional[str]  = None   # HOLD恢复时使用
        self._error_reason : str            = ""
        self._lock         : threading.Lock = threading.Lock()
        self._verbose      : bool           = verbose
        self._callbacks    : list[Callable] = []     # (old_state, new_state, reason) → None
        self._state_entry_time : float      = time.time()

        if self._verbose:
            print(f"  [状态机] 初始化完成 → {self._state}")

    # ──────────────────────────────────────────
    # 属性访问
    # ──────────────────────────────────────────  
    """ @property 是一个装饰器，它最大的作用是：让方法（函数）看起来像属性（变量）一样被调用。
    不用 @property————if sm.is_operable():(看起来像在执行一个复杂的动作)
    用了 @property————if sm.is_operable:(看起来像在读一个仪表盘的数值，更直观)"""
    @property
    def state(self) -> str:
        return self._state

    @property
    def error_reason(self) -> str:
        return self._error_reason

    @property
    def is_operable(self) -> bool:
        """当前是否可接受运动指令（非锁定/非错误/非急停）"""
        return self._state in FUNCTIONAL_STATES#只要是里面的一个就返回true

    @property
    def is_held(self) -> bool:
        return self._state == SysState.HOLD

    @property
    def is_error(self) -> bool:
        return self._state in ERROR_STATES

    @property
    def is_auto(self) -> bool:
        return self._state == SysState.AUTO

    @property
    def is_manual(self) -> bool:
        return self._state == SysState.MANUAL

    @property
    def state_duration(self) -> float:
        """已在当前状态停留的秒数"""
        return time.time() - self._state_entry_time

    # ──────────────────────────────────────────
    # 回调注册
    # ──────────────────────────────────────────
    def register_callback(self, fn: Callable):
        """
        注册状态变化回调。
        回调签名：fn(old_state: str, new_state: str, reason: str)
        """
        self._callbacks.append(fn)

    def _fire_callbacks(self, old: str, new: str, reason: str):
        for fn in self._callbacks:
            try:
                fn(old, new, reason)
            except Exception as e:
                print(f"  [状态机] 回调异常: {e}")

    # ──────────────────────────────────────────
    # 核心状态转移
    # ──────────────────────────────────────────
    def _transition(self, new_state: str, reason: str = "") -> bool:
        """
        执行状态转移。线程安全。
        返回 True 表示转移成功，False 表示非法转移被拦截。
        """
        with self._lock:
            old_state = self._state

            if new_state == old_state:
                if self._verbose:
                    print(f"  [状态机] 已在 {old_state}，无需切换")
                return True

            allowed = VALID_TRANSITIONS.get(old_state, [])#default ([])：如果字典里没有这个 old_state，别报错，直接给我一个空列表 []。
            if new_state not in allowed:
                print(f"  [状态机] ✗ 非法转移: {old_state} → {new_state}"
                      f"  (允许: {allowed})")
                return False

            self._prev_state       = old_state
            self._state            = new_state
            self._state_entry_time = time.time()

            tag = f" ({reason})" if reason else ""
            if self._verbose:
                print(f"  [状态机] {old_state} → {new_state}{tag}")

        self._fire_callbacks(old_state, new_state, reason)
        return True

    # ──────────────────────────────────────────
    # 公开转移接口
    # ──────────────────────────────────────────
    def startup(self) -> bool:
        """INIT → START_READY：设备自检通过，连接正常"""
        return self._transition(SysState.START_READY, "设备自检通过")#这个函数的返回值是true

    def to_manual(self) -> bool:
        """→ MANUAL：切换到手动模式"""
        return self._transition(SysState.MANUAL, "切换手动")

    def to_assist(self) -> bool:
        """→ ASSIST：切换到辅助模式"""
        return self._transition(SysState.ASSIST, "切换辅助")

    def to_auto_ready(self) -> bool:
        """→ AUTO_READY：目标已就绪，位于视野中心"""
        return self._transition(SysState.AUTO_READY, "目标就绪")

    def to_auto(self) -> bool:
        """→ AUTO：进入自动跟踪/运动模式。
        自动处理中间状态：
          AUTO_READY → AUTO
          MANUAL/ASSIST → AUTO_READY → AUTO
          START_READY → MANUAL → AUTO_READY → AUTO
        """
        if self._state == SysState.AUTO:
            return True
        if self._state == SysState.AUTO_READY:
            return self._transition(SysState.AUTO, "启动自动模式")
        # START_READY 需先到 MANUAL
        if self._state == SysState.START_READY:
            if not self._transition(SysState.MANUAL, "准备自动（经MANUAL）"):
                return False
        # MANUAL / ASSIST → AUTO_READY → AUTO
        if self._transition(SysState.AUTO_READY, "目标就绪"):
            return self._transition(SysState.AUTO, "启动自动模式")
        return False

    def hold(self) -> bool:
        """
        → HOLD：锁定当前位置，暂停所有运动指令。
        记住进入 HOLD 前的状态，以便 resume_from_hold() 恢复。
        """
        return self._transition(SysState.HOLD, "锁定")

    def resume_from_hold(self) -> bool:
        """
        HOLD → 恢复到锁定前的状态（MANUAL / ASSIST / AUTO 等）。
        若无历史状态或历史状态不合法，默认恢复到 MANUAL。
        """
        if self._state != SysState.HOLD:
            print(f"  [状态机] 当前不在 HOLD 状态（当前: {self._state}）")
            return False

        target = self._prev_state
        allowed = VALID_TRANSITIONS.get(SysState.HOLD, [])

        # 确保历史状态在合法转移表内
        if target not in allowed or target in ERROR_STATES:
            target = SysState.MANUAL

        return self._transition(target, f"解锁，恢复到 {target}")

    def trigger_error(self, reason: str = "未知错误") -> bool:
        """→ ERROR：任何状态均可调用"""
        self._error_reason = reason
        return self._transition(SysState.ERROR, reason)

    def recover_error(self) -> bool:
        """
        ERROR → ERROR_END → MANUAL：两步恢复流程。
        先确认设备正常（ERROR_END），再切回手动等待操作员确认。
        """
        if self._state == SysState.ERROR:
            if not self._transition(SysState.ERROR_END, "设备/数据恢复正常"):
                return False
        if self._state == SysState.ERROR_END:
            self._error_reason = ""
            return self._transition(SysState.MANUAL, "错误处理完毕，切换手动")
        print(f"  [状态机] recover_error 只能在 ERROR/ERROR_END 状态调用")
        return False

    def estop(self, reason: str = "急停触发") -> bool:
        """→ ESTOP：任何状态均可调用，最高优先级"""
        self._error_reason = reason
        return self._transition(SysState.ESTOP, reason)

    def reset_estop(self) -> bool:
        """ESTOP → INIT：急停复位，重新初始化"""
        if self._state != SysState.ESTOP:
            print(f"  [状态机] 当前不在 ESTOP 状态")
            return False
        self._error_reason = ""
        ok = self._transition(SysState.INIT, "急停复位")
        return ok

    # ──────────────────────────────────────────
    # 通用强制转移（高级用途，不走合法性检查）
    # ──────────────────────────────────────────
    def force_transition(self, new_state: str, reason: str = "强制转移") -> bool:
        """
        绕过合法性检查强制转移。
        仅在系统集成调试时使用，正式运行请勿调用。
        """
        print(f"  [状态机] ⚠ 强制转移: {self._state} → {new_state}")
        #会先锁线程
        with self._lock:
            old               = self._state
            self._prev_state  = old
            self._state       = new_state
            self._state_entry_time = time.time()
        self._fire_callbacks(old, new_state, reason)
        return True

    # ──────────────────────────────────────────
    # 状态查询
    # ──────────────────────────────────────────
    def status(self) -> str:
        err = f"  错误原因: {self._error_reason}" if self._error_reason else ""
        dur = f"  已持续: {self.state_duration:.1f}s"
        return f"当前状态: {self._state}{dur}{err}"

    def can_move(self) -> bool:
        """对外暴露：是否允许下发运动指令"""
        if not self.is_operable:
            if self._state == SysState.HOLD:
                print("  [状态机] 当前 HOLD（锁定），运动指令被拒绝")
            elif self._state in ERROR_STATES:
                print(f"  [状态机] 当前 {self._state}，运动指令被拒绝  原因: {self._error_reason}")
            else:
                print(f"  [状态机] 当前 {self._state}，不在可运动状态")
            return False
        return True

    def __repr__(self) -> str:
        return f"<SystemStateMachine state={self._state}>"
