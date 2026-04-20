from typing import List


def get_default_start_joints() -> List[float]:
    """
    全工程统一默认起始关节角

    说明：
    - 当前先保留原默认值
    - 如果你后面在 RViz 中找到更合适的新工具起始姿态，
      只需要改这里，所有 ptp / scan / grid case 都会同步更新
    """
    return [0.000, -1.570, 1.570, 0.000, 0.000, 0.000]