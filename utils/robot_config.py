"""
robot_config.py — 共享机器人配置
==============================
所有需要 ROBOT_IDS 的文件从此处导入，修改一处即可全局生效。
"""

ROBOT_IDS = [3, 16]      # 机器人 ID 列表：robot03 pursuer，robot16 target
CMD_DOMAIN_ID = 0               # 算法节点 ROS_DOMAIN_ID
ROBOT_DOMAIN_BASE = 10          # 机器人 domain = robot_id + ROBOT_DOMAIN_BASE


def robot_name(robot_id: int) -> str:
    """机器人 ID → 名称: 1 → robot01"""
    return f"robot{robot_id:02d}"


def robot_domain(robot_id: int) -> int:
    """机器人 ID → ROS_DOMAIN_ID"""
    return robot_id + ROBOT_DOMAIN_BASE
