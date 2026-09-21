#!/usr/bin/env python3
"""
circle_demo.py — 多机器人圆周轨迹跟踪 demo
============================================
控制模式:
  1: 航向保持 — ω=0, 仅发 vx/vy (机体=世界, 机器人朝向不变)
  2: 阿克曼   — vy=0, 发 vx + ω   (机器人始终切向圆)

运行前需先启动:
  - odom_proxy.py  (提供 /{robot_name}/odometry)
  - proxy_vel.py   (转发 /{robot_name}/vel_cmd)
运行后按 ENTER 开始.
"""

import math
import os
import sys
import time

# 确保 utils/ 在 Python 路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
from multiprocessing import Process, Event
from robot_config import ROBOT_IDS, CMD_DOMAIN_ID, robot_name

# ========================== 配置 ==========================
CONTROL_MODE = 1          # 1=航向保持, 2=阿克曼
RADIUS = 0.5              # 圆半径 (m), 正=逆时针(圆心在左侧), 负=顺时针
LINEAR_SPEED = 0.2        # 线速度 (m/s)
PUB_RATE = 50             # 控制频率 (Hz)

# PID 参数 — 位置环
PID_POS_KP = 2.0
PID_POS_KI = 0.0
PID_POS_KD = 0.1
PID_LIMIT_XY = 0.3        # 位置环输出限幅 ±m/s

# PID 参数 — 航向环
PID_YAW_KP = 0.0
PID_YAW_KI = 0.0
PID_YAW_KD = 0.0
PID_LIMIT_YAW = 0.0       # 航向环输出限幅 ±rad/s

# ======================== 工具函数 ========================
def wrap_to_pi(a: float) -> float:
    """规整到 [-pi, pi]"""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def yaw_from_quaternion(qx, qy, qz, qw) -> float:
    """四元数 → yaw"""
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


# ========================== PID ===========================
class PID:
    """带积分限幅的 PID 控制器"""
    def __init__(self, kp, ki, kd, limit, i_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = limit
        self.i_limit = i_limit if i_limit is not None else limit
        self.integral = 0.0
        self.prev_err = 0.0

    def update(self, err: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        self.integral += err * dt
        self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        deriv = (err - self.prev_err) / dt
        self.prev_err = err
        out = self.kp * err + self.ki * self.integral + self.kd * deriv
        return max(-self.limit, min(self.limit, out))


# ====================== 单机器人控制进程 ======================
def robot_controller_process(robot_id: int, start_event: Event):
    rname = robot_name(robot_id)
    odom_topic = f"/{rname}/odom"
    vel_topic = f"/{rname}/vel_cmd"

    os.environ["ROS_DOMAIN_ID"] = str(CMD_DOMAIN_ID)
    rclpy.init()
    node = Node(f"{rname}_circle_ctrl")
    pub = node.create_publisher(TwistStamped, vel_topic, 10)

    # ---- 里程计缓存 ----
    odom = {"x": 0.0, "y": 0.0, "yaw": 0.0, "stale": True}

    def odom_cb(msg: Odometry):
        odom["x"] = msg.pose.pose.position.x
        odom["y"] = msg.pose.pose.position.y
        odom["yaw"] = yaw_from_quaternion(
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        odom["stale"] = False

    node.create_subscription(Odometry, odom_topic, odom_cb, 10)

    # ---- 等待 ENTER ----
    node.get_logger().info(f"[{rname}] Waiting for ENTER...")
    start_event.wait()

    # ---- 初始化控制器 ----
    pid_x   = PID(PID_POS_KP, PID_POS_KI, PID_POS_KD, PID_LIMIT_XY)
    pid_y   = PID(PID_POS_KP, PID_POS_KI, PID_POS_KD, PID_LIMIT_XY)
    pid_yaw = PID(PID_YAW_KP, PID_YAW_KI, PID_YAW_KD, PID_LIMIT_YAW)

    t0 = time.time()
    last_t = t0
    node.get_logger().info(f"[{rname}] Started!  mode={CONTROL_MODE}  R={RADIUS:.2f}  v={LINEAR_SPEED:.2f}")

    # ---- 控制回调 ----
    def control_cb():
        nonlocal last_t
        if odom["stale"]:
            return

        t_now = time.time()
        dt = t_now - last_t
        if dt <= 0.0:
            return
        last_t = t_now
        elapsed = t_now - t0
        v, R = LINEAR_SPEED, RADIUS

        # 期望轨迹 (圆心在 (0, R), t=0 时在原点切向 +x)
        theta = v * elapsed / R
        x_d = R * math.sin(theta)
        y_d = R * (1.0 - math.cos(theta))
        vx_d = v * math.cos(theta)   # 世界系期望速度
        vy_d = v * math.sin(theta)
        psi_d = theta                 # 切线航向

        x, y, psi = odom["x"], odom["y"], odom["yaw"]

        cmd = TwistStamped()
        cmd.header.stamp = node.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"

        if CONTROL_MODE == 1:
            # --- 航向保持: ω=0, body≈world ---
            vx_pid = pid_x.update(x_d - x, dt)
            vy_pid = pid_y.update(y_d - y, dt)

            cmd.twist.linear.x = vx_d + vx_pid
            cmd.twist.linear.y = vy_d + vy_pid
            cmd.twist.angular.z = 0.0

        else:  # CONTROL_MODE == 2
            # --- 阿克曼: vy=0, vx + ω ---
            # 位置误差 → 机体坐标系
            ex_w = x_d - x
            ey_w = y_d - y
            e_body_x =  ex_w * math.cos(psi) + ey_w * math.sin(psi)   # 纵向
            e_body_y = -ex_w * math.sin(psi) + ey_w * math.cos(psi)   # 横向

            vx_pid = pid_x.update(e_body_x, dt)
            w_cross_pid = pid_y.update(e_body_y, dt)                  # 横向 → ω 修正
            w_yaw_pid   = pid_yaw.update(wrap_to_pi(psi_d - psi), dt) # 航向修正

            cmd.twist.linear.x = v + vx_pid
            cmd.twist.linear.y = 0.0
            cmd.twist.angular.z = v / R + w_cross_pid + w_yaw_pid

        pub.publish(cmd)

    node.create_timer(1.0 / PUB_RATE, control_cb)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ======================== 主入口 =========================
def main():
    print("=" * 55)
    print("  Circle Demo — Multi-Robot Trajectory Tracking")
    print("=" * 55)
    print(f"  ROBOT_IDS:    {ROBOT_IDS}")
    print(f"  CONTROL_MODE: {CONTROL_MODE} ({'航向保持' if CONTROL_MODE == 1 else '阿克曼'})")
    print(f"  RADIUS:       {RADIUS:.2f} m")
    print(f"  LINEAR_SPEED: {LINEAR_SPEED:.2f} m/s")
    print(f"  PUB_RATE:     {PUB_RATE} Hz")
    print("=" * 55)
    print("  确保 odom_proxy.py 和 proxy_vel.py 已启动")
    print("=" * 55)

    start_event = Event()
    procs = []

    for rid in ROBOT_IDS:
        p = Process(target=robot_controller_process, args=(rid, start_event),
                    name=f"ctrl-robot{rid:02d}")
        p.start()
        procs.append(p)
        time.sleep(0.5)   # 错峰启动

    input("\n>>> 按 ENTER 开始圆周运动...\n")
    start_event.set()

    try:
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        print("\n[circle_demo] Shutting down...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=3)


if __name__ == "__main__":
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
