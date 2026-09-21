#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mouse_odom_proxy.py
====================
多机器人里程计跨域转发节点。

架构:
  每个机器人 → 一个子进程 (在其自己的 ROS_DOMAIN_ID 上订阅 mouse_odom/odom)
       │
       │  multiprocessing.Queue  (robot_name, odom_msg)
       ▼
  统一发布进程 (ROS_DOMAIN_ID=0) — 直接透传到 /{robot_name}/odometry

约定:
  - 机器人 ID → DOMAIN_ID:  DOMAIN_ID = robot_id + 10
  - 机器人 ID → 名称:        robot{id:02d}  (robot00, robot01, ...)
"""

import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from multiprocessing import Process, Queue

from robot_config import ROBOT_IDS, robot_name, robot_domain


# ╔══════════════════════════════════════════════════════════════╗
# ║              订阅子进程 (每个机器人一个)                       ║
# ╚══════════════════════════════════════════════════════════════╝
def subscriber_process(robot_id: int, queue: Queue):
    """
    在机器人自己的 ROS_DOMAIN_ID (如 robot01 → domain 11) 上
    订阅 /{robot_name}/mouse_odom/odom，
    将 (robot_name, odom_msg) 放入跨进程 Queue。
    """
    domain = robot_domain(robot_id)
    rname = robot_name(robot_id)
    topic_in = f"/{rname}/mouse_odom/odom"

    os.environ["ROS_DOMAIN_ID"] = str(domain)
    rclpy.init()
    node = Node(f"{rname}_proxy_sub")

    node.get_logger().info(f"[SUB] domain={domain}  ←  {topic_in}")

    def callback(msg: Odometry):
        queue.put((rname, msg))

    node.create_subscription(Odometry, topic_in, callback, 10)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


# ╔══════════════════════════════════════════════════════════════╗
# ║           发布进程 (domain 0, 统一处理所有机器人)              ║
# ╚══════════════════════════════════════════════════════════════╝
def publisher_process(queue: Queue, robot_ids: list):
    """
    运行在 ROS_DOMAIN_ID=0。

    从 Queue 收取所有机器人的原始数据，直接透传到 /{robot_name}/odometry。
    """
    os.environ["ROS_DOMAIN_ID"] = "0"
    rclpy.init()
    node = Node("mouse_odom_proxy_pub")

    # --- 为每个机器人创建 publisher ---
    publishers = {}
    for rid in robot_ids:
        name = robot_name(rid)
        topic_out = f"/{name}/odom"
        publishers[name] = node.create_publisher(Odometry, topic_out, 10)
        node.get_logger().info(f"[PUB] domain=0  →  {topic_out}")

    # 每个机器人的最新数据缓存
    latest_per_robot = {}       # robot_name → Odometry

    def timer_callback():
        # ---- 1. 一次性排空 Queue，只保留每个机器人最新帧 ----
        while not queue.empty():
            rname, odom_msg = queue.get()
            latest_per_robot[rname] = odom_msg

        # ---- 2. 逐机器人发布 ----
        for rname, odom_msg in latest_per_robot.items():
            pub = publishers.get(rname)
            if pub is None:
                continue

            # 更新 header 时间戳为当前时间（可选，保持原始时间戳也可以）
            # odom_msg.header.stamp = node.get_clock().now().to_msg()
            
            pub.publish(odom_msg)

    # 100 Hz
    node.create_timer(0.01, timer_callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


# ╔══════════════════════════════════════════════════════════════╗
# ║                       主入口                                 ║
# ╚══════════════════════════════════════════════════════════════╝
def main():
    robot_ids = ROBOT_IDS

    print("=" * 55)
    print("  mouse_odom_proxy — multi-robot odometry proxy")
    print("=" * 55)
    for rid in robot_ids:
        name = robot_name(rid)
        domain = robot_domain(rid)
        print(f"  {name}  ← domain {domain}  |  domain 0 → /{name}/odometry")
    print("=" * 55)

    queue = Queue()

    # 每个机器人一个订阅子进程
    sub_processes = []
    for rid in robot_ids:
        p = Process(target=subscriber_process, args=(rid, queue),
                    name=f"sub-{robot_name(rid)}")
        sub_processes.append(p)

    # 统一发布进程
    pub_p = Process(target=publisher_process, args=(queue, robot_ids), name="pub")

    for p in sub_processes:
        p.start()
    pub_p.start()

    all_procs = sub_processes + [pub_p]
    try:
        for p in all_procs:
            p.join()
    except KeyboardInterrupt:
        print("\n[proxy] Shutting down...")
        for p in all_procs:
            p.terminate()
        for p in all_procs:
            p.join(timeout=3)


if __name__ == "__main__":
    main()