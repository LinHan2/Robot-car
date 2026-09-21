#!/usr/bin/env python3
import os
import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from multiprocessing import Process, Queue, Event
from robot_config import ROBOT_IDS, CMD_DOMAIN_ID, robot_name, robot_domain

# -----------------------------
# 配置参数
# -----------------------------
CMD_TIMEOUT = 0.3       # 超时 (秒): 超过此时间无新指令则回退零速


def get_robot_config(robot_id):
    return {
        "id": robot_id,
        "name": robot_name(robot_id),
        "cmd_domain": CMD_DOMAIN_ID,
        "robot_domain": robot_domain(robot_id),
        "sub_topic": f"/{robot_name(robot_id)}/vel_cmd",
        "pub_topic": f"/{robot_name(robot_id)}/vel_cmd"
    }


# -----------------------------
# 订阅进程：监听算法指令
# -----------------------------
def subscriber_process(robot_config, vel_queue, ready_event):
    os.environ["ROS_DOMAIN_ID"] = str(robot_config["cmd_domain"])
    rclpy.init()
    node = Node(f"{robot_config['name']}_sub_node")

    def cmd_callback(msg: TwistStamped):
        while not vel_queue.empty():
            try: vel_queue.get_nowait()
            except: break
        vel_queue.put(msg)

    node.create_subscription(TwistStamped, robot_config["sub_topic"], cmd_callback, 10)
    ready_event.set()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# -----------------------------
# 发布进程：维持心跳并转发
# -----------------------------
def publisher_process(robot_config, vel_queue, ready_event):
    os.environ["ROS_DOMAIN_ID"] = str(robot_config["robot_domain"])
    rclpy.init()
    node = Node(f"{robot_config['name']}_pub_node")
    pub = node.create_publisher(TwistStamped, robot_config["pub_topic"], 10)

    last_msg = TwistStamped()
    last_msg.header.frame_id = "base_link"
    last_cmd_time = time.time()
    zero_msg = TwistStamped()
    zero_msg.header.frame_id = "base_link"

    ready_event.set()

    def timer_callback():
        nonlocal last_msg, last_cmd_time

        while not vel_queue.empty():
            try:
                last_msg = vel_queue.get_nowait()
                last_cmd_time = time.time()
            except:
                break

        if time.time() - last_cmd_time > CMD_TIMEOUT:
            zero_msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(zero_msg)
            return

        last_msg.header.stamp = node.get_clock().now().to_msg()
        pub.publish(last_msg)

    node.create_timer(0.02, timer_callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# -----------------------------
# 主程序
# -----------------------------
class RobotForwarder:
    def __init__(self, config):
        self.config = config
        self.vel_queue = Queue()
        self.sub_ready = Event()
        self.pub_ready = Event()
        self.sub_p = None
        self.pub_p = None

    def start(self):
        self.sub_p = Process(target=subscriber_process, args=(self.config, self.vel_queue, self.sub_ready))
        self.pub_p = Process(target=publisher_process, args=(self.config, self.vel_queue, self.pub_ready))
        self.sub_p.start()
        self.pub_p.start()
        return self.sub_ready, self.pub_ready

    def stop(self):
        if self.sub_p: self.sub_p.terminate()
        if self.pub_p: self.pub_p.terminate()


def main():
    print(">>> Starting Multi-Robot Forwarder (passthrough)...")
    configs = [get_robot_config(rid) for rid in ROBOT_IDS]
    forwarders = [RobotForwarder(c) for c in configs]

    sub_events, pub_events = [], []
    for f in forwarders:
        s_ev, p_ev = f.start()
        sub_events.append(s_ev)
        pub_events.append(p_ev)
        time.sleep(1)

    print("Waiting for all nodes to initialize...")
    for e in sub_events + pub_events:
        e.wait()

    print("\n✅ All nodes ready. Forwarding (passthrough, no rotation).")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
        for f in forwarders:
            f.stop()


if __name__ == "__main__":
    import multiprocessing
    try: multiprocessing.set_start_method('spawn')
    except: pass
    main()
