import os
import multiprocessing as mp
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Int32
#from robot_config import ROBOT_IDS as robot_ids

robot_ids = [3, 16]

# ================= 子进程逻辑：在特定 Domain 广播 =================
def domain_worker(domain_id, shared_sign):
    # 核心：强制修改当前子进程的环境变量
    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    
    # 初始化 rclpy（此时已处于目标 Domain）
    rclpy.init()
    node = rclpy.create_node(f'active_sign_broadcaster_{domain_id}')
    
    # 创建发布者 (BEST_EFFORT 匹配机器人端 chassis_controller 的 QoS)
    pub = node.create_publisher(
        Int32, '/active_sign',
        QoSProfile(depth=10,
                   reliability=ReliabilityPolicy.BEST_EFFORT,
                   durability=DurabilityPolicy.VOLATILE))
    
    # 定义 10Hz (0.1秒) 的定时器回调
    def timer_callback():
        msg = Int32()
        # 从多进程共享内存中实时读取最新的指令值
        msg.data = shared_sign.value  
        pub.publish(msg)
        
    timer = node.create_timer(0.1, timer_callback)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

# ================= 主进程逻辑：Domain 0 的监听与管理 =================
class MasterNode(Node):
    def __init__(self, shared_sign):
        super().__init__('master_domain_bridge')
        self.shared_sign = shared_sign
        
        # 保留外部话题控制的功能，方便后续用脚本一键急停
        self.sub = self.create_subscription(
            Int32,
            '/master_active_sign',
            self.master_callback,
            10
        )

    def master_callback(self, msg: Int32):
        if self.shared_sign.value != msg.data:
            self.shared_sign.value = msg.data
            self.get_logger().info(f">>> 接收到外部话题，全局指令已更新为: {msg.data}")

# ================= 键盘监听线程 =================
def wait_for_enter(shared_sign):
    # 阻塞等待用户敲击回车
    input("\n[等待指令] 正在以 10Hz 持续发送 0 建立通信...\n>>> 准备就绪后，请按下 [ENTER] 键同步发送 1 启动集群！\n")
    
    # 修改共享内存，所有子进程会在 0.1 秒内同步读取到 1
    shared_sign.value = 1
    print("\n=========================================")
    print("🚀 已触发！所有机器人的 active_sign 已同步切换为 1。")
    print("=========================================\n")
    print("（如需急停，可在终端使用 Ctrl+C，或通过话题发布 0）")

def main():
    # 1. 确保主进程运行在 Domain 0
    os.environ['ROS_DOMAIN_ID'] = '0'
    
    # 2. 创建跨进程共享内存变量，默认初始化为 0 (安全急停，同时用于建立 DDS 连接)
    shared_sign = mp.Value('i', 0)
    
    processes = []
    
    print("正在启动子 Domain 广播进程...")
    for r_id in robot_ids:
        target_domain = r_id + 10  
        
        # 启动子进程
        p = mp.Process(target=domain_worker, args=(target_domain, shared_sign))
        p.start()
        processes.append(p)
        print(f" [+] 已启动桥接: 目标 Robot {r_id} (ROS_DOMAIN_ID={target_domain})")
        
    # 3. 启动键盘监听线程
    # 设置为 daemon 守护线程，这样主程序退出时它会自动销毁
    enter_thread = threading.Thread(target=wait_for_enter, args=(shared_sign,))
    enter_thread.daemon = True
    enter_thread.start()

    # 4. 启动主进程节点
    rclpy.init()
    master_node = MasterNode(shared_sign)
    
    try:
        # spin 会维持 MasterNode 存活，并处理可能的话题回调
        rclpy.spin(master_node)
    except KeyboardInterrupt:
        print("\n收到退出信号，正在关闭跨域桥接集群...")
    finally:
        master_node.destroy_node()
        rclpy.shutdown()
        for p in processes:
            p.terminate()
            p.join()
        print("所有桥接进程已安全终止。")

if __name__ == '__main__':
    # 为了兼容性和稳定性，推荐使用 spawn 启动模式
    mp.set_start_method('spawn')  
    main()
