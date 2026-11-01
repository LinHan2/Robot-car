# 双车追踪算法启动

## 1. 控制电脑：启动 ROS Master

```bash
source /opt/ros/noetic/setup.bash
unset ROS_HOSTNAME
export ROS_IP=192.168.50.189
export ROS_MASTER_URI=http://192.168.50.189:11311
rosparam get /rosversion >/dev/null 2>&1 || roscore
```

## 2. 两辆车：连接底盘到当前 Master

目标车（`.208`）：

```bash
ssh tianbot@192.168.50.208
sudo systemctl restart tianbot1-base.service
exit
```

追踪车（`.154`）：

```bash
ssh tianbot@192.168.50.154
sudo systemctl restart tianbot2-base.service
exit
```

## 3. 控制电脑：启动 VRPN（新终端）

```bash
source /opt/ros/noetic/setup.bash
unset ROS_HOSTNAME
export ROS_IP=192.168.50.189
export ROS_MASTER_URI=http://192.168.50.189:11311
roslaunch vrpn_client_ros sample.launch server:=192.168.50.95
```

## 4. 控制电脑：启动追踪算法（新终端）

默认录制 180 秒，到时自动停车并保存全程数据；也可提前按 `Ctrl+C`。

```bash
cd ~/Robot-car
source /opt/ros/noetic/setup.bash
unset ROS_HOSTNAME
export ROS_IP=192.168.50.189
export ROS_MASTER_URI=http://192.168.50.189:11311
PYTHONNOUSERSITE=1 python3 demo_2d_bearing_tracking_direct_v1.py \
  --config bearing_tracking_tianbot_vrpn_params.yaml
```

每次数据保存在 `tracking_logs_vrpn/<时间戳_PID>/`：完整数值为
`tracking_data.csv` / `tracking_data.npz`，三项误差和两车位置另存为
`errors_and_positions.csv`（已按采样质量剔除异常段）。完整原始记录为
`tracking_data.csv` / `tracking_data.npz`，清理后的完整数据为
`tracking_data_clean.csv` / `tracking_data_clean.npz`，剔除时间与原因见
`recording_quality.json`。等待终端显示 `[DATA] saved experiment data` 后再关闭。
临时改时长可在启动命令末尾加 `--duration 240`；`--duration 0` 为手动停止。

字段类型、单位与清理规则见 [CSV 数据说明](CSV数据说明.md)。

## 只调整灯光（读取 bearing_tracking_params.yaml）

```bash
cd ~/Robot-car
source /opt/ros/noetic/setup.bash
unset ROS_HOSTNAME
export ROS_IP=192.168.50.189
export ROS_MASTER_URI=http://192.168.50.189:11311
PYTHONNOUSERSITE=1 python3 demo_2d_bearing_tracking_direct_v1.py --leds-only
```
