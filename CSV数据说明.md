# 双车追踪实验 CSV 数据说明

适用于当前实车程序 [demo_2d_bearing_tracking_direct_v1.py](demo_2d_bearing_tracking_direct_v1.py)
及异常处理程序 [recording_quality.py](recording_quality.py)。说明更新于 2026-10-29。
具体实验的参数以该次目录内的 `metadata.json` 为准。

## 1. 车辆身份与文件位置

| 角色 | 程序名称 | 小车名称 | 灯光默认颜色 | IP |
| --- | --- | --- | --- | --- |
| 被追者、目标车 | target | tianbot1 | 蓝色 | 192.168.50.208 |
| 追捕者 | pursuer | tianbot2 | 红色 | 192.168.50.154 |

每次实验创建独立目录：

```text
tracking_logs_vrpn/年月日_时分秒_进程号/
```

例如 `tracking_logs_vrpn/20261029_144013_50270/`。不同实验目录中的文件相互独立。

## 2. 应该使用哪一份文件

| 文件 | 内容 | 用途 |
| --- | --- | --- |
| `errors_and_positions.csv` | 清理后的 20 列误差、位置及误差重算向量 | 日常分析、重新画三项 error 和两车轨迹 |
| `errors_and_positions_raw.csv` | 同样的 20 列，保留异常样本 | 核对尖刺及清理前后的差异 |
| `tracking_data.csv` | 完整控制周期记录，当前版本 103 列 | 保存原始诊断记录、分析控制器和辨识器 |
| `tracking_data_clean.csv` | 完整字段，剔除采样质量异常段后的记录 | 分析清理后的全部诊断量 |
| `recording_quality.csv` | 每条原始记录的保留标记和剔除原因 | 按时间或原始行号追溯异常 |
| `tracking_data.npz` | 完整原始记录的 NumPy 数组 | Python 读取，数组键名与 CSV 列名一致 |
| `tracking_data_clean.npz` | 清理后的完整 NumPy 数组 | Python 读取清理数据 |
| `metadata.json` | 启动参数、话题、角色、坐标说明等 | 确认本次实验配置 |
| `summary.json` | **原始记录**的样本数、极值、RMS 等摘要 | 检查原始实验结果；其中极值可能包含已剔除的尖刺 |
| `recording_quality.json` | 生效的清理阈值、剔除样本数、异常时间段 | 解释清理结果、复现实验数据处理 |

当前程序在实验结束后使用带有异常段空缺的数据生成全程图片。CSV/NPZ 原始记录保留。
早期实验可能缺少清理版文件、集中导出表或最后新增的时间戳字段，应以 CSV 表头为准。

## 3. CSV 格式、类型和时间基准

- 编码为 UTF-8，逗号分隔，第一行是字段名，小数点为 `.`。
- CSV 本身没有强制的数据类型。以下表格中的类型是读取时建议采用的逻辑类型。
- `float64`：双精度浮点数；位置、速度、时间、误差通常属于此类型。
- `int64`：计数、行号等整数。
- `0/1`：标记位，可读为整数或布尔值。清理版完整 CSV 中可能写成 `0.0/1.0`，其含义相同。
- `string`：文本，主要用于异常原因。`recording_quality.csv` 的正常行原因是空字符串。
- `nan`：当前数值不可用，例如速度尚未初始化或低速下无法可靠计算方向误差；不能当作零误差。

每行对应一个已记录的控制周期，当前名义频率为 50 Hz，正常间隔约 0.02 s。
这不是对每一条 VRPN 原始消息的逐条录制：一个控制周期使用最新位姿；不满足控制条件时可能没有诊断行。
文件中的“原始记录”指采样质量清理前的控制诊断数值，部分量已经经过速度拟合或滤波。

`t` 是控制器初始化后的单调时钟经过时间，单位为秒。第一条通常约为 0.02 s，绘图横轴从 0 开始。
当前配置默认运行 180 s，也可以提前按 Ctrl+C 停止。启动前等待动捕的时间不计入 `t`。

清理版保留原有 `t`，不重新编号时间、不压缩被删除的时间段、不插值补齐样本。
因此，清理版相邻两行的 `t` 之差可能大于 0.02 s；应使用实际 `t` 绘图和计算时间间隔。
`control_dt_raw` 仍是该样本在原始控制循环中的间隔，不代表清理后相邻两行的间隔。

## 4. 坐标系与三个 error

位置、相对位置、估计位置以及带 `world` 的速度均为动捕世界坐标系中的二维量。
世界坐标原点及轴方向由动捕系统的标定决定。带 `body` 的速度位于对应小车的车体坐标系。
位置单位为 m，速度为 m/s，航向角 yaw 为 rad；方向误差的列名明确使用 deg。
CSV 保存二维 x/y 和 yaw，不包含完整三维位姿、原始四元数或视频。

定义：

```text
pT = [target_x, target_y]                         目标位置
pO = [pursuer_x, pursuer_y]                       追捕者位置
r  = pT - pO                                    真实相对向量
rHat = [relative_estimate_x, relative_estimate_y] 相对位置估计
rd   = [desired_relative_x, desired_relative_y]   期望相对向量
uT   = [target_mocap_velocity_x, target_mocap_velocity_y]
uTHat = [target_velocity_estimate_x, target_velocity_estimate_y]
```

| error 列名 | 类型 | 单位 | 定义与含义 |
| --- | --- | --- | --- |
| `input_error` | float64 | m/s | `norm(uT - uTHat)`，目标速度估计误差 |
| `localization_error` | float64 | m | `norm(r - rHat)`，相对位置估计误差 |
| `tracking_error` | float64 | m | `norm(r - rd)`，相对轨迹跟踪误差 |

这里 `norm([x,y]) = sqrt(x*x + y*y)`。三项都是向量差的模长，正常情况下非负。
`tracking_error` 同时反映距离和相位差，不等于单纯的 `abs(norm(r) - relative_radius)`。
`uT` 来自动捕位置的窗口拟合和低通处理，是实际速度的测量参考，不是速度指令。
目前启用直接动捕相对位置反馈，因此 `localization_error` 主要用于诊断观测器；其大小不等于实际跟踪误差。

## 5. 常用表：errors_and_positions.csv（20 列）

`errors_and_positions_raw.csv` 的字段和类型相同，只是保留所有原始样本。

| 列名 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `t` | float64 | s | 控制器初始化后的经过时间 |
| `input_error` | float64 | m/s | 目标速度估计误差 |
| `localization_error` | float64 | m | 相对位置估计误差 |
| `tracking_error` | float64 | m | 相对轨迹跟踪误差 |
| `target_x`, `target_y` | float64 | m | 蓝色目标车的世界坐标位置 |
| `target_yaw` | float64 | rad | 目标车航向角 |
| `pursuer_x`, `pursuer_y` | float64 | m | 红色追捕者的世界坐标位置 |
| `pursuer_yaw` | float64 | rad | 追捕者航向角 |
| `relative_x`, `relative_y` | float64 | m | 当前真实相对向量 r，方向从追捕者指向目标 |
| `relative_estimate_x`, `relative_estimate_y` | float64 | m | 观测器估计的相对位置 rHat |
| `desired_relative_x`, `desired_relative_y` | float64 | m | 期望相对向量 rd；其半径为配置中的 relative_radius |
| `target_mocap_velocity_x`, `target_mocap_velocity_y` | float64 | m/s | 目标车动捕位置拟合、低通后的实际速度参考 |
| `target_velocity_estimate_x`, `target_velocity_estimate_y` | float64 | m/s | 算法输出的目标速度估计 uTHat |

这些列足以重新计算三项 error，以及绘制两车的二维轨迹。

## 6. 完整表：tracking_data.csv 的其他字段

本节加上第 5 节覆盖当前完整表的全部 103 个字段。`tracking_data_clean.csv` 具有同一批字段。
下表中的 x/y 成对列各自占一列。

### 6.1 控制时间与位姿质量

| 列名 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `control_dt_raw` | float64 | s | 本次控制回调距上次回调的实际间隔 |
| `control_dt_used` | float64 | s | 实际供 RK4 积分使用的间隔；异常长间隔会受到数值限制 |
| `target_pose_age`, `pursuer_pose_age` | float64 | s | 当前控制时刻距该车最新位姿消息接收时刻的时间 |
| `target_pose_sample_time`, `pursuer_pose_sample_time` | float64 | s | 最新位姿消息的采样时间戳。通常来自 ROS header；header 时间无效时回退到本机单调时钟，不能直接与 t 或日历时间混用 |
| `target_velocity_timestamp_resets`, `pursuer_velocity_timestamp_resets` | int64 | 次 | 该速度估计器因收到非递增采样时间戳而重置窗口的累计次数 |

最后两行的四个字段为新增诊断字段，旧实验 CSV 可能没有。

### 6.2 相对状态

| 列名 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `bearing_x`, `bearing_y` | float64 | 无 | 送入辨识器的单位方位向量；可按 mocap_velocity_alignment_delay 回溯对齐，所以可能与当前 r 的归一化结果略有不同 |
| `range_value` | float64 | m | 实际两车距离 norm(r) |

### 6.3 目标车速度和执行误差

| 列名 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `target_command_world_x`, `target_command_world_y` | float64 | m/s | 目标车最终世界系速度指令，包含可选边界修正及限幅 |
| `target_nominal_command_world_x`, `target_nominal_command_world_y` | float64 | m/s | 原始目标运动律给出的指令，尚未叠加边界修正或限幅 |
| `target_boundary_command_world_x`, `target_boundary_command_world_y` | float64 | m/s | 可选软边界产生的速度修正；关闭软边界时为零 |
| `target_boundary_active` | 0/1 | 无 | 本周期是否有非零边界修正 |
| `target_boundary_clearance` | float64 | m | 目标位置到配置矩形边界的最小有符号间距；边界内为正、越界为负，即使关闭软边界也会计算 |
| `target_command_body_x`, `target_command_body_y` | float64 | m/s | 用目标车自身 yaw 转换后的车体系平移指令 |
| `target_raw_velocity_x`, `target_raw_velocity_y` | float64 | m/s | 目标车位置窗口最小二乘拟合的速度，尚未经过一阶低通；仍属于计算值 |
| `target_command_execution_error` | float64 | m/s | norm(目标世界系指令 - 目标动捕实际速度) |
| `target_input_estimation_error_command` | float64 | m/s | norm(目标世界系指令 - uTHat)，用于区分指令模型误差与实速估计误差 |

`target_mocap_velocity_x/y` 与 `target_velocity_estimate_x/y` 见第 5 节。

### 6.4 追捕者速度和执行误差

| 列名 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `pursuer_speed` | float64 | m/s | 下发世界系速度指令的模长，不是动捕实速 |
| `raw_command_world_x`, `raw_command_world_y` | float64 | m/s | 追捕者控制律的原始世界系输出，尚未限幅 |
| `command_world_x`, `command_world_y` | float64 | m/s | 追捕者最终世界系速度指令 |
| `command_body_x`, `command_body_y` | float64 | m/s | 用追捕者自身 yaw 转换后的车体系平移指令 |
| `mocap_velocity_x`, `mocap_velocity_y` | float64 | m/s | 追捕者动捕位置拟合、低通后的实际速度参考；字段名没有 pursuer 前缀 |
| `raw_mocap_velocity_x`, `raw_mocap_velocity_y` | float64 | m/s | 追捕者位置窗口拟合速度，尚未经过一阶低通 |
| `command_execution_error` | float64 | m/s | norm(追捕者最终世界系指令 - 追捕者动捕实际速度) |
| `direction_error_deg` | float64 | deg | 从指令速度方向到动捕实速方向的有符号夹角；任一速度过小时为 nan |

计算追捕者实际速度模长时使用：

```text
sqrt(mocap_velocity_x**2 + mocap_velocity_y**2)
```

### 6.5 追捕者控制律的组成项

| 列名 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `control_target_estimate_x`, `control_target_estimate_y` | float64 | m/s | 生成当前指令时使用的目标速度估计项 |
| `control_relative_velocity_x`, `control_relative_velocity_y` | float64 | m/s | 观测器的目标输入残差估计 deltaVHat；名称沿用旧版，不是普通的相对速度或追捕者速度 |
| `control_trajectory_x`, `control_trajectory_y` | float64 | m/s | 期望相对轨迹的前馈项 -rdDot |
| `control_feedback_x`, `control_feedback_y` | float64 | m/s | 位置反馈项 Kr × (反馈相对位置 - rd) |
| `control_feedback_position_x`, `control_feedback_position_y` | float64 | m | 当前实际用于位置反馈的相对位置 |
| `control_feedback_uses_mocap` | 0/1 | 无 | 1 使用动捕 r 反馈；0 使用观测器 rHat 反馈 |

限幅前的追捕者指令满足：

```text
raw_command_world = control_target_estimate
                  + control_relative_velocity
                  + control_trajectory
                  + control_feedback
```

`control_target_estimate_x/y` 是计算指令时使用的值，`target_velocity_estimate_x/y`
是推进本次估计状态后用于诊断的值，两者可能略有差别。核对控制量分解应使用前者。

### 6.6 辨识器、观测器和历史栈

“算法量”表示内部矩阵或激励量，不能直接作为位置或速度误差解读。

| 列名 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `theta_hat_norm` | float64 | m/s | RBF 速度模型参数向量 thetaHat 的模长 |
| `identifier_current_theta_dot_norm` | float64 | m/s² | 当前窗口学习项产生的参数变化率模长 |
| `identifier_history_theta_dot_norm` | float64 | m/s² | 历史栈学习项产生的参数变化率模长 |
| `identifier_leak_theta_dot_norm` | float64 | m/s² | 泄漏/正则项产生的参数变化率模长 |
| `identifier_current_output_rate_x`, `identifier_current_output_rate_y` | float64 | m/s² | 当前窗口项经 Phi(b) 映射后的速度输出变化率贡献 |
| `identifier_history_output_rate_x`, `identifier_history_output_rate_y` | float64 | m/s² | 历史栈项经 Phi(b) 映射后的速度输出变化率贡献 |
| `identifier_leak_output_rate_x`, `identifier_leak_output_rate_y` | float64 | m/s² | 泄漏项经 Phi(b) 映射后的速度输出变化率贡献 |
| `identifier_total_output_rate_x`, `identifier_total_output_rate_y` | float64 | m/s² | 上述三种输出变化率贡献之和 |
| `xi_position_norm` | float64 | m | 观测器位置部分的模长 |
| `xi_velocity_norm` | float64 | m/s | 观测器输入残差估计部分的模长，不是小车实速 |
| `po_min_eigenvalue`, `po_max_eigenvalue` | float64 | 算法量 | 观测器 Riccati 矩阵 Po 的最小、最大特征值 |
| `po_condition` | float64 | 无 | Po 的特征值条件数 |
| `history_count` | int64 | 个 | 已保留的历史窗口数，上限由 history_size 配置 |
| `active_directions` | int64 | 个 | 历史投影矩阵的秩，即激活的 RBF 参数方向数，不必等于 RBF 节点数 |
| `history_min_active_eigenvalue` | float64 | 算法量 | 历史聚合矩阵的最小激活特征值 |
| `history_max_eigenvalue` | float64 | 算法量 | 历史聚合矩阵的最大特征值 |
| `history_condition` | float64 | 无 | 历史聚合矩阵激活谱的条件数 |
| `history_update_gated` | 0/1 | 无 | 1 表示历史学习项因条件数门限而被抑制 |
| `current_window_count` | int64 | 个 | 当前有限数据窗口中累计的控制样本数 |
| `current_window_excitation` | float64 | s | 当前窗口的方向激励积分 a；不是单纯的窗口持续时间 |
| `chi` | float64（0/1） | 无 | 当前窗口是否满足参与在线学习的激励条件 |
| `history_last_window_execution_error` | float64 | m/s | 最近完成窗口内追捕者指令与实速差的 RMS，用于历史窗口质量门 |
| `history_last_window_accepted` | 0/1 | 无 | 最近完成的窗口是否通过质量门并进入历史栈 |

`identifier_*_output_rate` 是 Phi(b) × 相应 thetaHat 参数变化率，不是完整的 uTHat 时间导数，
还需区分 RBF 基函数随方位的变化及目标速度估计的一阶滤波。

### 6.7 速度拟合质量

| 列名 | 类型 | 单位 | 含义 |
| --- | --- | --- | --- |
| `target_velocity_sample_count`, `pursuer_velocity_sample_count` | int64 | 帧 | 当前速度回归窗口实际使用的动捕位置样本数 |
| `target_velocity_sample_span`, `pursuer_velocity_sample_span` | float64 | s | 回归窗口内最早至最新位置样本的时间跨度 |
| `target_velocity_sample_accepted`, `pursuer_velocity_sample_accepted` | 0/1 | 无 | 本次调用是否更新了有效速度估计。0 也可能只是尚未收到新位姿，并不单独作为异常剔除依据 |

## 7. 异常记录表：recording_quality.csv

此文件与原始 `tracking_data.csv` 一行对应一行。

| 列名 | 类型 | 含义 |
| --- | --- | --- |
| `sample_index` | int64 | 原始数据的行号，从 0 开始，不计表头 |
| `t` | float64 | 对应原始记录的经过时间，单位 s |
| `recording_valid` | 0/1 | 1 保留在清理版；0 从清理版排除 |
| `reason` | string | 排除原因；多个原因用 `|` 连接，正常行为留空 |

| reason 取值 | 含义 |
| --- | --- |
| `pose_update_gap` | 最新位姿的接收年龄超过记录质量阈值 |
| `velocity_window_too_short` | 完成启动预热后，速度回归窗口的实际跨度过短 |
| `control_timing_gap` | 控制间隔异常，或记录时间没有递增 |
| `nonfinite_measurement` | 处理规则检查的位置、速度等必要数值包含 nan/inf |
| `pose_timestamp_reset` | 速度估计器累计的非递增位姿时间戳重置次数增加 |

异常标记还会覆盖异常前的一个拟合窗口和异常后的恢复期，因此部分被剔除行可能没有独立尖刺。
当前默认条件为：位姿接收年龄超过 0.02 s；速度拟合跨度短于 0.06 s × 0.5 = 0.03 s；
控制间隔超过两个名义周期或非正值；必要数值非有限；或位姿时间戳发生重置。
速度窗口过短的判定在 t ≥ 两个名义拟合窗口后才启用，以保留正常启动过程。
异常段默认向前覆盖 0.06 s，向后覆盖 0.15 s。实际生效阈值见本次 `recording_quality.json`。

剔除规则不使用三项 error 的幅值，因此正常启动、收敛过程中的较大误差仍会保留。
绘图在剔除位置留空，不插值补点。原始 `summary.json` 保持原始统计口径，
清理后的指标应从清理版 CSV/NPZ 重新计算。

## 8. 读取示例

### Python / NumPy

```python
from pathlib import Path
import numpy as np

folder = Path("tracking_logs_vrpn/20261029_144013_50270")
data = np.atleast_1d(np.genfromtxt(
    folder / "errors_and_positions.csv",
    delimiter=",", names=True, encoding="utf-8",
))

t = data["t"]
e_u = data["input_error"]
e_pos = data["localization_error"]
e_track = data["tracking_error"]
target_xy = np.column_stack((data["target_x"], data["target_y"]))
pursuer_xy = np.column_stack((data["pursuer_x"], data["pursuer_y"]))

# 示例：取 60 s 之后的有效样本，计算跟踪误差的样本 RMS。
mask = (t >= 60.0) & np.isfinite(e_track)
if mask.any():
    tracking_rms = np.sqrt(np.mean(e_track[mask] ** 2))
    print("tracking RMS (m):", tracking_rms)
```

这是对保留样本的 RMS；需要严格的时间加权统计时，应结合实际 `t` 和异常区间计算。
`recording_quality.csv` 含文本 reason，建议使用 `csv.DictReader` 或 pandas 读取，不能把所有列都强制读为浮点数。

### MATLAB

```matlab
T = readtable('tracking_logs_vrpn/20261029_144013_50270/errors_and_positions.csv');
plot(T.t, T.tracking_error);
xlabel('time (s)'); ylabel('tracking error (m)'); grid on;
```

## 9. 文件来源与追溯

- 记录字段及三项误差定义：[实车程序](demo_2d_bearing_tracking_direct_v1.py)。
- 清理规则及原因编码：[recording_quality.py](recording_quality.py)。
- 当前启动配置：[bearing_tracking_tianbot_vrpn_params.yaml](bearing_tracking_tianbot_vrpn_params.yaml)。
- 启动与录制流程：[tianbot.md](tianbot.md)。
- 已有数据示例：[20261029_144013_50270](tracking_logs_vrpn/20261029_144013_50270/)。
