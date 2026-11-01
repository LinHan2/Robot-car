#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_2d_bearing_tracking_direct_v1.py
=====================================

ROS 1 implementation of ``demo_2d_bearing_tracking_direct_v1.m``.

The node controls a fully-actuated omnidirectional target (T) with the
MATLAB evasion law and a pursuer (O) that tracks it using bearing-only data.

Default topics:
    /vrpn_client_node/tianbot{1,2}/pose  geometry_msgs/PoseStamped
    /tianbot{1,2}/cmd_vel                geometry_msgs/Twist

The paper/MATLAB algorithm is written in a world frame.  The robot command
interface is body-frame, so the computed world velocity is rotated using each
robot's VRPN yaw before publishing.  ``angular.z`` is always zero.

Default dual-Tianbot launch:
    python3 demo_2d_bearing_tracking_direct_v1.py

The bridge and both chassis should already be running, and the chassis should
be unlocked separately with ``utils/activate_sign.py``.

中文阅读说明
------------
1. 所有带 ``world`` 的二维向量都位于动捕世界坐标系（x 向北、y 向西）。
2. ``vel_cmd`` 要求车体坐标系速度，因此只在发布前执行 world -> body 旋转。
3. 默认论文模式对目标只使用单位方位 ``b``；可选混合模式允许反馈项直接使用
   动捕相对位置，但此时不再是严格的 bearing-only 控制。
4. ``uO_cmd`` 是 pursuer 的控制指令，``u_o_actual`` 是由 pursuer 动捕状态
   估计得到的实际世界速度。辨识器和观测器使用后者，不用控制指令冒充实际速度。
5. 默认读取双 Tianbot VRPN 配置；可用 ``--config`` 切换其他实验配置。
"""

import argparse
import csv
import json
import math
import os
import queue
import signal
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import yaml

from recording_quality import filter_recording, save_quality_exports

import rospy
from geometry_msgs.msg import PoseStamped, Twist


DEFAULT_PLOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tracking_diagnostics.png"
)
DEFAULT_ANALYSIS_PLOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tracking_analysis.png"
)
DEFAULT_COMPARISON_PLOT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "bearing_tracking_comparison.png"
)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "tracking_logs"
)
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "bearing_tracking_tianbot_vrpn_params.yaml",
)


# ============================================================================
# 基础工具：机器人命名、二维坐标变换和速度限幅
# ============================================================================


def robot_name(robot_id: int) -> str:
    """把整数 ID 转成 ROS 名称，例如 3 -> ``robot03``。"""
    return f"robot{robot_id:02d}"


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """ROS 四元数 -> 平面偏航角 yaw（绕 z 轴，单位 rad）。"""
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def world_to_body(v_world: np.ndarray, yaw: float) -> np.ndarray:
    """世界系速度 -> 车体系速度。

    ``yaw`` 是车体 x 轴相对世界 x 轴的角度。该旋转只改变方向、不改变模长。
    返回值的 x 表示车头方向，y 表示车体左侧方向。
    """
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([
        c * v_world[0] + s * v_world[1],
        -s * v_world[0] + c * v_world[1],
    ], dtype=float)


def limit_vector_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    """保持向量方向不变，只限制二维速度模长；maximum<=0 表示不处理。"""
    norm = float(np.linalg.norm(vector))
    if norm > maximum > 0.0:
        return vector * (maximum / norm)
    return vector


@dataclass
class PoseCache:
    """某辆车最新一帧动捕位姿。

    ``receive_time`` 用于判断 ROS 数据是否超时；``sample_time`` 来自消息时间戳，
    用于位置序列拟合速度，避免回调调度抖动污染差分时间。
    """
    position: Optional[np.ndarray] = None
    yaw: float = 0.0
    receive_time: float = 0.0
    sample_time: float = 0.0


@dataclass
class TelemetrySample:
    """一帧诊断数据。

    此结构通过非阻塞队列送往打印线程和绘图进程。除已经单独计算好的
    ``mocap_velocity_*`` 外，其中误差、真实距离等量都不会反馈到控制器。
    """

    # 控制时序和动捕状态。
    t: float
    control_dt_raw: float
    control_dt_used: float
    target_x: float
    target_y: float
    target_yaw: float
    pursuer_x: float
    pursuer_y: float
    pursuer_yaw: float
    target_pose_age: float
    pursuer_pose_age: float

    # 真实相对向量、唯一送入算法的单位方位、期望/估计相对向量。
    relative_x: float
    relative_y: float
    bearing_x: float
    bearing_y: float
    desired_relative_x: float
    desired_relative_y: float
    relative_estimate_x: float
    relative_estimate_y: float
    range_value: float
    tracking_error: float
    localization_error: float
    input_error: float

    # target：运动律给出的世界系指令、动捕实际速度、算法速度估计。
    target_command_world_x: float
    target_command_world_y: float
    target_nominal_command_world_x: float
    target_nominal_command_world_y: float
    target_boundary_command_world_x: float
    target_boundary_command_world_y: float
    target_boundary_active: int
    target_boundary_clearance: float
    target_command_body_x: float
    target_command_body_y: float
    target_mocap_velocity_x: float
    target_mocap_velocity_y: float
    target_raw_velocity_x: float
    target_raw_velocity_y: float
    target_velocity_estimate_x: float
    target_velocity_estimate_y: float
    target_command_execution_error: float
    target_input_estimation_error_command: float
    target_velocity_sample_count: int
    target_velocity_sample_span: float
    target_velocity_sample_accepted: int

    # pursuer：控制器原始输出、最终发送值、车体系值和动捕实际速度。
    pursuer_speed: float
    raw_command_world_x: float
    raw_command_world_y: float
    command_world_x: float
    command_world_y: float
    command_body_x: float
    command_body_y: float
    mocap_velocity_x: float
    mocap_velocity_y: float
    raw_mocap_velocity_x: float
    raw_mocap_velocity_y: float
    command_execution_error: float
    direction_error_deg: float

    # 组成 uO 的四项，满足 raw_uO = c1+c2+c3+c4。
    # 兼容旧 CSV 保留 control_relative_velocity 字段名；它实际是 deltaVHat，
    # 即 target 输入估计残差，不是 pursuer 或普通相对速度。
    control_target_estimate_x: float
    control_target_estimate_y: float
    control_relative_velocity_x: float
    control_relative_velocity_y: float
    control_trajectory_x: float
    control_trajectory_y: float
    control_feedback_x: float
    control_feedback_y: float
    control_feedback_position_x: float
    control_feedback_position_y: float
    control_feedback_uses_mocap: int

    # 辨识器、观测器、历史栈和速度差分质量。
    theta_hat_norm: float
    identifier_current_theta_dot_norm: float
    identifier_history_theta_dot_norm: float
    identifier_leak_theta_dot_norm: float
    identifier_current_output_rate_x: float
    identifier_current_output_rate_y: float
    identifier_history_output_rate_x: float
    identifier_history_output_rate_y: float
    identifier_leak_output_rate_x: float
    identifier_leak_output_rate_y: float
    identifier_total_output_rate_x: float
    identifier_total_output_rate_y: float
    xi_position_norm: float
    xi_velocity_norm: float
    po_min_eigenvalue: float
    po_max_eigenvalue: float
    po_condition: float
    history_count: int
    active_directions: int
    history_min_active_eigenvalue: float
    history_max_eigenvalue: float
    history_condition: float
    history_update_gated: int
    current_window_count: int
    current_window_excitation: float
    chi: float
    history_last_window_execution_error: float
    history_last_window_accepted: int
    pursuer_velocity_sample_count: int
    pursuer_velocity_sample_span: float
    pursuer_velocity_sample_accepted: int
    target_pose_sample_time: float = float("nan")
    pursuer_pose_sample_time: float = float("nan")
    target_velocity_timestamp_resets: int = 0
    pursuer_velocity_timestamp_resets: int = 0


class MocapVelocityEstimator:
    """用 VRPN 位置窗口拟合世界系实际速度，并保留完整质量诊断。

    不订阅里程计或任何速度话题。pursuer 的位置差分结果进入算法积分；target
    的同类估计只用于检验“指令=实际”这一 MATLAB 理想假设。异常位姿跳变时保留
    上一有效结果，避免把单帧动捕错误注入辨识器和观测器。
    """

    def __init__(self, window: float, filter_tau: float,
                 max_valid_speed: float, max_samples: int = 256):
        self.window = window
        self.filter_tau = filter_tau
        self.max_valid_speed = max_valid_speed
        self.samples = deque(maxlen=max_samples)
        self.timestamp_reset_count = 0
        self.previous_estimate_time: Optional[float] = None
        self.filtered_velocity = np.zeros(2, dtype=float)
        self.have_filtered_velocity = False
        self.raw_velocity = np.zeros(2, dtype=float)
        self.sample_count = 0
        self.sample_span = 0.0
        self.sample_accepted = False

    def reset(self) -> None:
        """清除跨停机、时间戳跳变或实验启动前的速度历史。"""
        self.samples.clear()
        self.previous_estimate_time = None
        self.filtered_velocity.fill(0.0)
        self.have_filtered_velocity = False
        self.raw_velocity.fill(0.0)
        self.sample_count = 0
        self.sample_span = 0.0
        self.sample_accepted = False

    def add_pose(self, sample_time: float, position: np.ndarray) -> None:
        """加入动捕位置；时间不递增时自动切断旧回归窗口。"""
        if self.samples and sample_time <= self.samples[-1][0]:
            self.timestamp_reset_count += 1
            self.reset()
        self.samples.append((sample_time, position.copy()))

    def estimate(self, nominal_dt: float) -> np.ndarray:
        """返回最近窗口的位置拟合速度；数据不足/异常时保持上一有效结果。"""
        self.sample_accepted = False
        if len(self.samples) < 2:
            self.sample_count = len(self.samples)
            self.sample_span = 0.0
            return self.filtered_velocity.copy()

        latest_time = self.samples[-1][0]
        cutoff_time = latest_time - self.window
        window_samples = [
            sample for sample in self.samples
            if sample[0] >= cutoff_time - 1.0e-9
        ]
        self.sample_count = len(window_samples)
        self.sample_span = float(latest_time - window_samples[0][0])
        if self.sample_span <= 1.0e-4:
            return self.filtered_velocity.copy()

        times = np.asarray([sample[0] for sample in window_samples], dtype=float)
        positions = np.asarray([sample[1] for sample in window_samples], dtype=float)
        centered_times = times - float(np.mean(times))
        time_energy = float(centered_times @ centered_times)
        if time_energy <= 1.0e-12:
            return self.filtered_velocity.copy()
        self.raw_velocity = np.asarray(
            centered_times @ positions / time_energy, dtype=float
        )

        # 同一批动捕帧只更新一次滤波器，50 Hz 控制定时器可重复读取当前结果。
        if self.previous_estimate_time == latest_time:
            return self.filtered_velocity.copy()
        estimate_dt = (
            nominal_dt if self.previous_estimate_time is None
            else max(latest_time - self.previous_estimate_time, 1.0e-4)
        )
        self.previous_estimate_time = latest_time

        raw_speed = float(np.linalg.norm(self.raw_velocity))
        if (not np.all(np.isfinite(self.raw_velocity))
                or (self.max_valid_speed > 0.0
                    and raw_speed > self.max_valid_speed)):
            return self.filtered_velocity.copy()

        if not self.have_filtered_velocity:
            self.filtered_velocity = self.raw_velocity.copy()
            self.have_filtered_velocity = True
        else:
            alpha = (
                1.0 if self.filter_tau <= 0.0
                else 1.0 - math.exp(-estimate_dt / self.filter_tau)
            )
            self.filtered_velocity += alpha * (
                self.raw_velocity - self.filtered_velocity
            )
        self.sample_accepted = True
        return self.filtered_velocity.copy()


class DiagnosticsRecorderWorker(threading.Thread):
    """后台保存完整实验数据；磁盘 I/O 不在 ROS 控制定时器中执行。"""

    def __init__(self, samples: queue.Queue, output_dir: str, metadata: dict,
                 latest_comparison_path: Optional[str],
                 latest_plot_path: Optional[str] = None,
                 latest_analysis_path: Optional[str] = None,
                 save_figures: bool = True):
        super().__init__(name="tracking-recorder", daemon=False)
        self.samples = samples
        self.output_dir = output_dir
        self.metadata = metadata
        self.stop_event = threading.Event()
        self.csv_path = os.path.join(output_dir, "tracking_data.csv")
        self.npz_path = os.path.join(output_dir, "tracking_data.npz")
        self.summary_path = os.path.join(output_dir, "summary.json")
        self.comparison_path = os.path.join(
            output_dir, "bearing_tracking_sim_comparison.png"
        )
        self.latest_comparison_path = latest_comparison_path
        self.latest_plot_path = latest_plot_path
        self.latest_analysis_path = latest_analysis_path
        self.save_figures = save_figures

    def stop(self) -> None:
        self.stop_event.set()

    @staticmethod
    def _finite_max(values: np.ndarray) -> Optional[float]:
        finite = values[np.isfinite(values)]
        return None if finite.size == 0 else float(np.max(finite))

    def _save_summary(self, columns: dict) -> None:
        """生成无需打开图就能快速筛查的实验统计摘要。"""
        if not columns["t"]:
            summary = {"samples": 0, "duration_s": 0.0}
        else:
            arrays = {key: np.asarray(value) for key, value in columns.items()}
            actual_speed = np.hypot(
                arrays["mocap_velocity_x"], arrays["mocap_velocity_y"]
            )
            target_actual_speed = np.hypot(
                arrays["target_mocap_velocity_x"],
                arrays["target_mocap_velocity_y"],
            )
            summary = {
                "samples": int(len(arrays["t"])),
                "duration_s": float(arrays["t"][-1] - arrays["t"][0]),
                "mean_raw_control_dt_s": float(np.mean(arrays["control_dt_raw"])),
                "max_raw_control_dt_s": self._finite_max(arrays["control_dt_raw"]),
                "minimum_range_m": float(np.min(arrays["range_value"])),
                "maximum_range_m": float(np.max(arrays["range_value"])),
                "maximum_tracking_error_m": self._finite_max(arrays["tracking_error"]),
                "maximum_localization_error_m": self._finite_max(arrays["localization_error"]),
                "maximum_input_error_mps": self._finite_max(arrays["input_error"]),
                "maximum_pursuer_actual_speed_mps": self._finite_max(actual_speed),
                "maximum_target_actual_speed_mps": self._finite_max(target_actual_speed),
                "rms_pursuer_command_execution_error_mps": float(np.sqrt(np.mean(
                    arrays["command_execution_error"] ** 2
                ))),
                "rms_target_command_execution_error_mps": float(np.sqrt(np.mean(
                    arrays["target_command_execution_error"] ** 2
                ))),
                "final_history_windows": int(arrays["history_count"][-1]),
                "final_active_directions": int(arrays["active_directions"][-1]),
            }
        with open(self.summary_path, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2, ensure_ascii=False)

    def _save_recording_export(self, columns: dict,
                               filename: str = "errors_and_positions.csv") -> None:
        """另存实验录制常用的逐帧误差、位置及误差重算所需的向量。"""
        names = (
            "t", "input_error", "localization_error", "tracking_error",
            "target_x", "target_y", "target_yaw",
            "pursuer_x", "pursuer_y", "pursuer_yaw",
            "relative_x", "relative_y",
            "relative_estimate_x", "relative_estimate_y",
            "desired_relative_x", "desired_relative_y",
            "target_mocap_velocity_x", "target_mocap_velocity_y",
            "target_velocity_estimate_x", "target_velocity_estimate_y",
        )
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(names)
            writer.writerows(zip(*(columns[name] for name in names)))

    def _save_separate_plots(self, columns: dict) -> None:
        """使用完整 CSV 数据生成四张互相独立的实验图。

        该函数只在实验结束后由记录线程调用，不位于 ROS 控制回调或实时绘图
        进程中，因此 Matplotlib 写盘不会影响 50 Hz 控制周期。
        """
        if not columns["t"]:
            return

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        arrays = {
            name: np.asarray(values, dtype=float)
            for name, values in columns.items()
        }
        time_axis = arrays["t"]

        def finish_time_plot(figure, axis, title: str,
                             ylabel: str, filename: str) -> None:
            axis.set_title(title)
            axis.set_xlabel("time (s)")
            axis.set_xlim(0.0, max(float(time_axis[-1]), 0.02))
            axis.set_ylabel(ylabel)
            axis.set_ylim(bottom=0.0)
            axis.grid(True, color="#b0b0b0", alpha=0.55)
            axis.legend(loc="best")
            figure.tight_layout()
            figure.savefig(
                os.path.join(self.output_dir, filename),
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)

        # 1. 目标速度估计误差 ||uT_actual-uTHat||。
        figure, axis = plt.subplots(figsize=(9.0, 5.2))
        axis.plot(
            time_axis,
            arrays["input_error"],
            color="#1f77b4",
            linewidth=1.5,
            label=r"$\|u_T-\hat{u}_T\|$",
        )
        finish_time_plot(
            figure, axis,
            "Target velocity estimation error",
            "error (m/s)",
            "ut_estimation_error.png",
        )

        # 2. 相对位置估计误差 ||r-rHat||。直接动捕反馈模式下该曲线仅诊断
        # 原论文观测器，不再参与 Kr 的位置反馈。
        figure, axis = plt.subplots(figsize=(9.0, 5.2))
        axis.plot(
            time_axis,
            arrays["localization_error"],
            color="#e07a1f",
            linewidth=1.5,
            label=r"$\|r-\hat{r}\|$",
        )
        finish_time_plot(
            figure, axis,
            "Relative position estimation error",
            "error (m)",
            "relative_position_estimation_error.png",
        )

        # 3. 实际相对向量对期望相对轨迹的控制误差 ||r-rd||。
        figure, axis = plt.subplots(figsize=(9.0, 5.2))
        axis.plot(
            time_axis,
            arrays["tracking_error"],
            color="#3a7d44",
            linewidth=1.5,
            label=r"$\|r-r_d\|$",
        )
        finish_time_plot(
            figure, axis,
            "Relative tracking control error",
            "error (m)",
            "tracking_control_error.png",
        )

        # 4. 两车动捕世界坐标轨迹；起点用圆点，终点用叉号。
        figure, axis = plt.subplots(figsize=(7.4, 7.0))
        target_x, target_y = arrays["target_x"], arrays["target_y"]
        pursuer_x, pursuer_y = arrays["pursuer_x"], arrays["pursuer_y"]
        axis.plot(target_x, target_y, color="#1f77b4", linewidth=1.5,
                  label="target")
        axis.plot(pursuer_x, pursuer_y, color="#e07a1f", linewidth=1.5,
                  label="pursuer")
        axis.scatter(target_x[0], target_y[0], color="#1f77b4", marker="o",
                     s=42, label="target start", zorder=3)
        axis.scatter(pursuer_x[0], pursuer_y[0], color="#e07a1f", marker="o",
                     s=42, label="pursuer start", zorder=3)
        axis.scatter(target_x[-1], target_y[-1], color="#1f77b4", marker="x",
                     s=58, linewidths=1.8, label="target end", zorder=3)
        axis.scatter(pursuer_x[-1], pursuer_y[-1], color="#e07a1f", marker="x",
                     s=58, linewidths=1.8, label="pursuer end", zorder=3)

        arguments = self.metadata.get("arguments", {})
        if arguments.get("target_soft_boundary", False):
            x_min = float(arguments.get("field_x_min", -1.5))
            x_max = float(arguments.get("field_x_max", 1.5))
            y_min = float(arguments.get("field_y_min", -1.5))
            y_max = float(arguments.get("field_y_max", 1.5))
            axis.add_patch(Rectangle(
                (x_min, y_min), x_max - x_min, y_max - y_min,
                fill=False, edgecolor="#4d4d4d", linestyle="--",
                linewidth=1.2, label="field boundary",
            ))

        axis.set_title("World trajectory")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, color="#b0b0b0", alpha=0.55)
        axis.legend(loc="best", fontsize="small")
        figure.tight_layout()
        figure.savefig(
            os.path.join(self.output_dir, "world_trajectory.png"),
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)

        print(
            "[DATA] saved separate plots: ut_estimation_error.png, "
            "relative_position_estimation_error.png, "
            "tracking_control_error.png, world_trajectory.png",
            flush=True,
        )

    def _save_simulation_comparison(self, columns: dict) -> None:
        """保存与纯 Python 仿真同版式的全程 2x2 对比图。"""
        if not columns["t"]:
            return

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arrays = {
            name: np.asarray(values, dtype=float)
            for name, values in columns.items()
        }
        time_axis = arrays["t"]
        figure, axes = plt.subplots(2, 2, figsize=(12, 8))

        axis = axes[0, 0]
        axis.plot(arrays["target_x"], arrays["target_y"], label="target")
        axis.plot(arrays["pursuer_x"], arrays["pursuer_y"], label="pursuer")
        axis.set_title("World trajectory")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True)
        axis.legend(loc="best")

        axis = axes[0, 1]
        axis.plot(time_axis, arrays["input_error"], label=r"$\|u_T-\hat{u}_T\|$")
        axis.plot(time_axis, arrays["localization_error"], label=r"$\|r-\hat{r}\|$")
        axis.plot(time_axis, arrays["tracking_error"], label=r"$\|r-r_d\|$")
        axis.set_title("Errors")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("error")
        axis.grid(True)
        axis.legend(loc="best")

        axis = axes[1, 0]
        axis.plot(
            time_axis,
            np.hypot(arrays["target_mocap_velocity_x"], arrays["target_mocap_velocity_y"]),
            label=r"$\|u_T\|$ mocap",
        )
        axis.plot(
            time_axis,
            np.hypot(arrays["mocap_velocity_x"], arrays["mocap_velocity_y"]),
            label=r"$\|u_O\|$ mocap",
        )
        axis.plot(time_axis, arrays["pursuer_speed"], "--", label=r"$\|u_O\|$ command")
        axis.set_title("Command and position-difference speeds")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("m/s")
        axis.grid(True)
        axis.legend(loc="best")

        axis = axes[1, 1]
        axis.step(time_axis, arrays["active_directions"], where="post",
                  label="active RBF directions")
        axis.step(time_axis, arrays["history_count"], where="post",
                  label="history windows")
        axis.set_title("History-stack excitation")
        axis.set_xlabel("time (s)")
        axis.set_ylabel("count")
        axis.grid(True)
        axis.legend(loc="best")

        for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
            axis.set_xlim(0.0, max(float(time_axis[-1]), 0.02))
        figure.tight_layout()
        for path in (self.comparison_path, self.latest_comparison_path):
            if path:
                parent = os.path.dirname(os.path.abspath(path))
                os.makedirs(parent, exist_ok=True)
                figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(
            "[DATA] saved simulation-style comparison: "
            f"{self.comparison_path}",
            flush=True,
        )

    def run(self) -> None:
        names = [item.name for item in fields(TelemetrySample)]
        columns = {name: [] for name in names}
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(
                os.path.join(self.output_dir, "metadata.json"),
                "w", encoding="utf-8",
            ) as stream:
                json.dump(
                    self.metadata, stream, indent=2, ensure_ascii=False,
                    default=lambda value: (
                        value.item() if isinstance(value, np.generic) else str(value)
                    ),
                )

            with open(self.csv_path, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=names)
                writer.writeheader()
                last_flush = time.monotonic()
                while not (self.stop_event.is_set() and self.samples.empty()):
                    try:
                        sample = self.samples.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    row = asdict(sample)
                    writer.writerow(row)
                    for name in names:
                        columns[name].append(row[name])
                    if time.monotonic() - last_flush >= 1.0:
                        stream.flush()
                        last_flush = time.monotonic()
                stream.flush()

            np.savez_compressed(
                self.npz_path,
                **{name: np.asarray(value) for name, value in columns.items()},
            )
            self._save_summary(columns)
            self._save_recording_export(columns, "errors_and_positions_raw.csv")
            clean, plot_columns, flags, quality_report = filter_recording(
                columns, self.metadata.get("arguments", {})
            )
            save_quality_exports(self.output_dir, columns, clean, flags, quality_report)
            self._save_recording_export(clean)
            print(
                f"[DATA] excluded {quality_report['excluded_samples']}/"
                f"{quality_report['raw_samples']} low-quality samples from cleaned exports; "
                "raw CSV/NPZ retained",
                flush=True,
            )
            if self.save_figures:
                self._save_simulation_comparison(plot_columns)
                save_diagnostics_figures(
                    plot_columns,
                    save_path=self.latest_plot_path,
                    analysis_save_path=self.latest_analysis_path,
                    archive_plot_path=os.path.join(self.output_dir, "tracking_diagnostics.png"),
                    archive_analysis_path=os.path.join(self.output_dir, "tracking_analysis.png"),
                )
                self._save_separate_plots(plot_columns)
            print(f"[DATA] saved experiment data: {self.output_dir}", flush=True)
        except Exception as exc:
            print(f"[DATA] failed to save experiment data: {exc}", flush=True)


class ConsoleTelemetryWorker(threading.Thread):
    """后台打印线程：只取最新诊断帧，绝不阻塞 ROS 实时控制定时器。"""

    def __init__(self, samples: queue.Queue, period: float):
        super().__init__(name="tracking-telemetry", daemon=True)
        self.samples = samples
        self.period = period
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        latest: Optional[TelemetrySample] = None
        last_print = 0.0
        while not self.stop_event.is_set():
            try:
                latest = self.samples.get(timeout=0.2)
                while True:
                    latest = self.samples.get_nowait()
            except queue.Empty:
                pass

            now = time.monotonic()
            if latest is not None and now - last_print >= self.period:
                last_print = now
                mocap_speed = math.hypot(
                    latest.mocap_velocity_x, latest.mocap_velocity_y
                )
                print(
                    "[TRACK] t=%6.2fs  range=%5.3fm  "
                    "e_track=%5.3fm  e_loc=%5.3fm  e_u=%5.3fm/s  "
                    "|uO_cmd|=%5.3fm/s  |vO_mocap|=%5.3fm/s  "
                    "dir=%+6.1fdeg  hist=%02d  active=%02d  gate=%d  chi=%d"
                    % (
                        latest.t,
                        latest.range_value,
                        latest.tracking_error,
                        latest.localization_error,
                        latest.input_error,
                        latest.pursuer_speed,
                        mocap_speed,
                        latest.direction_error_deg,
                        latest.history_count,
                        latest.active_directions,
                        latest.history_update_gated,
                        int(latest.chi),
                    ),
                    flush=True,
                )


def save_diagnostics_figures(
    columns: dict,
    save_path: Optional[str] = None,
    analysis_save_path: Optional[str] = None,
    comparison_save_path: Optional[str] = None,
    archive_plot_path: Optional[str] = None,
    archive_analysis_path: Optional[str] = None,
) -> None:
    """实验结束后从完整记录绘图，所有时间轴固定为 0 到结束时刻。"""
    if not columns["t"]:
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    analysis_figure, analysis_axes = plt.subplots(4, 2, figsize=(13, 13))

    def save_figure(target_figure, path: Optional[str]) -> None:
        if not path:
            return
        try:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            target_figure.savefig(path, dpi=160, bbox_inches="tight")
        except Exception as exc:
            print(f"[PLOT] failed to save {path}: {exc}", flush=True)

    def array(values, name: str) -> np.ndarray:
        return np.asarray(values[name], dtype=float)

    def vector_norm(values, x_name: str, y_name: str) -> np.ndarray:
        return np.hypot(array(values, x_name), array(values, y_name))

    def style_axis(ax, title: str, ylabel: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("time (s)")
        ax.set_xlim(0.0, max(float(columns["t"][-1]), 0.02))
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.7)
        ax.legend(loc="best", fontsize="small")

    def draw_matlab_comparison(values, ts) -> None:
        """保存纯 Python 仿真 comparison 图的全程 2x2 布局。"""
        ax = axes[0, 0]
        ax.clear()
        ax.plot(array(values, "target_x"), array(values, "target_y"), label="target")
        ax.plot(array(values, "pursuer_x"), array(values, "pursuer_y"), label="pursuer")
        ax.set_title("World trajectory")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.7)
        ax.legend(loc="best", fontsize="small")

        ax = axes[0, 1]
        ax.clear()
        ax.plot(ts, array(values, "input_error"), label="||uT_actual-uTHat|| (m/s)")
        ax.plot(ts, array(values, "localization_error"), label="||r-rHat|| (m)")
        ax.plot(ts, array(values, "tracking_error"), label="||r-rd|| (m)")
        style_axis(ax, "Errors", "error")

        ax = axes[1, 0]
        ax.clear()
        ax.plot(
            ts, vector_norm(values, "target_mocap_velocity_x", "target_mocap_velocity_y"),
            label="||uT|| mocap actual",
        )
        ax.plot(
            ts, vector_norm(values, "mocap_velocity_x", "mocap_velocity_y"),
            label="||uO|| mocap actual",
        )
        ax.plot(ts, array(values, "pursuer_speed"), "--", label="||uO_cmd||")
        style_axis(ax, "Command and position-difference speeds", "m/s")

        ax = axes[1, 1]
        ax.clear()
        ax.step(ts, array(values, "active_directions"), where="post",
            label="active RBF directions")
        ax.step(ts, array(values, "history_count"), where="post",
            label="history windows")
        style_axis(ax, "History-stack excitation", "count")

    def draw_engineering_analysis(values, ts) -> None:
        """显示 MATLAB 理想模型与实车之间最可能产生差异的量。"""
        ax = analysis_axes[0, 0]
        ax.clear()
        ax.plot(
            ts, vector_norm(values, "raw_command_world_x", "raw_command_world_y"),
            label="raw controller output",
        )
        ax.plot(ts, array(values, "pursuer_speed"), label="sent command")
        ax.plot(
            ts, vector_norm(values, "mocap_velocity_x", "mocap_velocity_y"),
            label="mocap actual",
        )
        ax.plot(ts, array(values, "command_execution_error"), "--",
                label="||sent-actual||")
        style_axis(ax, "Pursuer command execution", "speed / error (m/s)")

        ax = analysis_axes[0, 1]
        ax.clear()
        ax.plot(
            ts, vector_norm(values, "target_command_world_x", "target_command_world_y"),
            label="target command",
        )
        ax.plot(
            ts, vector_norm(values, "target_mocap_velocity_x", "target_mocap_velocity_y"),
            label="target mocap actual",
        )
        ax.plot(
            ts, vector_norm(values, "target_velocity_estimate_x", "target_velocity_estimate_y"),
            label="target estimate uTHat",
        )
        ax.plot(ts, array(values, "target_command_execution_error"), "--",
                label="||target cmd-actual||")
        style_axis(ax, "Target model versus real motion", "speed / error (m/s)")

        ax = analysis_axes[1, 0]
        ax.clear()
        components = (
            ("control_target_estimate_x", "control_target_estimate_y", "uTHat"),
            ("control_relative_velocity_x", "control_relative_velocity_y", "deltaVHat"),
            ("control_trajectory_x", "control_trajectory_y", "-rdDot"),
            ("control_feedback_x", "control_feedback_y", "Kr(xiHat_r-rd)"),
        )
        for x_name, y_name, label in components:
            ax.plot(ts, vector_norm(values, x_name, y_name), label=label)
        style_axis(ax, "Four components of raw pursuer command", "component norm (m/s)")

        ax = analysis_axes[1, 1]
        ax.clear()
        ax.plot(ts, array(values, "range_value"), label="||r|| actual")
        ax.plot(
            ts, vector_norm(values, "relative_estimate_x", "relative_estimate_y"),
            label="||rHat||",
        )
        ax.plot(
            ts, vector_norm(values, "desired_relative_x", "desired_relative_y"),
            "--", label="||rd||",
        )
        style_axis(ax, "Relative state: actual, estimate, desired", "range (m)")

        ax = analysis_axes[2, 0]
        ax.clear()
        ax.plot(ts, array(values, "tracking_error"), label="tracking ||r-rd||")
        ax.plot(ts, array(values, "localization_error"), label="localization ||r-rHat||")
        ax.plot(ts, array(values, "input_error"), label="input ||uT_actual-uTHat||")
        ax.plot(ts, array(values, "command_execution_error"), "--",
                label="pursuer execution ||cmd-actual||")
        style_axis(ax, "Error source comparison", "error (m or m/s)")

        ax = analysis_axes[2, 1]
        ax.clear()
        floor = 1.0e-12
        ax.semilogy(ts, np.maximum(array(values, "theta_hat_norm"), floor), label="||thetaHat||")
        ax.semilogy(ts, np.maximum(array(values, "xi_position_norm"), floor), label="||xiHat_r||")
        ax.semilogy(ts, np.maximum(array(values, "xi_velocity_norm"), floor), label="||xiHat_v||")
        ax.semilogy(ts, np.maximum(array(values, "po_condition"), 1.0), label="cond(Po)")
        style_axis(ax, "Estimator state growth", "norm / condition (log scale)")

        ax = analysis_axes[3, 0]
        ax.clear()
        ax.step(ts, array(values, "history_count"), where="post", label="history windows")
        ax.step(ts, array(values, "active_directions"), where="post", label="active directions")
        ax.step(ts, array(values, "history_update_gated"), where="post", linestyle="--",
                label="condition gate")
        ax2 = ax.twinx()
        min_eigenvalue = array(values, "history_min_active_eigenvalue")
        max_eigenvalue = array(values, "history_max_eigenvalue")
        ax2.semilogy(ts, np.maximum(min_eigenvalue, 1.0e-14), "--",
                     color="tab:green", label="min active eigenvalue")
        ax2.semilogy(ts, np.maximum(max_eigenvalue, 1.0e-14), ":",
                     color="tab:red", label="max eigenvalue")
        ax.set_title("History-stack excitation spectrum")
        ax.set_xlabel("time (s)")
        ax.set_xlim(0.0, max(float(columns["t"][-1]), 0.02))
        ax.set_ylabel("count")
        ax2.set_ylabel("eigenvalue (log scale)")
        ax.grid(True, alpha=0.7)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize="small")

        ax = analysis_axes[3, 1]
        ax.clear()
        ax.plot(ts, 1000.0 * array(values, "control_dt_raw"), label="raw control dt")
        ax.plot(ts, 1000.0 * array(values, "control_dt_used"), label="dt used by RK4")
        ax.plot(ts, 1000.0 * array(values, "target_pose_age"), "--", label="target pose age")
        ax.plot(ts, 1000.0 * array(values, "pursuer_pose_age"), "--", label="pursuer pose age")
        ax.plot(ts, 1000.0 * array(values, "target_velocity_sample_span"), ":",
                label="target velocity fit span")
        ax.plot(ts, 1000.0 * array(values, "pursuer_velocity_sample_span"), ":",
                label="pursuer velocity fit span")
        style_axis(ax, "Timing and mocap freshness", "time (ms)")

    if columns["t"]:
        values = columns
        ts = array(values, "t")
        draw_matlab_comparison(values, ts)
        draw_engineering_analysis(values, ts)
        figure.tight_layout()
        analysis_figure.tight_layout()
        for target_figure, path in (
            (figure, save_path),
            (figure, comparison_save_path),
            (analysis_figure, analysis_save_path),
            (figure, archive_plot_path),
            (analysis_figure, archive_analysis_path),
        ):
            save_figure(target_figure, path)
        print(f"[PLOT] saved diagnostics figure: {save_path}", flush=True)
        print(f"[PLOT] saved analysis figure: {analysis_save_path}", flush=True)
        if archive_plot_path:
            print(f"[PLOT] archived diagnostics figure: {archive_plot_path}", flush=True)
        if archive_analysis_path:
            print(f"[PLOT] archived analysis figure: {archive_analysis_path}", flush=True)
    plt.close(figure)
    plt.close(analysis_figure)


@dataclass
class Parameters:
    """论文/Matlab 参数集合；矩阵维度由 ``n=2`` 和 ``q`` 自动派生。"""

    # ---------- 采样与期望相对轨迹 r_d(t) ----------
    dt: float = 0.02          # 控制器名义采样周期；ROS 中由 1/rate 覆盖
    delta_t: float = 1.0      # 每个有限积分数据窗的长度 DeltaT（秒）
    rd: float = 0.5           # 期望相对圆半径 Rd（ROS 中由命令行覆盖）
    wd: float = 0.5           # 期望相对圆角速度 wd（rad/s）
    phi0: float = 2.0 * math.pi / 3.0  # r_d 在 t=0 时的世界系初相位
    max_linear_speed: float = 0.25     # 昨天实车实验使用的速度模长上限
    tracking_gain: float = 0.10        # 相对轨迹反馈增益，Kr=tracking_gain*I

    # ---------- RBF 目标输入模型：论文式 (4) ----------
    q: int = 30               # 单个输出通道的 RBF 节点数量
    sigma: float = 0.52       # 高斯基函数宽度

    # ---------- 当前窗/历史栈辨识器：论文式 (11)-(24) ----------
    nh: int = 30              # 最多保留的完整历史窗口数量
    max_history_execution_error: float = 0.0  # 历史窗允许的最大执行误差 RMS；0 禁用
    max_history_target_execution_error: float = 0.0  # 目标侧命令-动捕执行误差门限；0 禁用
    history_forgetting_factor: float = 1.0  # 每接受一个窗口对历史栈应用的指数遗忘；1 禁用
    identifier_use_relative_velocity: bool = False  # 窗口回归目标是否含动捕相对速度
    gamma: float = 1.0e-3     # 昨天实车实验的激活特征值阈值
    ac: float = 1.0e-5        # 当前窗口可参与在线更新的激活阈值
    valid_tol: float = 100.0 * np.finfo(float).eps  # 完整窗口有效性阈值
    kc: float = 2.0           # 当前窗口并行学习增益
    kh: float = 1.2           # 历史栈并行学习增益
    max_history_condition: float = 1.0e3  # 昨天实车实验的条件数门限
    sigma_theta: float = 0.003  # 未激活参数方向上的泄漏/正则化系数

    # ---------- 目标输入滤波、Riccati 方程和相对状态观测器 ----------
    af: float = 5.0           # uT_hat 一阶滤波器带宽，论文式 (46)
    lambda_s: float = 0.10    # Riccati 方程中的稳定化参数（本次实车测试值）
    ro: float = 0.02          # 观测噪声权重 R_o = ro * I
    p_floor: float = 1.0e-9   # P_o 投影为正定矩阵时的最小特征值
    mocap_position_observer_gain: float = 0.0
    mocap_relative_velocity_observer_gain: float = 0.0

    @property
    def n(self) -> int:
        """二维平面系统状态维度。"""
        return 2

    @property
    def nq(self) -> int:
        """RBF 参数向量总维度 n*q = 60。"""
        return self.n * self.q

    @property
    def nw(self) -> int:
        """一个长度为 DeltaT 的窗口应包含的离散样本数。"""
        count = int(round(self.delta_t / self.dt))
        if count <= 0 or abs(count * self.dt - self.delta_t) > 1.0e-9:
            raise ValueError("delta_t/dt must be an integer")
        return count


class HistoryStack:
    """有限历史栈及谱投影，对应论文式 (21)-(23)。

    ``windows[i]`` 保存 W_i，``vectors[i]`` 保存 Z_i。历史栈满后并非简单
    FIFO，而是尝试替换任一旧窗口，保留能让聚合矩阵谱更优的组合。
    ``pi`` 是可辨识子空间投影，``theta_gamma`` 是该子空间内的参数解。
    """

    def __init__(self, p: Parameters):
        self.p = p
        self.windows: List[np.ndarray] = []
        self.vectors: List[np.ndarray] = []
        self.pi = np.zeros((p.nq, p.nq), dtype=float)
        self.theta_gamma = np.zeros(p.nq, dtype=float)
        # 保存最近一次聚合矩阵的完整谱，供诊断 gamma 是否在放大噪声。
        self.eigenvalues = np.zeros(p.nq, dtype=float)
        self.condition = 0.0
        self.update_gated = False

    @staticmethod
    def spectrum(matrix: np.ndarray) -> np.ndarray:
        """返回对称矩阵从小到大排列的实特征值，并清除数值零。"""
        values = np.linalg.eigvalsh((matrix + matrix.T) * 0.5)
        values[np.abs(values) < 100.0 * np.finfo(float).eps * max(1.0, np.linalg.norm(matrix, 2))] = 0.0
        return np.sort(np.real(values))

    @staticmethod
    def lex_greater(a: np.ndarray, b: np.ndarray) -> bool:
        """按字典序比较两个谱：优先提高最弱的特征方向。"""
        for av, bv in zip(a, b):
            scale = max(1.0, abs(float(av)), abs(float(bv)))
            if av > bv + 1.0e-12 * scale:
                return True
            if av < bv - 1.0e-12 * scale:
                return False
        return False

    def add(self, w_new: np.ndarray, z_new: np.ndarray) -> None:
        """加入有效窗口；历史栈满时执行论文规定的谱改进替换。"""
        if len(self.windows) < self.p.nh:
            self.windows.append(w_new.copy())
            self.vectors.append(z_new.copy())
        else:
            aggregate = np.sum(self.windows, axis=0)
            best = self.spectrum(aggregate)
            replacement = -1
            for index in range(self.p.nh):
                candidate = aggregate - self.windows[index] + w_new
                candidate_spectrum = self.spectrum(candidate)
                if self.lex_greater(candidate_spectrum, best):
                    best = candidate_spectrum
                    replacement = index
            if replacement >= 0:
                self.windows[replacement] = w_new.copy()
                self.vectors[replacement] = z_new.copy()

        if self.p.history_forgetting_factor < 1.0:
            factor = self.p.history_forgetting_factor
            for index in range(len(self.windows)):
                self.windows[index] *= factor
                self.vectors[index] *= factor

        self._recompute_projection()

    def _recompute_projection(self) -> None:
        """根据全部保留窗口重新计算 Pi 和 theta_Gamma。

        只有特征值 >= gamma 的方向才求逆。gamma 在无噪声 MATLAB 中很小，
        实车噪声可能在这里被放大，是分析发散时需要重点检查的位置。
        """
        if not self.windows:
            self.pi.fill(0.0)
            self.theta_gamma.fill(0.0)
            self.eigenvalues.fill(0.0)
            self.condition = 0.0
            self.update_gated = False
            return

        wh = np.sum(self.windows, axis=0)
        wh = (wh + wh.T) * 0.5
        zh = np.sum(self.vectors, axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(wh)
        self.eigenvalues = np.real(eigenvalues).copy()
        active = np.real(eigenvalues) >= self.p.gamma
        if not np.any(active):
            self.pi.fill(0.0)
            self.theta_gamma.fill(0.0)
            self.condition = 0.0
            self.update_gated = False
            return

        qa = np.real(eigenvectors[:, active])
        la = np.real(eigenvalues[active])
        self.condition = float(np.max(la) / np.min(la))
        self.update_gated = (
            self.p.max_history_condition > 0.0
            and self.condition > self.p.max_history_condition
        )
        if self.update_gated:
            # 保留上一次可用的 Pi/thetaGamma。窗口仍保留，后续新增数据可能
            # 改善聚合谱并使门控自动解除。
            return
        self.pi = qa @ qa.T
        self.pi = (self.pi + self.pi.T) * 0.5
        self.theta_gamma = qa @ ((qa.T @ zh) / la)


class CurrentWindow:
    """当前有限数据窗的在线积分统计量。

    只累计构造 W、Z 所需的充分统计量，不保存窗口内每一个完整大矩阵，
    从而降低控制循环的内存和计算负担。
    """

    def __init__(self, b0: np.ndarray, p: Parameters):
        self.p = p
        self.b0 = b0.copy()
        self.g = np.zeros((2, p.nq), dtype=float)
        self.u = np.zeros(2, dtype=float)
        self.a = 0.0
        self.cf = np.zeros(p.nq, dtype=float)
        self.ch = 0.0
        self.sff = np.zeros((p.nq, p.nq), dtype=float)
        self.sfh = np.zeros(p.nq, dtype=float)
        self.execution_error_integral = 0.0
        self.target_execution_error_integral = 0.0
        self.duration = 0.0
        self.count = 0

    def execution_error_rms(self) -> float:
        """返回本窗指令与动捕实际速度之差的时间加权 RMS。"""
        if self.duration <= 0.0:
            return 0.0
        return math.sqrt(self.execution_error_integral / self.duration)

    def target_execution_error_rms(self) -> float:
        """返回本窗目标指令与目标动捕速度之差的时间加权 RMS。"""
        if self.duration <= 0.0:
            return 0.0
        return math.sqrt(self.target_execution_error_integral / self.duration)

    def statistics(self, threshold: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """由累计量构造 (W, Z, chi)；激励不足时返回零并令 chi=0。"""
        if self.a < threshold:
            return (
                np.zeros((self.p.nq, self.p.nq), dtype=float),
                np.zeros(self.p.nq, dtype=float),
                0.0,
            )
        w = self.sff - np.outer(self.cf, self.cf) / self.a
        z = self.sfh - self.cf * self.ch / self.a
        return (w + w.T) * 0.5, z, 1.0

    def accumulate(self, b: np.ndarray, pb: np.ndarray, phi: np.ndarray,
                   u_ident: np.ndarray, u_o_actual: np.ndarray,
                   u_o_command: np.ndarray, dt: float,
                   target_execution_error: Optional[np.ndarray] = None) -> None:
        """用左矩形法累计一个样本，与 MATLAB 循环的积分顺序一致。

        ``u_ident`` 是当前窗回归目标（默认等于 pursuer 实际速度；启用相对
        速度修正时为 Pb*(u_o+dr/dt)）。``u_o_actual`` 与 ``u_o_command`` 只用于
        计算底盘执行质量，不进入回归方程。``target_execution_error`` 用于
        目标侧窗口质量门，同样不进入回归方程。参数 ``b`` 保留在接口中用于
        表明数据来源，实际公式使用已计算好的投影矩阵 ``pb``。
        """
        f = pb @ self.g
        h = pb @ self.u
        v = pb @ self.b0
        self.a += dt * float(v.T @ v)
        self.cf += dt * (v @ f)
        self.ch += dt * float(v @ h)
        self.sff += dt * (f.T @ f)
        self.sfh += dt * (f.T @ h)
        self.g += dt * phi
        self.u += dt * u_ident
        execution_error = np.asarray(u_o_command) - np.asarray(u_o_actual)
        self.execution_error_integral += dt * float(execution_error @ execution_error)
        if target_execution_error is not None:
            error = np.asarray(target_execution_error, dtype=float)
            if np.all(np.isfinite(error)):
                self.target_execution_error_integral += dt * float(
                    error @ error
                )
        self.duration += dt
        self.count += 1


class BearingTrackingController:
    """MATLAB/论文核心算法的数值实现。

    控制器输入：
      * ``b``：目标相对 pursuer 的单位方位，目标侧唯一观测量；
      * ``u_o_actual``：pursuer 自身的实际世界速度估计（已知自运动）。

    控制器内部不接收真实距离 ``rho``、真实相对位置 ``r`` 或真实目标速度。
    输出 ``uO_cmd`` 是世界坐标系速度，ROS 节点在发布前再转成车体坐标系。
    """

    def __init__(self, p: Parameters, b0: np.ndarray,
                 initial_relative_position: Optional[np.ndarray] = None):
        # I、J 分别是二维单位阵和逆时针旋转 90 度矩阵。
        self.p = p
        self.identity2 = np.eye(2)
        self.identity_nq = np.eye(p.nq)
        self.j = np.array([[0.0, -1.0], [1.0, 0.0]])
        # q 个 RBF 中心均匀放置在单位圆上，输入 b 本身也是单位向量。
        angles = 2.0 * math.pi * np.arange(p.q) / p.q
        self.centers = np.vstack((np.cos(angles), np.sin(angles)))

        # 与 MATLAB 一致：所有未知量从零开始，P_o(0)=0.3 I。
        self.theta_hat = np.zeros(p.nq, dtype=float)  # RBF 参数估计，60x1
        self.u_t_hat = np.zeros(2, dtype=float)       # 目标输入估计，2x1
        self.xi_hat = np.zeros(4, dtype=float)        # 增广相对状态估计，4x1
        if initial_relative_position is not None:
            self.xi_hat[0:2] = np.asarray(
                initial_relative_position, dtype=float
            )
        self.po = 0.30 * np.eye(4, dtype=float)       # Riccati 矩阵，4x4
        self.history = HistoryStack(p)
        self.window = CurrentWindow(b0, p)
        self.last_history_window_execution_error = 0.0
        self.last_history_target_execution_error = 0.0
        self.last_history_window_accepted = 0

    def rbf_regressor(self, b: np.ndarray) -> np.ndarray:
        """计算论文式 (4) 的块对角 RBF 回归矩阵 Phi(b)，尺寸 2x60。"""
        error = self.centers - b.reshape(2, 1)
        phi = np.exp(-np.sum(error * error, axis=0) / (2.0 * self.p.sigma ** 2))
        return np.kron(self.identity2, phi.reshape(1, -1))

    def desired_trajectory(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        """期望相对圆及其导数。

        r_d = Rd[cos(a), sin(a)]，a=phi0+wd*t；这里的圆位于相对坐标，
        并不要求 pursuer 在世界坐标中画标准圆。
        """
        angle = self.p.phi0 + self.p.wd * t
        rd = self.p.rd * np.array([math.cos(angle), math.sin(angle)])
        rd_dot = self.p.rd * self.p.wd * np.array([-math.sin(angle), math.cos(angle)])
        return rd, rd_dot

    def signals(self, t: float, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """由方位构造本采样周期共用的 Pb、Phi、rd、rdDot。"""
        # Pb=I-bb^T：把向量投影到与视线 b 正交的方向。
        pb = self.identity2 - np.outer(b, b)
        phi = self.rbf_regressor(b)
        rd, rd_dot = self.desired_trajectory(t)
        return pb, phi, rd, rd_dot

    def step(self, t: float, b: np.ndarray, dt: float,
             u_o_actual: np.ndarray,
             relative_position_feedback: Optional[np.ndarray] = None,
             u_ident: Optional[np.ndarray] = None,
             relative_velocity_feedback: Optional[np.ndarray] = None,
             target_execution_error: Optional[np.ndarray] = None,
             ) -> Tuple[np.ndarray, dict]:
        """推进一次辨识器/观测器，并返回世界系 pursuer 指令。

        ``u_o_actual`` 来自动捕状态拟合。理想 MATLAB 模型中它与指令完全相等；
        实车中用实际速度可避免把底盘限速、滞后误认为目标运动。

        ``relative_position_feedback`` 为 None 时，反馈项使用论文观测器的
        ``rHat=xiHat[0:2]``；提供动捕 ``r=pT-pO`` 时，仅反馈项改用该直接位置。
        观测器仍继续独立运行，因此可以在日志中比较 r 与 rHat。
        """
        p = self.p
        b = b / max(float(np.linalg.norm(b)), 1.0e-12)
        pb, phi, rd, rd_dot = self.signals(t, b)
        wc, zc, chi = self.window.statistics(p.ac)
        identifier_terms = self._identifier_terms(
            phi, self.theta_hat, wc, zc, chi
        )

        # 论文式 (63)：
        # uO = uT_hat + xiHat_v - rdDot + Kr(xiHat_r-rd)，
        # 其中 Kr=tracking_gain*I，可在 YAML 中调整。
        # 分别保留四项和限幅前结果，用于判断究竟是哪一项先异常增大。
        control_target_estimate = self.u_t_hat.copy()
        # xiHat 后两维是 deltaVHat=uT-uTHat 的估计，不是底盘实际速度。
        control_relative_velocity = self.xi_hat[2:4].copy()
        control_trajectory = -rd_dot.copy()
        feedback_uses_mocap = relative_position_feedback is not None
        feedback_position = (
            np.asarray(relative_position_feedback, dtype=float).copy()
            if feedback_uses_mocap else self.xi_hat[0:2].copy()
        )
        control_feedback = p.tracking_gain * (feedback_position - rd)
        u_o_raw = np.asarray(
            control_target_estimate
            + control_relative_velocity
            + control_trajectory
            + control_feedback,
            dtype=float,
        )
        # 这里只得到“要求底盘执行”的指令；实际执行速度另由动捕估计。
        u_o_cmd = limit_vector_norm(u_o_raw, p.max_linear_speed)
        u_o_actual = np.asarray(u_o_actual, dtype=float)
        u_ident = (
            u_o_actual if u_ident is None
            else np.asarray(u_ident, dtype=float)
        )

        # RK4 同时积分 thetaHat、uTHat、xiHat、Po。一个采样周期内把方位、
        # 当前窗统计量和实测自速度视为零阶保持量，与离散实时控制相对应。
        state = (self.theta_hat.copy(), self.u_t_hat.copy(),
                 self.xi_hat.copy(), self.po.copy())
        k1 = self._derivatives(
            t, b, *state, wc, zc, chi, u_o_actual,
            relative_position_feedback, relative_velocity_feedback,
        )
        k2_state = self._add_state(state, k1, 0.5 * dt)
        k2 = self._derivatives(
            t + 0.5 * dt, b, *k2_state, wc, zc, chi, u_o_actual,
            relative_position_feedback, relative_velocity_feedback,
        )
        k3_state = self._add_state(state, k2, 0.5 * dt)
        k3 = self._derivatives(
            t + 0.5 * dt, b, *k3_state, wc, zc, chi, u_o_actual,
            relative_position_feedback, relative_velocity_feedback,
        )
        k4_state = self._add_state(state, k3, dt)
        k4 = self._derivatives(
            t + dt, b, *k4_state, wc, zc, chi, u_o_actual,
            relative_position_feedback, relative_velocity_feedback,
        )
        increment = tuple(
            dt * (a + 2.0 * b_ + 2.0 * c + d) / 6.0
            for a, b_, c, d in zip(k1, k2, k3, k4)
        )
        updated = self._add_state(state, increment, 1.0)
        self.theta_hat, self.u_t_hat, self.xi_hat, self.po = updated
        self.po = self.project_spd(self.po, p.p_floor)

        # 用当前样本完成有限窗口积分。达到 Nw 个样本后经质量门加入历史栈。
        # 回归目标默认是 Pb*uO；启用相对速度修正时拟合 Pb*(uO+dr/dt)=Pb*uT。
        # 执行质量始终按命令与实际速度之差计算，不随回归目标改变。
        self.window.accumulate(
            b, pb, phi, u_ident, u_o_actual, u_o_cmd, dt,
            target_execution_error=target_execution_error,
        )
        if self.window.count >= p.nw:
            w_new, z_new, valid = self.window.statistics(p.valid_tol)
            self.last_history_window_execution_error = self.window.execution_error_rms()
            self.last_history_target_execution_error = (
                self.window.target_execution_error_rms()
            )
            self.last_history_window_accepted = int(
                valid > 0.0 and (
                    p.max_history_execution_error <= 0.0
                    or self.last_history_window_execution_error
                    <= p.max_history_execution_error
                ) and (
                    p.max_history_target_execution_error <= 0.0
                    or self.last_history_target_execution_error
                    <= p.max_history_target_execution_error
                )
            )
            if self.last_history_window_accepted:
                self.history.add(w_new, z_new)
            self.window = CurrentWindow(b, p)

        # 谱和 Riccati 数值状态只用于离线判断实车噪声是否被小特征值放大。
        active_eigenvalues = self.history.eigenvalues[
            self.history.eigenvalues >= p.gamma
        ]
        history_min_active = (
            float(np.min(active_eigenvalues)) if active_eigenvalues.size else 0.0
        )
        history_max = (
            float(np.max(self.history.eigenvalues))
            if self.history.eigenvalues.size else 0.0
        )
        po_eigenvalues = np.linalg.eigvalsh(self.po)
        po_min = float(np.min(po_eigenvalues))
        po_max = float(np.max(po_eigenvalues))

        # 以下字典只供 ROS 层打印/画图；真实 r、rho 不在控制器内部出现。
        diagnostics = {
            "bearing": b.copy(),
            "relative_estimate": self.xi_hat[0:2].copy(),
            "desired_relative": rd.copy(),
            "target_velocity_estimate": self.u_t_hat.copy(),
            "world_velocity": u_o_cmd.copy(),
            "raw_world_velocity": u_o_raw.copy(),
            "actual_velocity_used": u_o_actual.copy(),
            "control_target_estimate": control_target_estimate,
            "control_relative_velocity": control_relative_velocity,
            "control_trajectory": control_trajectory,
            "control_feedback": control_feedback,
            "control_feedback_position": feedback_position,
            "control_feedback_uses_mocap": int(feedback_uses_mocap),
            "theta_hat_norm": float(np.linalg.norm(self.theta_hat)),
            "identifier_current_theta_dot_norm": float(np.linalg.norm(
                identifier_terms["current_theta_dot"]
            )),
            "identifier_history_theta_dot_norm": float(np.linalg.norm(
                identifier_terms["history_theta_dot"]
            )),
            "identifier_leak_theta_dot_norm": float(np.linalg.norm(
                identifier_terms["leak_theta_dot"]
            )),
            "identifier_current_output_rate": identifier_terms[
                "current_output_rate"
            ].copy(),
            "identifier_history_output_rate": identifier_terms[
                "history_output_rate"
            ].copy(),
            "identifier_leak_output_rate": identifier_terms[
                "leak_output_rate"
            ].copy(),
            "identifier_total_output_rate": identifier_terms[
                "total_output_rate"
            ].copy(),
            "xi_position_norm": float(np.linalg.norm(self.xi_hat[0:2])),
            "xi_velocity_norm": float(np.linalg.norm(self.xi_hat[2:4])),
            "po_min_eigenvalue": po_min,
            "po_max_eigenvalue": po_max,
            "po_condition": po_max / max(po_min, p.p_floor),
            "history_count": len(self.history.windows),
            "active_directions": int(np.linalg.matrix_rank(self.history.pi, tol=1.0e-8)),
            "history_min_active_eigenvalue": history_min_active,
            "history_max_eigenvalue": history_max,
            "history_condition": self.history.condition,
            "history_update_gated": int(self.history.update_gated),
            "current_window_count": self.window.count,
            "current_window_excitation": float(self.window.a),
            "chi": chi,
            "history_last_window_execution_error": (
                self.last_history_window_execution_error
            ),
            "history_last_window_accepted": self.last_history_window_accepted,
        }
        return u_o_cmd, diagnostics

    def _derivatives(self, t: float, b: np.ndarray, theta_hat: np.ndarray,
                     u_t_hat: np.ndarray, xi_hat: np.ndarray, po: np.ndarray,
                     wc: np.ndarray, zc: np.ndarray, chi: float,
                     u_o_actual: np.ndarray,
                     relative_position_feedback: Optional[np.ndarray],
                     relative_velocity_feedback: Optional[np.ndarray]):
        """论文式 (24)、(46)、(50)、(51) 的连续时间右端项。"""
        p = self.p
        pb, phi, rd, rd_dot = self.signals(t, b)
        # 参数估计器：当前窗口项 + 历史栈项 + 未激活方向泄漏项。
        theta_dot = self._identifier_terms(
            phi, theta_hat, wc, zc, chi
        )["total_theta_dot"]
        # RBF 输出经一阶滤波形成平滑的目标速度估计 uT_hat。
        u_t_hat_dot = p.af * (phi @ theta_hat - u_t_hat)

        # 方位只能约束相对位置在视线正交方向上的投影，因此 y_o=Pb*r=0。
        co = np.hstack((pb, np.zeros((2, 2), dtype=float)))
        ro_inv = (1.0 / p.ro) * self.identity2
        lo = po @ co.T @ ro_inv
        # 相对运动输入为 uT_hat-uO_actual。这里明确不用 uO_cmd 冒充实际速度。
        go = np.concatenate((u_t_hat - u_o_actual, np.zeros(2, dtype=float)))
        po_dot = (
            self._ao() @ po + po @ self._ao().T
            + p.lambda_s * po - po @ co.T @ ro_inv @ co @ po
        )
        xi_dot = self._ao() @ xi_hat + go + lo @ (-co @ xi_hat)
        if (relative_position_feedback is not None
                and p.mocap_position_observer_gain > 0.0):
            xi_dot[0:2] += p.mocap_position_observer_gain * (
                np.asarray(relative_position_feedback, dtype=float) - xi_hat[0:2]
            )
        if (relative_velocity_feedback is not None
                and p.mocap_relative_velocity_observer_gain > 0.0):
            predicted_relative_velocity = (
                u_t_hat - u_o_actual + xi_hat[2:4]
            )
            xi_dot[2:4] += p.mocap_relative_velocity_observer_gain * (
                np.asarray(relative_velocity_feedback, dtype=float)
                - predicted_relative_velocity
            )
        return theta_dot, u_t_hat_dot, xi_dot, po_dot

    def _identifier_terms(self, phi: np.ndarray, theta_hat: np.ndarray,
                          wc: np.ndarray, zc: np.ndarray, chi: float) -> dict:
        """分解参数更新项，供诊断各项对目标速度估计的驱动。"""
        p = self.p
        current_theta_dot = p.kc * chi * (zc - wc @ theta_hat)
        history_theta_dot = p.kh * (
            self.history.theta_gamma - self.history.pi @ theta_hat
        )
        leak_theta_dot = -p.sigma_theta * (
            self.identity_nq - self.history.pi
        ) @ theta_hat
        total_theta_dot = (
            current_theta_dot + history_theta_dot + leak_theta_dot
        )
        return {
            "current_theta_dot": current_theta_dot,
            "history_theta_dot": history_theta_dot,
            "leak_theta_dot": leak_theta_dot,
            "total_theta_dot": total_theta_dot,
            "current_output_rate": phi @ current_theta_dot,
            "history_output_rate": phi @ history_theta_dot,
            "leak_output_rate": phi @ leak_theta_dot,
            "total_output_rate": phi @ total_theta_dot,
        }

    @staticmethod
    def _add_state(state, derivative, scale: float):
        """RK4 使用的通用状态线性组合。"""
        return tuple(x + scale * dx for x, dx in zip(state, derivative))

    def _ao(self) -> np.ndarray:
        """构造 4x4 增广相对运动系统矩阵 A_o。"""
        ao = np.zeros((4, 4), dtype=float)
        ao[0:2, 2:4] = self.identity2
        return ao

    @staticmethod
    def project_spd(matrix: np.ndarray, floor_value: float) -> np.ndarray:
        """把数值积分后的 P_o 投影回对称正定矩阵集合。"""
        matrix = (matrix + matrix.T) * 0.5
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(np.real(values), floor_value)
        projected = vectors @ np.diag(values) @ vectors.T
        return (projected + projected.T) * 0.5


class EvasiveTargetController:
    """MATLAB ``targetInput`` 对应的目标车运动规律。

    u_T(b) = v_d d + v_e b + v_j J b

    ``v_d d`` 是固定世界方向分量；``v_e b`` 沿视线远离 pursuer；
    ``v_j Jb`` 是视线切向分量。pursuer 控制器不知道该公式，只能通过方位
    历史在线估计 target 的输入。
    """

    def __init__(self, vd: float, ve: float, vj: float,
                 direction_deg: float, max_speed: float,
                 soft_boundary: bool, field_x_min: float,
                 field_x_max: float, field_y_min: float,
                 field_y_max: float, boundary_margin: float,
                 boundary_speed: float):
        angle = math.radians(direction_deg)
        self.vd = vd
        self.ve = ve
        self.vj = vj
        self.d = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        self.j = np.array([[0.0, -1.0], [1.0, 0.0]])
        self.max_speed = max_speed
        self.soft_boundary = soft_boundary
        self.field_min = np.array([field_x_min, field_y_min], dtype=float)
        self.field_max = np.array([field_x_max, field_y_max], dtype=float)
        self.boundary_margin = boundary_margin
        self.boundary_speed = boundary_speed

    def nominal_command(self, bearing: np.ndarray) -> np.ndarray:
        """原始 MATLAB 目标运动律，不包含场地边界修正。"""
        b = bearing / max(float(np.linalg.norm(bearing)), 1.0e-12)
        return self.vd * self.d + self.ve * b + self.vj * (self.j @ b)

    @staticmethod
    def _smoothstep(value: float) -> float:
        """把 [0,1] 平滑映射到 [0,1]，两端斜率均为零。"""
        value = min(max(value, 0.0), 1.0)
        return value * value * (3.0 - 2.0 * value)

    def boundary_command(self, position: np.ndarray) -> np.ndarray:
        """计算世界坐标软边界回推速度。

        距围栏超过 ``boundary_margin`` 时为零；进入缓冲带后使用 smoothstep
        连续增大，到围栏处达到 ``boundary_speed``。这不是位置硬截断。
        """
        correction = np.zeros(2, dtype=float)
        if not self.soft_boundary or self.boundary_speed <= 0.0:
            return correction

        for axis in range(2):
            lower_start = self.field_min[axis] + self.boundary_margin
            upper_start = self.field_max[axis] - self.boundary_margin
            if position[axis] < lower_start:
                depth = (lower_start - position[axis]) / self.boundary_margin
                correction[axis] += self.boundary_speed * self._smoothstep(depth)
            elif position[axis] > upper_start:
                depth = (position[axis] - upper_start) / self.boundary_margin
                correction[axis] -= self.boundary_speed * self._smoothstep(depth)
        return correction

    def command_parts(self, bearing: np.ndarray,
                      position: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回最终指令、原始运动律和软边界修正，供控制与诊断共用。"""
        nominal = self.nominal_command(bearing)
        boundary = self.boundary_command(position)
        command = limit_vector_norm(nominal + boundary, self.max_speed)
        return command, nominal, boundary

    def command(self, bearing: np.ndarray, position: np.ndarray) -> np.ndarray:
        """根据方位和 target 世界位置计算最终世界系速度。"""
        command, _, _ = self.command_parts(bearing, position)
        return command


class BearingTrackingNode:
    """ROS 1 外层：收发话题、估计 pursuer 自速度并调用核心控制器。

    默认数据流：
      /robot04/pose -> target 动捕位姿
      /robot03/pose -> pursuer 动捕位姿和实际速度拟合
    世界系算法输出 -> yaw 旋转 -> /robotXX/cmd_vel 车体系指令
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # 把命令行中的实验参数写入论文参数结构。这里的 dt 是名义周期，
        # 实际每次积分仍使用定时器测得并限幅后的 dt。
        self.p = Parameters(
            dt=1.0 / args.rate,
            delta_t=args.window,
            rd=args.relative_radius,
            wd=args.relative_angular_speed,
            phi0=math.radians(args.relative_initial_phase_deg),
            max_linear_speed=args.max_speed,
            tracking_gain=args.tracking_gain,
            q=args.rbf_count,
            sigma=args.rbf_sigma,
            nh=args.history_size,
            max_history_execution_error=args.max_history_execution_error,
            max_history_target_execution_error=(
                args.max_history_target_execution_error
            ),
            history_forgetting_factor=args.history_forgetting_factor,
            identifier_use_relative_velocity=args.identifier_use_relative_velocity,
            gamma=args.gamma,
            ac=args.current_window_threshold,
            valid_tol=args.window_valid_tolerance,
            kc=args.current_window_gain,
            kh=args.history_gain,
            max_history_condition=args.max_history_condition,
            sigma_theta=args.sigma_theta,
            af=args.target_estimate_filter_bandwidth,
            lambda_s=args.riccati_lambda,
            ro=args.observer_noise_weight,
            p_floor=args.riccati_eigenvalue_floor,
            mocap_position_observer_gain=args.mocap_position_observer_gain,
            mocap_relative_velocity_observer_gain=(
                args.mocap_relative_velocity_observer_gain
            ),
        )
        # MATLAB 要求 DeltaT/dt 为整数；在创建定时器前先验证。
        _ = self.p.nw

        # 两辆车的最新动捕位姿缓存。controller 在两者都到达后才初始化。
        self.target = PoseCache()
        self.pursuer = PoseCache()
        self.controller: Optional[BearingTrackingController] = None
        self.target_evasion = EvasiveTargetController(
            args.target_vd,
            args.target_ve,
            args.target_vj,
            args.target_direction_deg,
            args.max_speed,
            args.target_soft_boundary,
            args.field_x_min,
            args.field_x_max,
            args.field_y_min,
            args.field_y_max,
            args.target_boundary_margin,
            args.target_boundary_speed,
        )
        # 控制器时钟与 ROS 数据超时管理。
        self.start_time: Optional[float] = None
        self.experiment_complete = threading.Event()
        self.stopped = False
        self.last_control_time: Optional[float] = None
        self.last_log_time = 0.0
        self.stale_timeout = args.stale_timeout
        self.mocap_velocity_alignment_delay = args.mocap_velocity_alignment_delay
        self.bearing_history = deque(maxlen=256)
        # 两辆车使用完全相同的动捕速度估计器。只有 pursuer 的结果进入算法；
        # target 的结果专门用于检验 MATLAB 中“速度指令立即实现”的理想假设。
        self.pursuer_velocity_estimator = MocapVelocityEstimator(
            args.mocap_velocity_window,
            args.mocap_velocity_filter_tau,
            args.mocap_max_valid_speed,
        )
        self.target_velocity_estimator = MocapVelocityEstimator(
            args.mocap_velocity_window,
            args.mocap_velocity_filter_tau,
            args.mocap_max_valid_speed,
        )
        # 相对位置速度估计：窗口回归目标 Pb*(uO+dr/dt) 所需的 dr/dt。
        self.relative_velocity_estimator = MocapVelocityEstimator(
            args.mocap_velocity_window,
            args.mocap_velocity_filter_tau,
            2.0 * args.mocap_max_valid_speed,
        )

        # 根据车辆 ID 和可选话题覆盖参数生成实际订阅/发布话题。
        target_name = robot_name(args.target_id)
        pursuer_name = robot_name(args.pursuer_id)
        target_topic = args.target_pose_topic or f"/{target_name}/pose"
        pursuer_topic = args.pursuer_pose_topic or f"/{pursuer_name}/pose"
        vel_topic = args.vel_topic or f"/{pursuer_name}/cmd_vel"
        target_vel_topic = args.target_vel_topic or f"/{target_name}/cmd_vel"

        # 诊断通道与控制通道解耦：队列满时丢诊断帧，不等待绘图或终端输出。
        self.telemetry_queue: Optional[queue.Queue] = None
        self.telemetry_worker: Optional[ConsoleTelemetryWorker] = None
        self.recorder_queue: Optional[queue.Queue] = None
        self.recorder_worker: Optional[DiagnosticsRecorderWorker] = None
        self.session_dir: Optional[str] = None

        # 后台文本诊断线程。
        if args.telemetry_period > 0.0:
            self.telemetry_queue = queue.Queue(maxsize=256)
            self.telemetry_worker = ConsoleTelemetryWorker(
                self.telemetry_queue, args.telemetry_period
            )
            self.telemetry_worker.start()

        # 每次启动建立独立目录，CSV 便于人工查看，NPZ 便于 NumPy/MATLAB 读取。
        if args.save_data:
            session_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
            self.session_dir = os.path.abspath(os.path.join(args.data_dir, session_name))
            metadata = {
                "created_at": datetime.now().astimezone().isoformat(),
                "ros_master_uri": os.environ.get(
                    "ROS_MASTER_URI", "http://localhost:11311"
                ),
                "arguments": vars(args),
                "algorithm_parameters": asdict(self.p),
                "roles": {"target": target_name, "pursuer": pursuer_name},
                "topics": {
                    "target_pose": target_topic,
                    "pursuer_pose": pursuer_topic,
                    "target_command": target_vel_topic,
                    "pursuer_command": vel_topic,
                },
                "coordinate_frames": {
                    "algorithm_and_logged_vectors": "mocap world frame",
                    "published_velocity_commands": "each robot body frame",
                },
            }
            self.recorder_queue = queue.Queue(maxsize=4096)
            self.recorder_worker = DiagnosticsRecorderWorker(
                self.recorder_queue, self.session_dir, metadata,
                args.save_comparison_plot,
                args.save_plot,
                args.save_analysis_plot,
                args.plot,
            )
            self.recorder_worker.start()

        # ROS 1 队列深度 20：动捕和速度均只关心近期数据。
        self.target_subscription = rospy.Subscriber(
            target_topic, PoseStamped, self._target_callback, queue_size=20
        )
        self.pursuer_subscription = rospy.Subscriber(
            pursuer_topic, PoseStamped, self._pursuer_callback, queue_size=20
        )
        self.velocity_publisher = rospy.Publisher(vel_topic, Twist, queue_size=20)
        self.target_velocity_publisher = rospy.Publisher(
            target_vel_topic, Twist, queue_size=20
        )
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / args.rate), self.control_callback
        )

        rospy.loginfo(f"target pose:  {target_topic}")
        rospy.loginfo(f"pursuer pose: {pursuer_topic}")
        rospy.loginfo(f"velocity cmd: {vel_topic}")
        rospy.loginfo(f"config file:  {args.config}")
        rospy.loginfo(f"target cmd:   {target_vel_topic}")
        rospy.loginfo(
            "experiment duration: %s",
            f"{args.duration:g} s from controller initialization" if args.duration > 0.0
            else "manual stop with Ctrl+C",
        )
        rospy.loginfo(
            f"rate={args.rate:.1f} Hz, window={args.window:.2f} s, "
            f"rd={args.relative_radius:.2f} m, wd={args.relative_angular_speed:.2f} rad/s, "
            f"Kr={args.tracking_gain:.3g}I, "
            f"target law vd={args.target_vd:.2f}, ve={args.target_ve:.2f}, "
            f"vj={args.target_vj:.2f} m/s"
        )
        rospy.loginfo(
            f"history identifier: gamma={args.gamma:.2e}, "
            f"kh={args.history_gain:.3g}, "
            f"condition gate="
            f"{'disabled' if args.max_history_condition == 0.0 else f'{args.max_history_condition:.2e}'}"
        )
        rospy.loginfo(
            "tracking feedback position: %s"
            % (
                "direct mocap r=pT-pO (hybrid mode)"
                if args.use_mocap_relative_position else
                "bearing observer rHat (paper mode)"
            )
        )
        if args.target_soft_boundary:
            rospy.loginfo(
                "target soft boundary: x=[%.2f, %.2f], y=[%.2f, %.2f] m, "
                "margin=%.2f m, correction<=%.2f m/s per axis"
                % (
                    args.field_x_min, args.field_x_max,
                    args.field_y_min, args.field_y_max,
                    args.target_boundary_margin, args.target_boundary_speed,
                )
            )
        speed_gate = (
            "disabled" if args.mocap_max_valid_speed == 0.0
            else f"<={args.mocap_max_valid_speed:.2f} m/s"
        )
        rospy.loginfo(
            "identifier/observer uO source: mocap regression velocity "
            f"(window={args.mocap_velocity_window:.3f} s, "
            f"tau={args.mocap_velocity_filter_tau:.3f} s, "
            f"valid-speed gate={speed_gate})"
        )
        rospy.loginfo(
            "identifier bearing alignment delay: "
            f"{args.mocap_velocity_alignment_delay:.3f} s"
        )
        if args.telemetry_period > 0.0:
            rospy.loginfo(
                f"console telemetry every {args.telemetry_period:.1f} s"
            )
        if args.plot:
            rospy.loginfo(
                f"full-run comparison plot saved after experiment: {args.save_plot}"
            )
            rospy.loginfo(
                f"full-run analysis plot saved after experiment: {args.save_analysis_plot}"
            )
        if self.session_dir:
            rospy.loginfo(f"experiment data directory: {self.session_dir}")

    @staticmethod
    def _pose_to_cache(msg: PoseStamped) -> PoseCache:
        """解析 PoseStamped，同时保留接收时刻和传感器采样时刻。"""
        receive_time = time.monotonic()
        position = np.array([
            float(msg.pose.position.x),
            float(msg.pose.position.y),
        ], dtype=float)
        orientation = msg.pose.orientation
        yaw = yaw_from_quaternion(
            float(orientation.x), float(orientation.y),
            float(orientation.z), float(orientation.w),
        )
        stamp = msg.header.stamp
        sample_time = float(stamp.secs) + 1.0e-9 * float(stamp.nsecs)
        if sample_time <= 0.0:
            sample_time = receive_time
        return PoseCache(
            position=position,
            yaw=yaw,
            receive_time=receive_time,
            sample_time=sample_time,
        )

    def _target_callback(self, msg: PoseStamped) -> None:
        """更新 target 动捕位姿，并缓存位置供诊断实际速度拟合。"""
        pose = self._pose_to_cache(msg)
        self.target_velocity_estimator.add_pose(pose.sample_time, pose.position)
        self.target = pose

    def _pursuer_callback(self, msg: PoseStamped) -> None:
        """更新 pursuer 位姿，并保留高频位置序列供实际速度拟合。"""
        pose = self._pose_to_cache(msg)
        self.pursuer_velocity_estimator.add_pose(pose.sample_time, pose.position)
        self.pursuer = pose

    def _fresh(self, cache: PoseCache, now: float) -> bool:
        """检查位姿是否存在且未超过 stale_timeout。"""
        return cache.position is not None and now - cache.receive_time <= self.stale_timeout

    def publish_velocity(self, velocity_body: np.ndarray) -> None:
        """发布 pursuer 的车体系速度指令。"""
        self._publish_velocity(self.velocity_publisher, velocity_body)

    def publish_target_velocity(self, velocity_body: np.ndarray) -> None:
        """发布 target 的车体系速度指令。"""
        self._publish_velocity(self.target_velocity_publisher, velocity_body)

    def _publish_velocity(self, publisher, velocity_body: np.ndarray) -> None:
        """组装 Twist；只发布车体系平移，角速度固定为零。"""
        msg = Twist()
        msg.linear.x = float(velocity_body[0])
        msg.linear.y = float(velocity_body[1])
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        publisher.publish(msg)

    def update_target_motion(self, now: float) -> None:
        """执行 MATLAB targetInput 目标运动。

        相对向量只在这里归一化为方位 b，再由 ``EvasiveTargetController``
        计算世界速度；发布前使用 target 自身 yaw 转换到车体系。
        """
        if (not self._fresh(self.target, now)
                or not self._fresh(self.pursuer, now)):
            self.publish_target_velocity(np.zeros(2, dtype=float))
            return

        # r=pT-pO；b 从 pursuer 指向 target，与 MATLAB 的符号定义一致。
        relative = self.target.position - self.pursuer.position
        range_value = float(np.linalg.norm(relative))
        if range_value < 1.0e-6:
            self.publish_target_velocity(np.zeros(2, dtype=float))
            return

        # target 不跟踪 r_d 圆；它只执行“径向逃逸 + 切向运动”的未知输入规律。
        bearing = relative / range_value
        world_velocity = self.target_evasion.command(bearing, self.target.position)
        body_velocity = world_to_body(
            world_velocity, self.target.yaw
        )
        self.publish_target_velocity(body_velocity)

    @staticmethod
    def _try_enqueue(samples, sample: TelemetrySample) -> None:
        """非阻塞发送诊断帧：队列满时宁可丢帧，也不让控制周期抖动。"""
        if samples is None:
            return
        try:
            samples.put_nowait(sample)
        except queue.Full:
            pass

    def publish_telemetry(self, sample: TelemetrySample) -> None:
        """把诊断帧送往终端和完整记录器，不阻塞控制。"""
        self._try_enqueue(self.telemetry_queue, sample)
        self._try_enqueue(self.recorder_queue, sample)

    def reset_pursuer_velocity_measurement(self) -> None:
        """清空 pursuer 速度拟合状态，防止跨停机/数据中断继续做差。"""
        self.pursuer_velocity_estimator.reset()

    def reset_all_velocity_measurements(self) -> None:
        """同时清除两车速度历史，用于启动和动捕中断后的干净重启。"""
        self.pursuer_velocity_estimator.reset()
        self.target_velocity_estimator.reset()
        self.relative_velocity_estimator.reset()

    def reset_bearing_alignment(self) -> None:
        """清除方位历史，避免动捕中断后将旧方位配给新速度。"""
        self.bearing_history.clear()

    def aligned_bearing(self, now: float, bearing: np.ndarray) -> Optional[np.ndarray]:
        """返回与因果动捕速度估计同一时刻的历史方位。

        最近位置窗的速度估计代表稍早的运动；用等量延迟的方位构造投影和
        RBF 回归量，避免把过去速度与当前方位错配。反馈位置和期望轨迹仍使用
        当前时刻，控制输出不额外引入整窗延迟。
        """
        self.bearing_history.append((now, bearing.copy()))
        if self.mocap_velocity_alignment_delay <= 0.0:
            return bearing
        target_time = now - self.mocap_velocity_alignment_delay
        while (len(self.bearing_history) >= 2
               and self.bearing_history[1][0] <= target_time):
            self.bearing_history.popleft()
        if not self.bearing_history or self.bearing_history[0][0] > target_time:
            return None
        earlier_time, earlier_bearing = self.bearing_history[0]
        if len(self.bearing_history) == 1:
            return earlier_bearing.copy()
        later_time, later_bearing = self.bearing_history[1]
        fraction = (target_time - earlier_time) / max(later_time - earlier_time, 1.0e-12)
        aligned = earlier_bearing + fraction * (later_bearing - earlier_bearing)
        return aligned / max(float(np.linalg.norm(aligned)), 1.0e-12)

    def measured_pursuer_velocity(self) -> np.ndarray:
        """由高频动捕位置拟合 pursuer 的实际世界速度。

        步骤：截取最近 ``mocap_velocity_window`` 秒数据 -> 对 x(t)、y(t)
        分别做同一条时间基线上的最小二乘斜率 -> 剔除不可能的速度尖峰 ->
        一阶低通。输出随后作为论文方程中的实际 ``uO``。
        """
        return self.pursuer_velocity_estimator.estimate(self.p.dt)

    def measured_target_velocity(self) -> np.ndarray:
        """由 target 动捕位置估计实际世界速度；仅用于诊断，不反馈算法。"""
        return self.target_velocity_estimator.estimate(self.p.dt)

    def control_callback(self, _event=None) -> None:
        """50 Hz（默认）主控制回调。

        顺序非常重要：
          1. 更新 target 指令并检查两车动捕是否新鲜；
          2. 由真实相对位置只构造单位方位 b；
          3. 估计 pursuer 自身实际速度；
          4. 用 b 和实际自速度推进论文辨识器/观测器；
          5. 将世界系 uO_cmd 旋转到车体系并发布；
          6. 用动捕真值计算诊断误差，但绝不反馈这些误差。
        """
        if self.stopped:
            return
        now = time.monotonic()
        if (self.experiment_complete.is_set()
                or (self.args.duration > 0.0 and self.start_time is not None
                    and now - self.start_time >= self.args.duration)):
            # 主线程负责停止定时器和落盘；回调先停车，且不再发送运动指令。
            self.publish_target_velocity(np.zeros(2, dtype=float))
            self.publish_velocity(np.zeros(2, dtype=float))
            if not self.experiment_complete.is_set():
                rospy.loginfo("experiment duration reached; stopping and saving data")
                self.experiment_complete.set()
            return
        target_fresh = self._fresh(self.target, now)
        pursuer_fresh = self._fresh(self.pursuer, now)
        if not target_fresh or not pursuer_fresh:
            rospy.logwarn_throttle(
                2.0,
                "waiting for fresh VRPN poses: target=%s, pursuer=%s",
                target_fresh,
                pursuer_fresh,
            )
            self.reset_all_velocity_measurements()
            self.reset_bearing_alignment()
            self.publish_target_velocity(np.zeros(2, dtype=float))
            self.publish_velocity(np.zeros(2, dtype=float))
            return

        # target 和 pursuer 共用同一个控制定时器，避免两个独立循环明显错拍。
        self.update_target_motion(now)

        # 真实 r、rho 在 ROS 外层可用于构造方位和诊断；核心控制器只收到 b。
        relative = self.target.position - self.pursuer.position
        range_value = float(np.linalg.norm(relative))
        if range_value < 1.0e-6:
            self.publish_velocity(np.zeros(2, dtype=float))
            self.get_logger().warn("target and pursuer positions are coincident; bearing is undefined")
            return
        bearing = relative / range_value
        identifier_bearing = self.aligned_bearing(now, bearing)
        if identifier_bearing is None:
            self.publish_velocity(np.zeros(2, dtype=float))
            return
        # 这是 pursuer 自身实际状态估计，不是 uO_cmd，也不是 target 速度。
        mocap_velocity = self.measured_pursuer_velocity()

        if self.controller is None:
            # 第一帧只初始化算法。清空启动前手动搬车产生的位置数据，避免污染
            # 第一个速度拟合和历史窗口；本周期给 pursuer 发送零速度。
            self.reset_all_velocity_measurements()
            initial_relative_position = (
                relative if self.args.use_mocap_relative_position else None
            )
            self.controller = BearingTrackingController(
                self.p, identifier_bearing, initial_relative_position
            )
            self.start_time = now
            self.last_control_time = now
            rospy.loginfo("received both poses; bearing controller initialized")
            self.publish_velocity(np.zeros(2, dtype=float))
            return

        # 使用真实回调间隔积分，同时把异常长间隔限制到 2 个名义周期，防止
        # ROS 短暂停顿造成一次过大的 RK4 跳变。
        raw_dt = now - (self.last_control_time or now)
        self.last_control_time = now
        dt = min(max(raw_dt, 1.0e-4), 2.0 * self.p.dt)
        elapsed = now - (self.start_time or now)
        # 窗口回归目标默认保持论文原式 Pb*uO。相对速度修正仅在该开关打开时
        # 才运行，关闭时本回调路径与已验证基线完全一致。
        relative_velocity = None
        if (self.args.identifier_use_relative_velocity
                or self.args.mocap_relative_velocity_observer_gain > 0.0):
            self.relative_velocity_estimator.add_pose(elapsed, relative.copy())
            relative_velocity = self.relative_velocity_estimator.estimate(self.p.dt)
        if self.args.identifier_use_relative_velocity:
            u_ident = mocap_velocity + relative_velocity
        else:
            u_ident = mocap_velocity
        # 目标侧执行误差用于历史窗口质量门：目标命令与动捕实际速度之差。
        (target_command_world, target_nominal_command_world,
         target_boundary_command_world) = self.target_evasion.command_parts(
            bearing, self.target.position
        )
        target_mocap_velocity = self.measured_target_velocity()
        target_execution_error = None
        if np.all(np.isfinite(target_mocap_velocity)):
            target_execution_error = target_command_world - target_mocap_velocity
        # 核心算法输出世界系 uO_cmd；这里传入的是实际自速度 mocap_velocity。
        world_velocity, diagnostics = self.controller.step(
            elapsed,
            identifier_bearing,
            dt,
            mocap_velocity,
            relative if self.args.use_mocap_relative_position else None,
            u_ident=u_ident,
            relative_velocity_feedback=relative_velocity,
            target_execution_error=target_execution_error,
        )
        # 最终硬件指令再次限幅作为发布边界，然后按 pursuer 当前 yaw 转车体系。
        world_velocity = limit_vector_norm(world_velocity, self.p.max_linear_speed)
        body_velocity = world_to_body(
            world_velocity, self.pursuer.yaw
        )
        self.publish_velocity(body_velocity)

        # 以下真实 target/relative 量只用于诊断：估计器并未收到真实目标距离、
        # 真实相对位置或真实 target 速度。目标命令与实际速度已在上方提前计算。
        target_command_body = world_to_body(
            target_command_world, self.target.yaw
        )
        target_velocity_valid = self.target_velocity_estimator.have_filtered_velocity
        tracking_error = float(np.linalg.norm(relative - diagnostics["desired_relative"]))
        localization_error = float(
            np.linalg.norm(relative - diagnostics["relative_estimate"])
        )
        # 实车主误差使用动捕实际 target 速度；另存“指令对估计”误差与 MATLAB
        # 理想值对照。速度拟合尚未有效时用 NaN，避免把零误当成真实静止。
        input_error = (
            float(np.linalg.norm(
                target_mocap_velocity - diagnostics["target_velocity_estimate"]
            ))
            if target_velocity_valid else float("nan")
        )
        target_input_estimation_error_command = float(
            np.linalg.norm(
                target_command_world - diagnostics["target_velocity_estimate"]
            )
        )
        target_command_execution_error = (
            float(np.linalg.norm(target_command_world - target_mocap_velocity))
            if target_velocity_valid else float("nan")
        )

        # 比较世界系指令与实际速度的夹角。dir≈0 表示方向执行正确，±180 度
        # 表示反向，±90 度通常表示坐标轴偏转。该角度只用于诊断。
        command_norm = float(np.linalg.norm(world_velocity))
        mocap_norm = float(np.linalg.norm(mocap_velocity))
        command_execution_error = (
            float(np.linalg.norm(world_velocity - mocap_velocity))
            if self.pursuer_velocity_estimator.have_filtered_velocity
            else float("nan")
        )
        direction_error_deg = float("nan")
        if command_norm > 0.03 and mocap_norm > 0.03:
            cross = (
                world_velocity[0] * mocap_velocity[1]
                - world_velocity[1] * mocap_velocity[0]
            )
            dot = float(world_velocity @ mocap_velocity)
            direction_error_deg = math.degrees(math.atan2(cross, dot))

        # 打包绘图和离线定位所需数据；队列操作不阻塞控制。
        pursuer_estimator = self.pursuer_velocity_estimator
        target_estimator = self.target_velocity_estimator
        raw_command = diagnostics["raw_world_velocity"]
        relative_estimate = diagnostics["relative_estimate"]
        desired_relative = diagnostics["desired_relative"]
        target_velocity_estimate = diagnostics["target_velocity_estimate"]
        c_target = diagnostics["control_target_estimate"]
        c_relative_velocity = diagnostics["control_relative_velocity"]
        c_trajectory = diagnostics["control_trajectory"]
        c_feedback = diagnostics["control_feedback"]
        c_feedback_position = diagnostics["control_feedback_position"]
        self.publish_telemetry(TelemetrySample(
            t=elapsed,
            control_dt_raw=raw_dt,
            control_dt_used=dt,
            target_x=float(self.target.position[0]),
            target_y=float(self.target.position[1]),
            target_yaw=float(self.target.yaw),
            pursuer_x=float(self.pursuer.position[0]),
            pursuer_y=float(self.pursuer.position[1]),
            pursuer_yaw=float(self.pursuer.yaw),
            target_pose_age=max(0.0, now - self.target.receive_time),
            pursuer_pose_age=max(0.0, now - self.pursuer.receive_time),
            relative_x=float(relative[0]),
            relative_y=float(relative[1]),
            bearing_x=float(identifier_bearing[0]),
            bearing_y=float(identifier_bearing[1]),
            desired_relative_x=float(desired_relative[0]),
            desired_relative_y=float(desired_relative[1]),
            relative_estimate_x=float(relative_estimate[0]),
            relative_estimate_y=float(relative_estimate[1]),
            range_value=range_value,
            tracking_error=tracking_error,
            localization_error=localization_error,
            input_error=input_error,
            target_command_world_x=float(target_command_world[0]),
            target_command_world_y=float(target_command_world[1]),
            target_nominal_command_world_x=float(target_nominal_command_world[0]),
            target_nominal_command_world_y=float(target_nominal_command_world[1]),
            target_boundary_command_world_x=float(target_boundary_command_world[0]),
            target_boundary_command_world_y=float(target_boundary_command_world[1]),
            target_boundary_active=int(
                np.linalg.norm(target_boundary_command_world) > 1.0e-9
            ),
            target_boundary_clearance=float(np.min(np.array([
                self.target.position[0] - self.target_evasion.field_min[0],
                self.target_evasion.field_max[0] - self.target.position[0],
                self.target.position[1] - self.target_evasion.field_min[1],
                self.target_evasion.field_max[1] - self.target.position[1],
            ]))),
            target_command_body_x=float(target_command_body[0]),
            target_command_body_y=float(target_command_body[1]),
            target_mocap_velocity_x=float(target_mocap_velocity[0]),
            target_mocap_velocity_y=float(target_mocap_velocity[1]),
            target_raw_velocity_x=float(target_estimator.raw_velocity[0]),
            target_raw_velocity_y=float(target_estimator.raw_velocity[1]),
            target_velocity_estimate_x=float(target_velocity_estimate[0]),
            target_velocity_estimate_y=float(target_velocity_estimate[1]),
            target_command_execution_error=target_command_execution_error,
            target_input_estimation_error_command=target_input_estimation_error_command,
            target_velocity_sample_count=target_estimator.sample_count,
            target_velocity_sample_span=target_estimator.sample_span,
            target_velocity_sample_accepted=int(target_estimator.sample_accepted),
            pursuer_speed=command_norm,
            raw_command_world_x=float(raw_command[0]),
            raw_command_world_y=float(raw_command[1]),
            command_world_x=float(world_velocity[0]),
            command_world_y=float(world_velocity[1]),
            command_body_x=float(body_velocity[0]),
            command_body_y=float(body_velocity[1]),
            mocap_velocity_x=float(mocap_velocity[0]),
            mocap_velocity_y=float(mocap_velocity[1]),
            raw_mocap_velocity_x=float(pursuer_estimator.raw_velocity[0]),
            raw_mocap_velocity_y=float(pursuer_estimator.raw_velocity[1]),
            command_execution_error=command_execution_error,
            direction_error_deg=direction_error_deg,
            control_target_estimate_x=float(c_target[0]),
            control_target_estimate_y=float(c_target[1]),
            control_relative_velocity_x=float(c_relative_velocity[0]),
            control_relative_velocity_y=float(c_relative_velocity[1]),
            control_trajectory_x=float(c_trajectory[0]),
            control_trajectory_y=float(c_trajectory[1]),
            control_feedback_x=float(c_feedback[0]),
            control_feedback_y=float(c_feedback[1]),
            control_feedback_position_x=float(c_feedback_position[0]),
            control_feedback_position_y=float(c_feedback_position[1]),
            control_feedback_uses_mocap=diagnostics["control_feedback_uses_mocap"],
            theta_hat_norm=diagnostics["theta_hat_norm"],
            identifier_current_theta_dot_norm=diagnostics[
                "identifier_current_theta_dot_norm"
            ],
            identifier_history_theta_dot_norm=diagnostics[
                "identifier_history_theta_dot_norm"
            ],
            identifier_leak_theta_dot_norm=diagnostics[
                "identifier_leak_theta_dot_norm"
            ],
            identifier_current_output_rate_x=float(diagnostics[
                "identifier_current_output_rate"
            ][0]),
            identifier_current_output_rate_y=float(diagnostics[
                "identifier_current_output_rate"
            ][1]),
            identifier_history_output_rate_x=float(diagnostics[
                "identifier_history_output_rate"
            ][0]),
            identifier_history_output_rate_y=float(diagnostics[
                "identifier_history_output_rate"
            ][1]),
            identifier_leak_output_rate_x=float(diagnostics[
                "identifier_leak_output_rate"
            ][0]),
            identifier_leak_output_rate_y=float(diagnostics[
                "identifier_leak_output_rate"
            ][1]),
            identifier_total_output_rate_x=float(diagnostics[
                "identifier_total_output_rate"
            ][0]),
            identifier_total_output_rate_y=float(diagnostics[
                "identifier_total_output_rate"
            ][1]),
            xi_position_norm=diagnostics["xi_position_norm"],
            xi_velocity_norm=diagnostics["xi_velocity_norm"],
            po_min_eigenvalue=diagnostics["po_min_eigenvalue"],
            po_max_eigenvalue=diagnostics["po_max_eigenvalue"],
            po_condition=diagnostics["po_condition"],
            history_count=diagnostics["history_count"],
            active_directions=diagnostics["active_directions"],
            history_min_active_eigenvalue=diagnostics["history_min_active_eigenvalue"],
            history_max_eigenvalue=diagnostics["history_max_eigenvalue"],
            history_condition=diagnostics["history_condition"],
            history_update_gated=diagnostics["history_update_gated"],
            current_window_count=diagnostics["current_window_count"],
            current_window_excitation=diagnostics["current_window_excitation"],
            chi=diagnostics["chi"],
            history_last_window_execution_error=diagnostics[
                "history_last_window_execution_error"
            ],
            history_last_window_accepted=diagnostics[
                "history_last_window_accepted"
            ],
            pursuer_velocity_sample_count=pursuer_estimator.sample_count,
            pursuer_velocity_sample_span=pursuer_estimator.sample_span,
            pursuer_velocity_sample_accepted=int(pursuer_estimator.sample_accepted),
            target_pose_sample_time=float(self.target.sample_time),
            pursuer_pose_sample_time=float(self.pursuer.sample_time),
            target_velocity_timestamp_resets=target_estimator.timestamp_reset_count,
            pursuer_velocity_timestamp_resets=pursuer_estimator.timestamp_reset_count,
        ))

        if now - self.last_log_time > 2.0:
            self.last_log_time = now
            rospy.loginfo(
                "range=%.2f m, |uO|=%.3f m/s, history=%d, active=%d, chi=%d"
                % (
                    range_value,
                    float(np.linalg.norm(world_velocity)),
                    diagnostics["history_count"],
                    diagnostics["active_directions"],
                    int(diagnostics["chi"]),
                )
            )

    def stop(self) -> None:
        """停止后台诊断并在 ROS 通信仍有效时向两辆车发送零速度。"""
        if self.stopped:
            return
        self.stopped = True
        self.control_timer.shutdown()
        # 在耗时的绘图/落盘前优先停止车辆；shutdown 回调中 ROS 可能已断开，
        # 因此只在通信仍可用时发布。
        if not rospy.is_shutdown():
            self.publish_velocity(np.zeros(2, dtype=float))
            self.publish_target_velocity(np.zeros(2, dtype=float))
        if self.telemetry_worker is not None:
            self.telemetry_worker.stop()
            self.telemetry_worker.join(timeout=0.5)
        if self.recorder_worker is not None:
            self.recorder_worker.stop()
            # 两车已收到零速度；现在等待完整 CSV/NPZ 和静态全程图保存完毕。
            print("[DATA] finalizing full-run data and figures...", flush=True)
            self.recorder_worker.join()


def _load_yaml_defaults(path: str, parser: argparse.ArgumentParser) -> dict:
    """读取扁平 YAML，并检查键名，防止参数拼写错误被静默忽略。"""
    absolute_path = os.path.abspath(os.path.expanduser(path))
    try:
        with open(absolute_path, "r", encoding="utf-8") as stream:
            values = yaml.safe_load(stream)
    except OSError as exc:
        parser.error(f"cannot read config file {absolute_path}: {exc}")
    except yaml.YAMLError as exc:
        parser.error(f"invalid YAML in {absolute_path}: {exc}")

    if values is None:
        values = {}
    if not isinstance(values, dict):
        parser.error("YAML root must be a mapping of parameter names to values")

    # 灯光配置由 --led-config 读取，不属于跟踪控制器的扁平参数。
    values.pop("led_control", None)
    valid_keys = {action.dest for action in parser._actions}
    unknown = sorted(set(values) - valid_keys)
    if unknown:
        parser.error("unknown YAML parameter(s): " + ", ".join(unknown))

    # 相对输出路径相对于配置文件目录，而不是启动终端的当前目录。
    config_dir = os.path.dirname(absolute_path)
    for key in (
            "save_plot", "save_analysis_plot", "save_comparison_plot", "data_dir"):
        value = values.get(key)
        if isinstance(value, str) and value and not os.path.isabs(value):
            values[key] = os.path.join(config_dir, value)
    values["config"] = absolute_path
    return values


def add_boolean_argument(parser: argparse.ArgumentParser, name: str,
                         default: bool, help_text: str) -> None:
    """添加 ``--flag`` 和 ``--no-flag``，兼容 ROS Noetic 的 Python 3.8。"""
    option_name = name[2:]
    destination = option_name.replace("-", "_")
    parser.add_argument(name, dest=destination, action="store_true",
                        default=default, help=help_text)
    parser.add_argument(f"--no-{option_name}", dest=destination,
                        action="store_false", help=argparse.SUPPRESS)


def parse_args() -> argparse.Namespace:
    """先读取 YAML 默认值，再由命令行参数作临时覆盖。"""
    parser = argparse.ArgumentParser(description="2-D bearing-only target tracking ROS 1 demo")
    parser.add_argument("--led-config", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "bearing_tracking_params.yaml"
    ), help="YAML file containing led_control colors, brightness and topics")
    parser.add_argument("--leds-only", action="store_true",
                        help="set LED colors and exit without controlling motion")
    parser.add_argument("--no-leds", action="store_true",
                        help="skip setting the LED colors")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_PATH,
        help="YAML parameter file loaded at startup",
    )
    parser.add_argument("--target-id", type=int, default=4, help="target robot id (default: 4)")
    parser.add_argument("--pursuer-id", type=int, default=3, help="pursuer robot id (default: 3)")
    parser.add_argument("--target-pose-topic", default=None)
    parser.add_argument("--pursuer-pose-topic", default=None)
    parser.add_argument("--vel-topic", default=None)
    parser.add_argument("--target-vel-topic", default=None)
    parser.add_argument("--rate", type=float, default=50.0, help="control/publish rate in Hz")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="experiment seconds from controller initialization; 0 means Ctrl+C to stop")
    parser.add_argument("--window", type=float, default=1.0, help="finite data window in seconds")
    parser.add_argument(
        "--gamma", type=float, default=1.0e-3,
        help="minimum active history-stack eigenvalue (default: 1e-3)",
    )
    parser.add_argument(
        "--history-gain", type=float, default=1.2,
        help="history-stack learning gain kh; 0 disables the history update term",
    )
    parser.add_argument(
        "--max-history-condition", type=float, default=1.0e3,
        help="gate ill-conditioned history estimates; 0 disables the gate",
    )
    parser.add_argument("--relative-radius", type=float, default=0.5, help="desired relative radius in m")
    parser.add_argument("--relative-angular-speed", type=float, default=0.5,
                        help="desired relative trajectory angular speed in rad/s")
    parser.add_argument("--relative-initial-phase-deg", type=float, default=120.0,
                        help="initial phase of desired relative trajectory in degrees")
    parser.add_argument(
        "--max-speed", type=float, default=0.25,
        help="world velocity norm limit in m/s; 0 disables saturation (MATLAB behavior)",
    )
    parser.add_argument(
        "--tracking-gain", type=float, default=0.10,
        help="relative-trajectory feedback gain Kr=tracking_gain*I",
    )
    parser.add_argument("--rbf-count", type=int, default=30,
                        help="RBF node count q for each output channel")
    parser.add_argument("--rbf-sigma", type=float, default=0.52,
                        help="Gaussian RBF width sigma")
    parser.add_argument("--history-size", type=int, default=30,
                        help="maximum number Nh of completed history windows")
    parser.add_argument(
        "--max-history-execution-error", type=float, default=0.0,
        help="maximum command-vs-mocap RMS for an accepted history window; 0 disables",
    )
    parser.add_argument(
        "--max-history-target-execution-error", type=float, default=0.0,
        help="maximum target command-vs-mocap RMS for an accepted window; 0 disables",
    )
    parser.add_argument(
        "--history-forgetting-factor", type=float, default=1.0,
        help="exponential forgetting applied per accepted history window; 1 disables",
    )
    parser.add_argument("--current-window-threshold", type=float, default=1.0e-5,
                        help="active eigenvalue threshold ac for the current window")
    parser.add_argument("--window-valid-tolerance", type=float,
                        default=100.0 * np.finfo(float).eps,
                        help="minimum excitation for accepting a completed window")
    parser.add_argument("--current-window-gain", type=float, default=2.0,
                        help="current-window concurrent-learning gain kc")
    parser.add_argument("--sigma-theta", type=float, default=0.003,
                        help="parameter leakage/regularization coefficient")
    parser.add_argument("--target-estimate-filter-bandwidth", type=float, default=5.0,
                        help="uT_hat first-order filter bandwidth af")
    add_boolean_argument(
        parser, "--use-mocap-relative-position", False,
        "use direct mocap r=pT-pO in Kr feedback instead of observer rHat",
    )
    parser.add_argument("--riccati-lambda", type=float, default=0.10,
                        help="Riccati stabilizing parameter lambda_s")
    parser.add_argument("--observer-noise-weight", type=float, default=0.02,
                        help="observer noise weight ro")
    parser.add_argument("--riccati-eigenvalue-floor", type=float, default=1.0e-9,
                        help="minimum eigenvalue used when projecting Po to SPD")
    parser.add_argument(
        "--mocap-position-observer-gain", type=float, default=0.0,
        help="hybrid-mode mocap correction gain for rHat; 0 disables it",
    )
    parser.add_argument(
        "--mocap-relative-velocity-observer-gain", type=float, default=0.0,
        help="mocap relative-velocity innovation gain for deltaVHat; 0 disables it",
    )
    parser.add_argument("--target-vd", type=float, default=0.0,
                        help="MATLAB target law fixed-direction speed")
    parser.add_argument("--target-ve", type=float, default=0.1,
                        help="MATLAB target law radial escape speed")
    parser.add_argument("--target-vj", type=float, default=0.1,
                        help="MATLAB target law tangential speed")
    parser.add_argument("--target-direction-deg", type=float, default=155.0,
                        help="MATLAB target law fixed direction in degrees")
    add_boolean_argument(
        parser, "--target-soft-boundary", False,
        "enable a smooth world-frame inward velocity near the target field boundary",
    )
    parser.add_argument("--field-x-min", type=float, default=-1.5)
    parser.add_argument("--field-x-max", type=float, default=1.5)
    parser.add_argument("--field-y-min", type=float, default=-1.5)
    parser.add_argument("--field-y-max", type=float, default=1.5)
    parser.add_argument("--target-boundary-margin", type=float, default=0.30,
                        help="distance from the fence at which target correction starts")
    parser.add_argument("--target-boundary-speed", type=float, default=0.15,
                        help="maximum inward correction per world axis in m/s")
    parser.add_argument("--stale-timeout", type=float, default=0.3,
                        help="stop if either pose is older than this many seconds")
    parser.add_argument("--mocap-velocity-window", type=float, default=0.06,
                        help="recent mocap window used for velocity line fitting in seconds")
    parser.add_argument("--mocap-velocity-filter-tau", type=float, default=0.02,
                        help="low-pass time constant for measured pursuer velocity; 0 disables filtering")
    parser.add_argument(
        "--mocap-velocity-alignment-delay", type=float, default=0.0,
        help="delay bearing used by identifier to align it with causal mocap velocity",
    )
    add_boolean_argument(
        parser, "--identifier-use-relative-velocity", False,
        "use u_o plus mocap relative velocity as the identifier window target",
    )
    parser.add_argument(
        "--mocap-max-valid-speed", type=float, default=0.8,
        help="reject larger mocap-derived speeds; 0 disables this rejection",
    )
    parser.add_argument("--telemetry-period", type=float, default=1.0,
                        help="console diagnostics period in seconds; 0 disables it")
    add_boolean_argument(parser, "--recording-filter", True,
                         "exclude poor sensor/timing samples from cleaned exports and plots")
    parser.add_argument("--recording-max-pose-age", type=float, default=0.02)
    parser.add_argument("--recording-min-velocity-span-ratio", type=float, default=0.5)
    parser.add_argument("--recording-recovery-time", type=float, default=0.15)
    add_boolean_argument(
        parser, "--plot", True,
        "save full-run figures after the experiment; --no-plot disables figures",
    )
    parser.add_argument("--plot-history", type=float, default=30.0,
                        help="legacy option; saved plots always cover the complete run")
    parser.add_argument("--save-plot", default=DEFAULT_PLOT_PATH,
                        help="full-run comparison PNG saved after the experiment")
    parser.add_argument("--save-analysis-plot", default=DEFAULT_ANALYSIS_PLOT_PATH,
                        help="full-run analysis PNG saved after the experiment")
    parser.add_argument("--save-comparison-plot", default=DEFAULT_COMPARISON_PLOT_PATH,
                        help="full-run simulation-style PNG; saved when the experiment stops")
    add_boolean_argument(
        parser, "--save-data", True,
        "save full CSV/NPZ/metadata by default; use --no-save-data to disable",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="parent directory for timestamped experiment data")

    # 先确定配置文件路径。完整解析时，显式命令行参数优先于 YAML 值。
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args()
    parser.set_defaults(**_load_yaml_defaults(config_args.config, parser))
    return parser.parse_args()


def main() -> None:
    """参数校验、ROS 初始化、spin 和安全退出入口。"""
    args = parse_args()
    from tianbot_led_control import load_led_commands, publish_led_commands
    led_commands, led_timeout = (
        ([], 0.0) if args.no_leds else load_led_commands(args.led_config)
    )
    if args.leds_only:
        rospy.init_node("tianbot_set_leds")
        publish_led_commands(led_commands, led_timeout)
        return
    if args.rate <= 0.0 or args.window <= 0.0:
        raise ValueError("rate and window must be positive")
    if (not math.isfinite(args.recording_max_pose_age) or args.recording_max_pose_age <= 0
            or not math.isfinite(args.recording_min_velocity_span_ratio)
            or not 0 < args.recording_min_velocity_span_ratio <= 1
            or not math.isfinite(args.recording_recovery_time) or args.recording_recovery_time < 0):
        raise ValueError("invalid recording quality thresholds")
    if not math.isfinite(args.duration) or args.duration < 0.0:
        raise ValueError("duration must be finite and non-negative; 0 means manual stop")
    if args.max_speed < 0.0:
        raise ValueError("max-speed must be non-negative; use 0 to disable saturation")
    if args.tracking_gain < 0.0:
        raise ValueError("tracking-gain must be non-negative")
    if args.gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if args.history_gain < 0.0 or args.max_history_condition < 0.0:
        raise ValueError("history-gain and max-history-condition must be non-negative")
    if args.rbf_count <= 0 or args.rbf_sigma <= 0.0 or args.history_size <= 0:
        raise ValueError("rbf-count, rbf-sigma and history-size must be positive")
    if args.max_history_execution_error < 0.0:
        raise ValueError("max-history-execution-error must be non-negative")
    if args.max_history_target_execution_error < 0.0:
        raise ValueError(
            "max-history-target-execution-error must be non-negative"
        )
    if not 0.0 < args.history_forgetting_factor <= 1.0:
        raise ValueError("history-forgetting-factor must be in (0, 1]")
    if args.current_window_threshold < 0.0 or args.window_valid_tolerance < 0.0:
        raise ValueError("window thresholds must be non-negative")
    if args.current_window_gain < 0.0 or args.sigma_theta < 0.0:
        raise ValueError("learning gains and sigma-theta must be non-negative")
    if args.target_estimate_filter_bandwidth <= 0.0:
        raise ValueError("target-estimate-filter-bandwidth must be positive")
    if args.riccati_lambda <= 0.0 or args.observer_noise_weight <= 0.0:
        raise ValueError("riccati-lambda and observer-noise-weight must be positive")
    if args.riccati_eigenvalue_floor <= 0.0:
        raise ValueError("riccati-eigenvalue-floor must be positive")
    if args.mocap_position_observer_gain < 0.0:
        raise ValueError("mocap-position-observer-gain must be non-negative")
    if args.mocap_relative_velocity_observer_gain < 0.0:
        raise ValueError(
            "mocap-relative-velocity-observer-gain must be non-negative"
        )
    if args.telemetry_period < 0.0 or args.plot_history <= 0.0:
        raise ValueError("telemetry-period must be non-negative and plot-history positive")
    if args.mocap_velocity_window <= 0.0:
        raise ValueError("mocap-velocity-window must be positive")
    if args.mocap_velocity_alignment_delay < 0.0:
        raise ValueError("mocap-velocity-alignment-delay must be non-negative")
    if args.mocap_velocity_filter_tau < 0.0:
        raise ValueError("mocap-velocity-filter-tau must be non-negative")
    if args.mocap_max_valid_speed < 0.0:
        raise ValueError(
            "mocap-max-valid-speed must be non-negative; use 0 to disable rejection"
        )
    if args.target_id == args.pursuer_id:
        raise ValueError("target-id and pursuer-id must be different")
    if args.field_x_min >= args.field_x_max or args.field_y_min >= args.field_y_max:
        raise ValueError("each field minimum must be smaller than its maximum")
    half_smallest_field_side = 0.5 * min(
        args.field_x_max - args.field_x_min,
        args.field_y_max - args.field_y_min,
    )
    if not 0.0 < args.target_boundary_margin < half_smallest_field_side:
        raise ValueError("target-boundary-margin must be positive and less than half the field size")
    if args.target_boundary_speed < 0.0:
        raise ValueError("target-boundary-speed must be non-negative")

    # 让 KeyboardInterrupt 先进入 finally 中的 node.stop()，从而在 ROS
    # shutdown 断开发布器前发送零速度；默认 rospy 处理顺序恰好相反。
    rospy.init_node("bearing_tracking_direct_v1", disable_signals=True)
    try:
        publish_led_commands(led_commands, led_timeout)
    except RuntimeError as error:
        rospy.logwarn("LED setup: %s", error)
    node = BearingTrackingNode(args)
    try:
        # 自动计时结束时保持 ROS 通信可用，让 finally 先向双车发零速再保存。
        while not rospy.is_shutdown():
            if node.experiment_complete.wait(timeout=0.1):
                break
    except KeyboardInterrupt:
        rospy.loginfo("Ctrl+C received; stopping vehicles and diagnostics")
    finally:
        # 后续清理有严格超时；忽略重复 Ctrl+C，避免零速发布和图片落盘
        # 在 join 中途再次被 KeyboardInterrupt 打断。
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        node.stop()
        if not rospy.is_shutdown():
            rospy.signal_shutdown("bearing tracking stopped")


if __name__ == "__main__":
    main()
