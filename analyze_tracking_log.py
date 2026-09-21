#!/usr/bin/env python3
"""从完整 CSV 重画关键诊断量，不参与 ROS 实时控制。"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """把日志的每一列读成 NumPy 数组；NaN 保留为 NaN。"""
    with open(path, newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty tracking log: {path}")
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def vector_norm(data, x_name, y_name):
    return np.hypot(data[x_name], data[y_name])


def mark_history_updates(axes, time, history_count):
    """用竖线标出每个 1 s 历史窗首次进入辨识器的时刻。"""
    changes = np.flatnonzero(np.diff(history_count, prepend=history_count[0]) > 0)
    for axis in axes:
        for index in changes:
            axis.axvline(time[index], color="0.75", linewidth=0.8, zorder=0)


def main():
    parser = argparse.ArgumentParser(description="Plot a complete bearing-tracking CSV")
    parser.add_argument("csv_path")
    parser.add_argument("--output", default="latest_experiment_full_diagnosis.png")
    parser.add_argument("--desired-range", type=float, default=0.5)
    args = parser.parse_args()

    data = load_csv(args.csv_path)
    time = data["t"]
    raw_command = vector_norm(data, "raw_command_world_x", "raw_command_world_y")
    sent_command = vector_norm(data, "command_world_x", "command_world_y")
    actual_speed = vector_norm(data, "mocap_velocity_x", "mocap_velocity_y")
    target_actual = vector_norm(
        data, "target_mocap_velocity_x", "target_mocap_velocity_y"
    )
    target_estimate = vector_norm(
        data, "target_velocity_estimate_x", "target_velocity_estimate_y"
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)

    ax = axes[0, 0]
    ax.plot(time, data["range_value"], label=r"actual $\|r\|$")
    ax.axhline(args.desired_range, linestyle="--", color="0.3", label=r"desired $R_d$")
    ax.plot(time, data["tracking_error"], label=r"tracking $\|r-r_d\|$")
    ax.plot(time, data["localization_error"], label=r"localization $\|r-\hat r\|$")
    ax.set_title("Range and state errors")
    ax.set_ylabel("m")

    ax = axes[0, 1]
    ax.plot(time, raw_command, label=r"raw $\|u_O\|$")
    ax.plot(time, sent_command, label=r"sent $\|u_{O,cmd}\|$")
    ax.plot(time, actual_speed, label=r"mocap $\|u_{O,actual}\|$")
    ax.plot(time, target_actual, label=r"target actual $\|u_T\|$")
    ax.set_title("Command and actual speeds")
    ax.set_ylabel("m/s")

    ax = axes[1, 0]
    ax.plot(time, target_estimate, label=r"$\|\hat u_T\|$")
    for x_name, y_name, label in (
        ("control_relative_velocity_x", "control_relative_velocity_y", r"$\|\hat\xi_v\|$"),
        ("control_trajectory_x", "control_trajectory_y", r"$\|-\dot r_d\|$"),
        ("control_feedback_x", "control_feedback_y", r"$\|K_r(\hat r-r_d)\|$"),
    ):
        ax.plot(time, vector_norm(data, x_name, y_name), label=label)
    ax.set_title("Terms forming the raw pursuer command")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("m/s")

    ax = axes[1, 1]
    condition = np.divide(
        data["history_max_eigenvalue"],
        data["history_min_active_eigenvalue"],
        out=np.full_like(time, np.nan),
        where=data["history_min_active_eigenvalue"] > 0.0,
    )
    ax.semilogy(time, condition, label="active-spectrum condition number")
    ax.semilogy(time, np.maximum(data["theta_hat_norm"], 1.0e-12), label=r"$\|\hat\theta\|$")
    ax2 = ax.twinx()
    ax2.step(time, data["active_directions"], where="post", color="tab:green",
             label="active directions")
    ax.set_title("History-stack conditioning and estimator growth")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("condition / norm (log)")
    ax2.set_ylabel("active directions")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize="small")

    mark_history_updates(axes.flat, time, data["history_count"])
    for ax in axes.flat:
        ax.grid(True, alpha=0.35)
        if ax is not axes[1, 1]:
            ax.legend(fontsize="small")

    figure.suptitle(
        f"Complete experiment diagnosis: {os.path.basename(os.path.dirname(args.csv_path))}"
    )
    figure.tight_layout()
    figure.savefig(args.output, dpi=170, bbox_inches="tight")
    print(os.path.abspath(args.output))


if __name__ == "__main__":
    main()
