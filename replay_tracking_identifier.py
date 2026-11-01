#!/usr/bin/env python3
"""Replay the bearing identifier/observer against a recorded real trajectory."""

import argparse
import json
import math
import os
from dataclasses import replace
from itertools import product

import numpy as np

from demo_2d_bearing_tracking_direct_v1 import (
    BearingTrackingController,
    MocapVelocityEstimator,
    Parameters,
)


def load_run(run_dir):
    with open(os.path.join(run_dir, "metadata.json"), encoding="utf-8") as stream:
        metadata = json.load(stream)
    data = np.genfromtxt(
        os.path.join(run_dir, "tracking_data.csv"),
        delimiter=",",
        names=True,
        invalid_raise=False,
    )
    data = data[np.isfinite(data["t"])]
    if data.size == 0:
        raise ValueError("tracking_data.csv contains no complete samples")
    return metadata, data


def parameters_from_metadata(metadata):
    values = metadata["algorithm_parameters"]
    return Parameters(
        dt=float(values["dt"]),
        delta_t=float(values["delta_t"]),
        rd=float(values["rd"]),
        wd=float(values["wd"]),
        phi0=float(values["phi0"]),
        max_linear_speed=float(values["max_linear_speed"]),
        tracking_gain=float(values["tracking_gain"]),
        q=int(values["q"]),
        sigma=float(values["sigma"]),
        nh=int(values["nh"]),
        max_history_execution_error=float(
            values.get("max_history_execution_error", 0.0)
        ),
        max_history_target_execution_error=float(
            values.get("max_history_target_execution_error", 0.0)
        ),
        history_forgetting_factor=float(
            values.get("history_forgetting_factor", 1.0)
        ),
        gamma=float(values["gamma"]),
        ac=float(values["ac"]),
        valid_tol=float(values["valid_tol"]),
        kc=float(values["kc"]),
        kh=float(values["kh"]),
        max_history_condition=float(values["max_history_condition"]),
        sigma_theta=float(values["sigma_theta"]),
        af=float(values["af"]),
        lambda_s=float(values["lambda_s"]),
        ro=float(values["ro"]),
        p_floor=float(values["p_floor"]),
        mocap_position_observer_gain=float(values["mocap_position_observer_gain"]),
        mocap_relative_velocity_observer_gain=float(
            values.get("mocap_relative_velocity_observer_gain", 0.0)
        ),
        identifier_use_relative_velocity=bool(
            values.get("identifier_use_relative_velocity", False)
        ),
    )


def replay(parameters, data, tail_seconds, velocity_window, velocity_filter_tau,
           max_relative_speed):
    first_bearing = np.array([data["bearing_x"][0], data["bearing_y"][0]])
    first_relative = np.array([data["relative_x"][0], data["relative_y"][0]])
    controller = BearingTrackingController(parameters, first_bearing, first_relative)
    input_errors = []
    localization_errors = []
    estimate_norms = []
    conditions = []
    gated = []
    identifier_current_rates = []
    identifier_history_rates = []
    identifier_leak_rates = []
    identifier_total_rates = []
    bearings = []

    relative_velocity_estimator = MocapVelocityEstimator(
        velocity_window, velocity_filter_tau, max_relative_speed
    )
    for index, row in enumerate(data):
        bearing = np.array([row["bearing_x"], row["bearing_y"]])
        relative = np.array([row["relative_x"], row["relative_y"]])
        pursuer_velocity = np.array([row["mocap_velocity_x"], row["mocap_velocity_y"]])
        target_velocity = np.array([
            row["target_mocap_velocity_x"],
            row["target_mocap_velocity_y"],
        ])
        target_execution_error = None
        target_command = np.array([
            row["target_command_world_x"],
            row["target_command_world_y"],
        ])
        if (np.all(np.isfinite(target_velocity))
                and np.all(np.isfinite(target_command))):
            target_execution_error = target_command - target_velocity
        relative_velocity = None
        if (parameters.identifier_use_relative_velocity
                or parameters.mocap_relative_velocity_observer_gain > 0.0):
            relative_velocity_estimator.add_pose(float(row["t"]), relative)
            relative_velocity = relative_velocity_estimator.estimate(parameters.dt)
        if parameters.identifier_use_relative_velocity:
            u_ident = pursuer_velocity + relative_velocity
        else:
            u_ident = pursuer_velocity
        _, diagnostics = controller.step(
            float(row["t"]),
            bearing,
            float(row["control_dt_used"]),
            pursuer_velocity,
            relative,
            u_ident=u_ident,
            relative_velocity_feedback=relative_velocity,
            target_execution_error=target_execution_error,
        )
        estimate = diagnostics["target_velocity_estimate"]
        input_errors.append(np.linalg.norm(target_velocity - estimate))
        localization_errors.append(
            np.linalg.norm(relative - diagnostics["relative_estimate"])
        )
        estimate_norms.append(np.linalg.norm(estimate))
        conditions.append(diagnostics["history_condition"])
        gated.append(diagnostics["history_update_gated"])
        identifier_current_rates.append(
            diagnostics["identifier_current_output_rate"]
        )
        identifier_history_rates.append(
            diagnostics["identifier_history_output_rate"]
        )
        identifier_leak_rates.append(diagnostics["identifier_leak_output_rate"])
        identifier_total_rates.append(diagnostics["identifier_total_output_rate"])
        bearings.append(bearing)

    times = np.asarray(data["t"], dtype=float)
    tail = times >= times[-1] - tail_seconds
    input_errors = np.asarray(input_errors)
    localization_errors = np.asarray(localization_errors)
    estimate_norms = np.asarray(estimate_norms)
    conditions = np.asarray(conditions)
    gated = np.asarray(gated)
    identifier_current_rates = np.asarray(identifier_current_rates)
    identifier_history_rates = np.asarray(identifier_history_rates)
    identifier_leak_rates = np.asarray(identifier_leak_rates)
    identifier_total_rates = np.asarray(identifier_total_rates)
    bearings = np.asarray(bearings)
    valid_conditions = conditions[tail & (conditions > 0.0)]
    phase = parameters.phi0 + parameters.wd * times[tail]
    harmonic_regressor = np.column_stack(
        (np.ones(np.sum(tail)), np.cos(phase), np.sin(phase))
    )
    harmonic_coefficients = np.linalg.lstsq(
        harmonic_regressor, input_errors[tail], rcond=None
    )[0]

    def drive_metrics(rates):
        tail_rates = rates[tail]
        tail_bearings = bearings[tail]
        tangential = (
            -tail_bearings[:, 1] * tail_rates[:, 0]
            + tail_bearings[:, 0] * tail_rates[:, 1]
        )
        return (
            float(np.sqrt(np.mean(np.sum(tail_rates ** 2, axis=1)))),
            float(np.sqrt(np.mean(tangential ** 2))),
        )

    current_drive_rms, current_drive_tangential_rms = drive_metrics(
        identifier_current_rates
    )
    history_drive_rms, history_drive_tangential_rms = drive_metrics(
        identifier_history_rates
    )
    leak_drive_rms, leak_drive_tangential_rms = drive_metrics(identifier_leak_rates)
    total_drive_rms, total_drive_tangential_rms = drive_metrics(identifier_total_rates)
    return {
        "input_rms": float(np.sqrt(np.mean(input_errors[tail] ** 2))),
        "localization_rms": float(np.sqrt(np.mean(localization_errors[tail] ** 2))),
        "input_peak": float(np.max(input_errors)),
        "localization_peak": float(np.max(localization_errors)),
        "estimate_peak": float(np.max(estimate_norms)),
        "input_harmonic_amplitude": float(
            np.hypot(harmonic_coefficients[1], harmonic_coefficients[2])
        ),
        "condition_median": (
            float(np.median(valid_conditions)) if valid_conditions.size else 0.0
        ),
        "condition_max": (
            float(np.max(valid_conditions)) if valid_conditions.size else 0.0
        ),
        "gated_percent": float(100.0 * np.mean(gated[tail])),
        "history_count": len(controller.history.windows),
        "active_directions": int(np.linalg.matrix_rank(controller.history.pi, tol=1.0e-8)),
        "current_drive_rms": current_drive_rms,
        "current_drive_tangential_rms": current_drive_tangential_rms,
        "history_drive_rms": history_drive_rms,
        "history_drive_tangential_rms": history_drive_tangential_rms,
        "leak_drive_rms": leak_drive_rms,
        "leak_drive_tangential_rms": leak_drive_tangential_rms,
        "total_drive_rms": total_drive_rms,
        "total_drive_tangential_rms": total_drive_tangential_rms,
    }


def parse_values(text):
    return [float(value) for value in text.split(",")]


def parse_int_values(text):
    return [int(value) for value in text.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--gammas", default="0.003,0.005,0.01")
    parser.add_argument("--condition-gates", default="1500,3000,5000,0")
    parser.add_argument("--history-gains", default="1.2")
    parser.add_argument("--current-window-gains", default=None)
    parser.add_argument("--filter-bandwidths", default=None)
    parser.add_argument("--rbf-counts", default=None)
    parser.add_argument("--rbf-sigmas", default=None)
    parser.add_argument("--windows", default=None)
    parser.add_argument("--history-sizes", default=None)
    parser.add_argument("--history-execution-gates", default=None)
    parser.add_argument("--tail-seconds", type=float, default=30.0)
    args = parser.parse_args()

    metadata, data = load_run(args.run_dir)
    base = parameters_from_metadata(metadata)
    run_arguments = metadata.get("arguments", {})
    velocity_window = float(run_arguments.get("mocap_velocity_window", 0.06))
    velocity_filter_tau = float(run_arguments.get("mocap_velocity_filter_tau", 0.02))
    max_relative_speed = 2.0 * float(
        run_arguments.get("mocap_max_valid_speed", 0.35)
    )
    current_window_gains = (
        parse_values(args.current_window_gains)
        if args.current_window_gains else [base.kc]
    )
    filter_bandwidths = (
        parse_values(args.filter_bandwidths)
        if args.filter_bandwidths else [base.af]
    )
    rbf_counts = parse_int_values(args.rbf_counts) if args.rbf_counts else [base.q]
    rbf_sigmas = parse_values(args.rbf_sigmas) if args.rbf_sigmas else [base.sigma]
    windows = parse_values(args.windows) if args.windows else [base.delta_t]
    history_sizes = (
        parse_int_values(args.history_sizes) if args.history_sizes else [base.nh]
    )
    history_execution_gates = (
        parse_values(args.history_execution_gates)
        if args.history_execution_gates else [base.max_history_execution_error]
    )
    results = []
    for (gamma, gate, history_gain, current_window_gain, filter_bandwidth,
         rbf_count, rbf_sigma, window, history_size, history_execution_gate) in product(
        parse_values(args.gammas),
        parse_values(args.condition_gates),
        parse_values(args.history_gains),
        current_window_gains,
        filter_bandwidths,
        rbf_counts,
        rbf_sigmas,
        windows,
        history_sizes,
        history_execution_gates,
    ):
        parameters = replace(
            base,
            gamma=gamma,
            max_history_condition=gate,
            kh=history_gain,
            kc=current_window_gain,
            af=filter_bandwidth,
            q=rbf_count,
            sigma=rbf_sigma,
            delta_t=window,
            nh=history_size,
            max_history_execution_error=history_execution_gate,
        )
        metrics = replay(
            parameters, data, args.tail_seconds, velocity_window,
            velocity_filter_tau, max_relative_speed
        )
        metrics.update(
            gamma=gamma,
            gate=gate,
            history_gain=history_gain,
            current_window_gain=current_window_gain,
            filter_bandwidth=filter_bandwidth,
            rbf_count=rbf_count,
            rbf_sigma=rbf_sigma,
            window=window,
            history_size=history_size,
            history_execution_gate=history_execution_gate,
        )
        results.append(metrics)
        print(
            "q=%-2d sigma=%.3g win=%-3g nh=%-2d exec=%-4g gamma=%-7g gate=%-6g "
            "kh=%-4g kc=%-4g af=%-4g "
            "input_rms=%.4f loc_rms=%.4f "
            "harmonic=%.4f input_peak=%.4f uhat_peak=%.4f "
            "drive(current/history/leak)=%.4f/%.4f/%.4f "
            "gated=%5.1f%% cond=%7.1f active=%d"
            % (
                rbf_count,
                rbf_sigma,
                window,
                history_size,
                history_execution_gate,
                gamma,
                gate,
                history_gain,
                current_window_gain,
                filter_bandwidth,
                metrics["input_rms"],
                metrics["localization_rms"],
                metrics["input_harmonic_amplitude"],
                metrics["input_peak"],
                metrics["estimate_peak"],
                metrics["current_drive_rms"],
                metrics["history_drive_rms"],
                metrics["leak_drive_rms"],
                metrics["gated_percent"],
                metrics["condition_median"],
                metrics["active_directions"],
            )
        )

    best = min(results, key=lambda item: (item["input_rms"], item["input_peak"]))
    print("\nBest recorded-trajectory replay:")
    print(json.dumps(best, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
