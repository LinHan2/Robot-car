#!/usr/bin/env python3
"""审计动捕日志是否满足 rho*b_dot = P_b(uT-uO)。"""

import argparse
import csv
import math

import numpy as np


def load_csv(path: str) -> dict:
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty tracking log: {path}")
    return {
        name: np.asarray([float(row[name]) for row in rows], dtype=float)
        for name in rows[0]
    }


def velocity_at(time: np.ndarray, velocity: np.ndarray,
                query_time: np.ndarray) -> np.ndarray:
    """线性插值同一时钟下的二维速度，并返回共同的有效掩码。"""
    return np.column_stack((
        np.interp(query_time, time, velocity[:, 0]),
        np.interp(query_time, time, velocity[:, 1]),
    ))


def residual_metrics(time: np.ndarray, relative: np.ndarray, target_velocity: np.ndarray,
                     pursuer_velocity: np.ndarray, shift: float,
                     steady_start: float) -> tuple:
    range_value = np.linalg.norm(relative, axis=1)
    bearing = relative / range_value[:, None]
    bearing_dot = np.column_stack((
        np.gradient(bearing[:, 0], time),
        np.gradient(bearing[:, 1], time),
    ))
    lhs = range_value[:, None] * bearing_dot
    query_time = time + shift
    valid = (
        (query_time >= time[0])
        & (query_time <= time[-1])
        & (time >= steady_start)
        & np.isfinite(lhs).all(axis=1)
    )
    target_shifted = velocity_at(time, target_velocity, query_time)
    pursuer_shifted = velocity_at(time, pursuer_velocity, query_time)
    pb = np.eye(2)[None, :, :] - np.einsum("ni,nj->nij", bearing, bearing)
    rhs = np.einsum("nij,nj->ni", pb, target_shifted - pursuer_shifted)
    residual = lhs - rhs
    tangent = np.column_stack((-bearing[:, 1], bearing[:, 0]))
    radial = np.sum(residual * bearing, axis=1)
    tangential = np.sum(residual * tangent, axis=1)
    return (
        math.sqrt(float(np.mean(np.sum(residual[valid] ** 2, axis=1)))),
        math.sqrt(float(np.mean(radial[valid] ** 2))),
        math.sqrt(float(np.mean(tangential[valid] ** 2))),
        int(np.sum(valid)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    parser.add_argument("--steady-start", type=float, default=55.0)
    parser.add_argument("--max-shift", type=float, default=0.15)
    parser.add_argument("--shift-step", type=float, default=0.002)
    parser.add_argument(
        "--raw-velocities", action="store_true",
        help="use unfiltered mocap velocity fits instead of controller inputs",
    )
    args = parser.parse_args()
    if args.max_shift < 0.0 or args.shift_step <= 0.0:
        raise ValueError("max-shift must be non-negative and shift-step must be positive")

    data = load_csv(args.csv_path)
    time = data["t"] - data["t"][0]
    relative = np.column_stack((data["relative_x"], data["relative_y"]))
    if args.raw_velocities:
        target_columns = ("target_raw_velocity_x", "target_raw_velocity_y")
        pursuer_columns = ("raw_mocap_velocity_x", "raw_mocap_velocity_y")
    else:
        target_columns = ("target_mocap_velocity_x", "target_mocap_velocity_y")
        pursuer_columns = ("mocap_velocity_x", "mocap_velocity_y")
    target_velocity = np.column_stack((data[target_columns[0]], data[target_columns[1]]))
    pursuer_velocity = np.column_stack((data[pursuer_columns[0]], data[pursuer_columns[1]]))
    shifts = np.arange(-args.max_shift, args.max_shift + args.shift_step / 2.0,
                       args.shift_step)
    results = [
        (shift,) + residual_metrics(
            time, relative, target_velocity, pursuer_velocity, shift,
            args.steady_start,
        )
        for shift in shifts
    ]
    best = min(results, key=lambda result: result[1])
    zero = min(results, key=lambda result: abs(result[0]))
    print("velocity source:", "raw fits" if args.raw_velocities else "controller inputs")
    print("steady interval: %.3f to %.3f s" % (args.steady_start, time[-1]))
    print("shift convention: positive means velocity evaluated later than bearing")
    print("zero shift: residual=%.5f radial=%.5f tangential=%.5f samples=%d" % zero[1:])
    print("best shift: %+0.3f s residual=%.5f radial=%.5f tangential=%.5f samples=%d" % best)
    print("nearby shifts:")
    for shift, total, radial, tangential, samples in results:
        if abs(shift - best[0]) <= 0.010 + 1.0e-12:
            print("  %+0.3f s total=%.5f radial=%.5f tangential=%.5f" % (
                shift, total, radial, tangential))


if __name__ == "__main__":
    main()