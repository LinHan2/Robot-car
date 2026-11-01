"""Mark recording artifacts using sensor support and timing, never error size."""

import csv
import json
from pathlib import Path

import numpy as np


QUALITY_REASONS = {
    1: "pose_update_gap",
    2: "velocity_window_too_short",
    4: "control_timing_gap",
    8: "nonfinite_measurement",
    16: "pose_timestamp_reset",
}


def reason_text(code):
    return "|".join(label for bit, label in QUALITY_REASONS.items() if code & bit)


def filter_recording(columns, arguments):
    """Return accepted rows, gap-preserving plot data, flags, and an audit report.

    Original columns are left intact. Missing intervals keep their original
    timestamps; plots show gaps instead of interpolated replacement values.
    """
    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    times = arrays["t"]
    count = len(times)
    if any(len(values) != count for values in arrays.values()):
        raise ValueError("recording columns have different sample counts")
    enabled = arguments.get("recording_filter", True)
    window = float(arguments.get("mocap_velocity_window", 0.06))
    max_pose_age = float(arguments.get("recording_max_pose_age", 0.02))
    min_span_ratio = float(arguments.get("recording_min_velocity_span_ratio", 0.5))
    recovery = float(arguments.get("recording_recovery_time", 0.15))
    nominal_dt = 1.0 / float(arguments.get("rate", 50.0))
    if (not all(np.isfinite(value) for value in (window, max_pose_age, min_span_ratio, recovery))
            or window <= 0 or max_pose_age <= 0 or not 0 < min_span_ratio <= 1 or recovery < 0):
        raise ValueError("invalid recording quality thresholds")
    seed = np.zeros(count, dtype=np.uint16)
    if enabled:
        for name in ("target_pose_age", "pursuer_pose_age"):
            seed[arrays[name] > max_pose_age] |= 1
        # Initial velocity-window warmup is part of the real startup transient.
        for name in ("target_velocity_sample_span", "pursuer_velocity_sample_span"):
            seed[(times >= 2 * window) & (arrays[name] < window * min_span_ratio)] |= 2
        seed[(arrays["control_dt_raw"] > 2 * nominal_dt)
             | (arrays["control_dt_raw"] <= 0)] |= 4
        if count > 1:
            seed[np.r_[False, np.diff(times) <= 0]] |= 4
        for name in (
            "t", "target_x", "target_y", "pursuer_x", "pursuer_y",
            "relative_x", "relative_y", "relative_estimate_x", "relative_estimate_y",
            "target_mocap_velocity_x", "target_mocap_velocity_y",
            "target_velocity_estimate_x", "target_velocity_estimate_y",
        ):
            seed[~np.isfinite(arrays[name])] |= 8
        for name in ("target_velocity_timestamp_resets", "pursuer_velocity_timestamp_resets"):
            if name in arrays:
                seed[np.diff(np.r_[0, arrays[name]]) > 0] |= 16

    flags = seed.copy()
    for index in np.flatnonzero(seed):
        guard = (times >= times[index] - window) & (times <= times[index] + recovery)
        flags[guard] |= seed[index]
    accepted = flags == 0
    clean = {name: values[accepted].tolist() for name, values in arrays.items()}
    plot = {name: values.copy() for name, values in arrays.items()}
    for name, values in plot.items():
        if name != "t":
            values[~accepted] = np.nan
    plot = {name: values.tolist() for name, values in plot.items()}

    excluded = np.flatnonzero(~accepted)
    groups = np.split(excluded, np.flatnonzero(np.diff(excluded) > 1) + 1)
    intervals = []
    for group in groups:
        if len(group):
            code = int(np.bitwise_or.reduce(flags[group]))
            intervals.append({
                "start_s": float(times[group[0]]), "end_s": float(times[group[-1]]),
                "samples": int(len(group)), "reason": reason_text(code),
            })
    report = {
        "enabled": bool(enabled),
        "method": "pose_support_and_timing_not_error_amplitude",
        "raw_samples": count, "accepted_samples": int(accepted.sum()),
        "excluded_samples": int((~accepted).sum()),
        "thresholds": {
            "max_pose_age_s": max_pose_age,
            "min_velocity_window_span_s": window * min_span_ratio,
            "max_control_interval_s": 2 * nominal_dt,
            "guard_before_s": window, "guard_after_s": recovery,
        },
        "excluded_intervals": intervals,
    }
    return clean, plot, flags, report


def save_quality_exports(output_dir, columns, clean, flags, report):
    output = Path(output_dir)
    names = list(columns)
    with (output / "tracking_data_clean.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(names)
        writer.writerows(zip(*(clean[name] for name in names)))
    np.savez_compressed(output / "tracking_data_clean.npz",
                        **{name: np.asarray(values) for name, values in clean.items()})
    with (output / "recording_quality.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_index", "t", "recording_valid", "reason"))
        writer.writerows((index, time, int(code == 0), reason_text(int(code)))
                         for index, (time, code) in enumerate(zip(columns["t"], flags)))
    with (output / "recording_quality.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
