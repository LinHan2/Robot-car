import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recording_quality import filter_recording, save_quality_exports


def recording():
    times = np.arange(1, 101, dtype=float) * 0.02
    values = {"t": times.tolist()}
    for name in (
        "target_x", "target_y", "pursuer_x", "pursuer_y",
        "relative_x", "relative_y", "relative_estimate_x", "relative_estimate_y",
        "target_mocap_velocity_x", "target_mocap_velocity_y",
        "target_velocity_estimate_x", "target_velocity_estimate_y",
    ):
        values[name] = np.zeros(len(times)).tolist()
    for name, value in (("target_pose_age", 0.005), ("pursuer_pose_age", 0.005),
                        ("target_velocity_sample_span", 0.06),
                        ("pursuer_velocity_sample_span", 0.06),
                        ("control_dt_raw", 0.02), ("input_error", 0.04)):
        values[name] = [value] * len(times)
    return values


class RecordingQualityTests(unittest.TestCase):
    def test_sensor_gap_excludes_recovery_but_keeps_real_large_error(self):
        columns = recording()
        columns["target_pose_age"][50] = 0.03
        columns["target_velocity_sample_span"][52] = 0.02
        columns["input_error"][52] = 0.3
        columns["input_error"][20] = 10.0
        original = copy.deepcopy(columns)
        clean, plot, flags, report = filter_recording(columns, {})
        self.assertNotEqual(flags[50], 0)
        self.assertNotEqual(flags[52], 0)
        self.assertNotEqual(flags[55], 0)
        self.assertEqual(flags[20], 0)
        self.assertIn(10.0, clean["input_error"])
        self.assertNotIn(0.3, clean["input_error"])
        self.assertEqual(plot["t"], columns["t"])
        self.assertTrue(np.isnan(plot["input_error"][52]))
        self.assertEqual(columns, original)
        self.assertEqual(report["excluded_samples"], int((flags != 0).sum()))

    def test_healthy_startup_and_large_errors_are_not_removed(self):
        columns = recording()
        columns["target_velocity_sample_span"][0] = 0.01
        columns["input_error"][0] = 1.0
        clean, _, flags, _ = filter_recording(columns, {})
        self.assertFalse(flags.any())
        self.assertEqual(clean, columns)

    def test_timestamp_reset_counter_only_flags_the_reset_and_guard(self):
        columns = recording()
        columns["target_velocity_timestamp_resets"] = [0] * 50 + [1] * 50
        _, _, flags, _ = filter_recording(columns, {})
        self.assertTrue(flags[50] & 16)
        self.assertEqual(flags[-1], 0)

    def test_disabled_filter_preserves_every_row(self):
        columns = recording()
        columns["target_pose_age"][50] = 0.5
        clean, _, flags, _ = filter_recording(columns, {"recording_filter": False})
        self.assertFalse(flags.any())
        self.assertEqual(clean, columns)

    def test_export_rows_keep_original_timestamps_and_audit_reasons(self):
        columns = recording()
        columns["control_dt_raw"][50] = 0.08
        clean, _, flags, report = filter_recording(columns, {})
        with tempfile.TemporaryDirectory() as folder:
            save_quality_exports(folder, columns, clean, flags, report)
            with open(Path(folder) / "tracking_data_clean.csv") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([float(row["t"]) for row in rows], clean["t"])
            with np.load(Path(folder) / "tracking_data_clean.npz") as arrays:
                np.testing.assert_array_equal(arrays["t"], clean["t"])
            with open(Path(folder) / "recording_quality.csv") as stream:
                quality = list(csv.DictReader(stream))
            self.assertEqual(len(quality), len(columns["t"]))
            self.assertIn("control_timing_gap", quality[50]["reason"])
            saved = json.loads((Path(folder) / "recording_quality.json").read_text())
            self.assertEqual(saved, report)


if __name__ == "__main__":
    unittest.main()
