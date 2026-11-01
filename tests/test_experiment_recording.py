import csv
import math
import queue
import sys
import tempfile
import threading
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import demo_2d_bearing_tracking_direct_v1 as experiment
import tianbot_led_control


class ExperimentRecordingTests(unittest.TestCase):
    def make_node(self, duration=180.0, start_time=100.0):
        node = experiment.BearingTrackingNode.__new__(experiment.BearingTrackingNode)
        node.args = SimpleNamespace(duration=duration)
        node.start_time = start_time
        node.experiment_complete = threading.Event()
        node.stopped = False
        node.target = object()
        node.pursuer = object()
        node._fresh = Mock(return_value=False)
        node.reset_all_velocity_measurements = Mock()
        node.reset_bearing_alignment = Mock()
        node.publish_target_velocity = Mock()
        node.publish_velocity = Mock()
        return node

    def test_deadline_stops_both_cars_before_another_motion_update(self):
        node = self.make_node()
        with patch.object(experiment.time, "monotonic", return_value=280.0), \
                patch.object(experiment.rospy, "loginfo"):
            node.control_callback()
        self.assertTrue(node.experiment_complete.is_set())
        node._fresh.assert_not_called()
        for publisher in (node.publish_target_velocity, node.publish_velocity):
            publisher.assert_called_once()
            np.testing.assert_array_equal(publisher.call_args.args[0], [0.0, 0.0])

    def test_waiting_for_poses_and_manual_mode_do_not_expire(self):
        for duration, start, now in ((180.0, None, 10000.0),
                                     (180.0, 100.0, 279.99),
                                     (0.0, 100.0, 10000.0)):
            with self.subTest(duration=duration, start=start, now=now):
                node = self.make_node(duration, start)
                with patch.object(experiment.time, "monotonic", return_value=now), \
                        patch.object(experiment.rospy, "logwarn_throttle"):
                    node.control_callback()
                self.assertFalse(node.experiment_complete.is_set())

    def test_main_cleans_up_before_shutting_down_ros(self):
        with patch.object(sys, "argv", ["experiment", "--no-leds"]):
            args = experiment.parse_args()
        events = []
        node = Mock()
        node.experiment_complete.wait.return_value = True
        node.stop.side_effect = lambda: events.append("stop_and_save")
        with patch.object(experiment, "parse_args", return_value=args), \
                patch.object(experiment, "BearingTrackingNode", return_value=node), \
                patch.object(experiment.rospy, "init_node"), \
                patch.object(experiment.rospy, "is_shutdown", return_value=False), \
                patch.object(experiment.rospy, "signal_shutdown",
                             side_effect=lambda *_: events.append("ros_shutdown")), \
                patch.object(experiment.signal, "signal"), \
                patch.object(tianbot_led_control, "publish_led_commands"):
            experiment.main()
        self.assertEqual(events, ["stop_and_save", "ros_shutdown"])

    def test_duration_cli_overrides_yaml(self):
        for duration in (0.0, 240.0):
            with patch.object(sys, "argv", ["experiment", "--duration", str(duration)]):
                self.assertEqual(experiment.parse_args().duration, duration)

    def test_full_and_reduced_exports_keep_every_sample(self):
        samples = queue.Queue()
        for index in range(3):
            values = {field.name: 0.0 for field in fields(experiment.TelemetrySample)}
            values.update(
                t=(index + 1) * 0.02, control_dt_raw=0.02, control_dt_used=0.02,
                target_x=1.0, target_y=2.0, target_yaw=-0.2, pursuer_yaw=0.3,
                relative_x=1.0, relative_y=2.0,
                relative_estimate_x=0.9, relative_estimate_y=1.9,
                desired_relative_x=0.7, desired_relative_y=0.4,
                target_mocap_velocity_x=0.3, target_mocap_velocity_y=0.4,
                target_velocity_estimate_x=0.1, target_velocity_estimate_y=0.2,
                input_error=math.hypot(0.2, 0.2),
                localization_error=math.hypot(0.1, 0.1),
                tracking_error=math.hypot(0.3, 1.6),
            )
            samples.put(experiment.TelemetrySample(**values))
        with tempfile.TemporaryDirectory() as output:
            recorder = experiment.DiagnosticsRecorderWorker(
                samples, output, {}, None, save_figures=False
            )
            recorder.stop()
            recorder.start()
            recorder.join(timeout=5)
            self.assertFalse(recorder.is_alive())
            with open(Path(output) / "errors_and_positions.csv") as stream:
                reduced = list(csv.DictReader(stream))
            with open(Path(output) / "tracking_data.csv") as stream:
                full = list(csv.DictReader(stream))
            self.assertEqual(len(reduced), 3)
            self.assertEqual(len(full), 3)
            with np.load(Path(output) / "tracking_data.npz") as arrays:
                for name in reduced[0]:
                    np.testing.assert_array_equal(
                        [float(row[name]) for row in reduced], arrays[name]
                    )
                    self.assertEqual([row[name] for row in reduced],
                                     [row[name] for row in full])


if __name__ == "__main__":
    unittest.main()
