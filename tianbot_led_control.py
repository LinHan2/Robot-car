"""ROS 1 control of the two Tianbot LED arrays from YAML settings."""

import math
import time

import yaml


def load_led_commands(path):
    with open(path, encoding="utf-8") as stream:
        values = yaml.safe_load(stream) or {}
    settings = values.get("led_control", {})
    if not isinstance(settings, dict):
        raise ValueError("led_control must be a mapping")
    enabled = settings.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("led_control.enabled must be true or false")
    if not enabled:
        return [], 0.0
    timeout = settings.get("connection_timeout", 5.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("LED connection_timeout must be positive and finite")
    commands = []
    for name in ("tianbot1", "tianbot2"):
        car = settings.get(name)
        if not isinstance(car, dict):
            raise ValueError(f"led_control.{name} must be a mapping")
        topic = car.get("topic")
        if not isinstance(topic, str) or not topic.startswith("/"):
            raise ValueError(f"{name} LED topic must be an absolute ROS topic")
        rgb = car.get("rgb")
        if (not isinstance(rgb, list) or len(rgb) != 3
                or any(type(value) is not int or not 0 <= value <= 255 for value in rgb)):
            raise ValueError(f"{name} rgb must contain three integers in [0, 255]")
        brightness = car.get("brightness", 1.0)
        if (isinstance(brightness, bool) or not isinstance(brightness, (int, float))
                or not math.isfinite(brightness) or not 0 <= brightness <= 1):
            raise ValueError(f"{name} brightness must be in [0, 1]")
        color = [int(round(value * brightness)) for value in rgb]
        commands.append((topic, color * 16))
    if commands[0][0] == commands[1][0]:
        raise ValueError("the two LED topics must be different")
    return commands, float(timeout)


def publish_led_commands(commands, timeout):
    # Import only when controlling hardware so offline tools can load the config.
    import rospy
    from std_msgs.msg import Int32MultiArray

    publishers = [(topic, data, rospy.Publisher(
        topic, Int32MultiArray, queue_size=1, latch=True
    )) for topic, data in commands]
    deadline = time.monotonic() + timeout
    pending = list(publishers)
    while pending and not rospy.is_shutdown():
        for item in list(pending):
            topic, data, publisher = item
            if publisher.get_num_connections() > 0:
                publisher.publish(Int32MultiArray(data=data))
                rospy.loginfo("LED %s RGB=%s", topic, data[:3])
                pending.remove(item)
        if not pending or time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    time.sleep(0.3)
    for _, _, publisher in publishers:
        publisher.unregister()
    if pending:
        raise RuntimeError("No LED subscriber connected: " + ", ".join(item[0] for item in pending))
