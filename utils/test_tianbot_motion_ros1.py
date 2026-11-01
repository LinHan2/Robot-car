#!/usr/bin/env python3
"""Send a short, continuous ROS 1 velocity command to a Tianbot chassis."""

import argparse

import rospy
from geometry_msgs.msg import Twist


def make_velocity(linear_x: float, linear_y: float, angular_z: float) -> Twist:
    message = Twist()
    message.linear.x = linear_x
    message.linear.y = linear_y
    message.angular.z = angular_z
    return message


def main() -> None:
    parser = argparse.ArgumentParser(description="Tianbot ROS 1 motion test")
    parser.add_argument("--topic", default="/tianbot/cmd_vel")
    parser.add_argument("--linear-x", type=float, default=0.1)
    parser.add_argument("--linear-y", type=float, default=0.0)
    parser.add_argument("--angular-z", type=float, default=0.0)
    parser.add_argument("--rate", type=float, default=10.0)
    args = parser.parse_args(rospy.myargv()[1:])

    if args.rate <= 0.0:
        parser.error("--rate must be positive")

    rospy.init_node("tianbot_motion_test")
    publisher = rospy.Publisher(args.topic, Twist, queue_size=10)
    rate = rospy.Rate(args.rate)
    command = make_velocity(args.linear_x, args.linear_y, args.angular_z)
    stop_command = Twist()

    def stop() -> None:
        publisher.publish(stop_command)

    rospy.on_shutdown(stop)
    rospy.loginfo(
        "Publishing Twist to %s: linear=(%.3f, %.3f), angular.z=%.3f",
        args.topic,
        args.linear_x,
        args.linear_y,
        args.angular_z,
    )

    while not rospy.is_shutdown():
        publisher.publish(command)
        rate.sleep()


if __name__ == "__main__":
    main()
