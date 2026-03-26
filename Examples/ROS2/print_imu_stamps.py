#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

class StampPrinter(Node):
    def __init__(self):
        super().__init__('stamp_printer')
        self.count = 0
        self.sub = self.create_subscription(Imu, '/imu0', self.cb, 100)

    def cb(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        print(f"IMU header.stamp = {t:.9f}")
        self.count += 1
        if self.count >= 50:
            rclpy.shutdown()

def main():
    rclpy.init()
    node = StampPrinter()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()