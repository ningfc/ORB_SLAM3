#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image


class TimeComparator(Node):
    def __init__(self):
        super().__init__('time_comparator')
        self.declare_parameter('imu_topic', '/imu0')
        self.declare_parameter('image_topic', '/cam0/image_raw')
        self.declare_parameter('samples', 100)
        self.declare_parameter('warn_threshold', 0.02)  # seconds

        imu_topic = self.get_parameter('imu_topic').get_parameter_value().string_value
        img_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.max_samples = self.get_parameter('samples').get_parameter_value().integer_value
        self.warn_threshold = self.get_parameter('warn_threshold').get_parameter_value().double_value

        self.last_imu_time = None
        self.imu_count = 0
        self.img_count = 0

        self.create_subscription(Imu, imu_topic, self.imu_cb, 200)
        self.create_subscription(Image, img_topic, self.img_cb, 10)

    def to_seconds(self, stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def imu_cb(self, msg: Imu):
        t = self.to_seconds(msg.header.stamp)
        self.last_imu_time = t
        self.imu_count += 1
        if self.imu_count <= 5:
            print(f"IM U[{self.imu_count}] header.stamp = {t:.9f}")

    def img_cb(self, msg: Image):
        t_img = self.to_seconds(msg.header.stamp)
        self.img_count += 1
        imu_t = self.last_imu_time
        if imu_t is None:
            delta = None
        else:
            delta = t_img - imu_t
        print(f"IM G[{self.img_count}] stamp={t_img:.9f}  IMU_latest={imu_t if imu_t is not None else 'None'}  delta={delta if delta is not None else 'N/A'}")
        if delta is not None and abs(delta) > self.warn_threshold:
            print(f"  WARNING: |delta|={abs(delta):.6f}s > warn_threshold={self.warn_threshold}s")

        if self.img_count >= self.max_samples:
            print('Reached max samples, shutting down.')
            rclpy.shutdown()
        self.imu_count = 0  # reset IMU count for next image


def main():
    rclpy.init()
    node = TimeComparator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
