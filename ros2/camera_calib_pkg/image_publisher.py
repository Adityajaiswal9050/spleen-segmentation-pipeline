#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/raw_image', 10)
        self.bridge = CvBridge()
        self.image = cv2.imread('/calib/images/left01.jpg')
        if self.image is None:
            self.get_logger().error('Could not load test image')
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.count = 0

    def timer_callback(self):
        if self.image is not None:
            msg = self.bridge.cv2_to_imgmsg(self.image, encoding='bgr8')
            self.publisher_.publish(msg)
            self.count += 1
            self.get_logger().info(f'Published frame {self.count}')

def main():
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
