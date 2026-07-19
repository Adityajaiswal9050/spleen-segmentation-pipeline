#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class UndistortNode(Node):
    def __init__(self):
        super().__init__('undistort_node')
        data = np.load('/calib/calibration_results.npz')
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, '/camera/raw_image', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Image, '/camera/undistorted_image', 10)
        self.get_logger().info('Undistort node ready, using saved calibration')

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        undistorted = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeffs)
        out_msg = self.bridge.cv2_to_imgmsg(undistorted, encoding='bgr8')
        self.publisher_.publish(out_msg)
        self.get_logger().info('Published undistorted frame')

def main():
    rclpy.init()
    node = UndistortNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
