#!/usr/bin/env python3
"""ROS2 node that closes the Phase 3 <-> Phase 4 gap: subscribes to the
undistorted camera topic, runs every frame through the trained MONAI 3D U-Net
checkpoint, and republishes a visualized result.

HONEST CAVEAT (do not remove): the model was trained on 3D abdominal CT
volumes (Task09_Spleen, Hounsfield-unit intensities, ~1x1x1mm anatomical
scans). This node's input is 2D RGB video of a chessboard calibration target
-- a completely different domain. To keep the *pipeline* real (real model,
real weights, real forward pass, real publish loop) each 2D frame is resized
and tiled into a synthetic depth axis so its shape matches what the network
expects. The resulting "segmentation" is NOT a meaningful clinical
prediction -- it demonstrates that Phase 3 (the trained model) and Phase 4
(the ROS2 graph) are now structurally wired together, nothing more.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from monai.networks.nets import UNet

CHECKPOINT_PATH = "/monai_project/checkpoints/best_spleen_model.pth"
PATCH_SIZE = (64, 64, 64)  # (D, H, W) -- must match train_spleen.py / evaluate_spleen.py


class SegmentationBridgeNode(Node):
    def __init__(self):
        super().__init__('segmentation_bridge_node')

        self.bridge = CvBridge()
        self.device = torch.device("cpu")

        self.model = UNet(
            spatial_dims=3, in_channels=1, out_channels=2,
            channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
        ).to(self.device)
        self.model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=self.device))
        self.model.eval()

        self.subscription = self.create_subscription(
            Image, '/camera/undistorted_image', self.listener_callback, 10)
        self.publisher_ = self.create_publisher(Image, '/camera/segmentation_overlay', 10)

        self.get_logger().warn(
            "segmentation_bridge_node: loaded a 3D CT spleen model but is running it on a "
            "2D camera feed. This proves the pipeline is wired end-to-end (Phase 3 -> Phase 4) "
            "-- the output masks carry no real anatomical meaning for this input."
        )
        self.get_logger().info(f"Loaded checkpoint: {CHECKPOINT_PATH}")

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (PATCH_SIZE[2], PATCH_SIZE[1]))
        normalized = resized.astype(np.float32) / 255.0

        # Synthetic depth axis: tile the single 2D slice D times so the tensor
        # shape matches what the 3D network expects. See module docstring.
        volume = np.tile(normalized[None, :, :], (PATCH_SIZE[0], 1, 1))
        input_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (D, H, W)

        mid_slice_mask = pred[PATCH_SIZE[0] // 2]  # take the middle depth slice
        mask_resized = cv2.resize(mid_slice_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        overlay = frame.copy()
        overlay[mask_resized == 1] = (0, 0, 255)  # red where the model predicts foreground
        blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

        out_msg = self.bridge.cv2_to_imgmsg(blended, encoding='bgr8')
        self.publisher_.publish(out_msg)
        self.get_logger().info(
            f'Published segmentation overlay (foreground px: {int((mask_resized == 1).sum())}/{h*w})'
        )


def main():
    rclpy.init()
    node = SegmentationBridgeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
