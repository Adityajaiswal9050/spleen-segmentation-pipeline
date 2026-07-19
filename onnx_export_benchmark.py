"""Export the trained checkpoint to ONNX, benchmark real inference latency
(PyTorch vs ONNX Runtime, CPU), and compare against the ROS2 pipeline's real
publish rate from Phase 4 -- connecting the model and robotics halves of the
project with an actual number instead of leaving them conceptually separate.
"""
import time

import numpy as np
import onnxruntime as ort
import torch

from monai.networks.nets import UNet

DEVICE = torch.device("cpu")
CHECKPOINT_PATH = "checkpoints/best_spleen_model.pth"
ONNX_PATH = "checkpoints/spleen_model.onnx"
PATCH_SIZE = (64, 64, 64)
NUM_WARMUP = 3
NUM_TIMED_RUNS = 20

# Real, measured publish rate of the Phase 4 ROS2 camera pipeline
# (image_publisher.py uses create_timer(1.0, ...) -- a fixed 1 Hz timer,
# independently confirmed live via `ros2 topic hz` / per-frame log timestamps
# during the Phase 3->4 integration work).
ROS2_PUBLISH_RATE_HZ = 1.0

model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

dummy_input = torch.rand(1, 1, *PATCH_SIZE)

print("Exporting to ONNX...")
torch.onnx.export(
    model, dummy_input, ONNX_PATH,
    input_names=["input"], output_names=["output"],
    opset_version=17,
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
print(f"Saved {ONNX_PATH}")

# ---- Verify ONNX output matches PyTorch output (correctness, not just speed) ----
ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
with torch.no_grad():
    torch_output = model(dummy_input).numpy()
onnx_output = ort_session.run(None, {"input": dummy_input.numpy()})[0]
max_abs_diff = np.abs(torch_output - onnx_output).max()
print(f"Max abs diff between PyTorch and ONNX Runtime output: {max_abs_diff:.2e}")
assert max_abs_diff < 1e-3, "ONNX export diverges from PyTorch output -- do not trust the latency numbers below"

# ---- Latency benchmark: PyTorch (CPU) ----
for _ in range(NUM_WARMUP):
    with torch.no_grad():
        model(dummy_input)

torch_times = []
for _ in range(NUM_TIMED_RUNS):
    start = time.perf_counter()
    with torch.no_grad():
        model(dummy_input)
    torch_times.append(time.perf_counter() - start)

# ---- Latency benchmark: ONNX Runtime (CPU) ----
input_np = dummy_input.numpy()
for _ in range(NUM_WARMUP):
    ort_session.run(None, {"input": input_np})

onnx_times = []
for _ in range(NUM_TIMED_RUNS):
    start = time.perf_counter()
    ort_session.run(None, {"input": input_np})
    onnx_times.append(time.perf_counter() - start)

torch_mean, torch_std = np.mean(torch_times), np.std(torch_times)
onnx_mean, onnx_std = np.mean(onnx_times), np.std(onnx_times)

print(f"\n=== Latency on a single {PATCH_SIZE} patch, CPU, {NUM_TIMED_RUNS} runs ===")
print(f"PyTorch:      {torch_mean*1000:.1f} ms +/- {torch_std*1000:.1f} ms  ({1/torch_mean:.2f} inferences/sec)")
print(f"ONNX Runtime: {onnx_mean*1000:.1f} ms +/- {onnx_std*1000:.1f} ms  ({1/onnx_mean:.2f} inferences/sec)")
speedup = torch_mean / onnx_mean
print(f"ONNX Runtime speedup: {speedup:.2f}x")

print(f"\n=== Compared to the real Phase 4 ROS2 pipeline ===")
print(f"Camera publish rate (measured, image_publisher.py's 1.0s timer): {ROS2_PUBLISH_RATE_HZ:.2f} Hz "
      f"(one frame every {1/ROS2_PUBLISH_RATE_HZ:.2f}s)")
print(f"PyTorch single-patch inference:      {torch_mean:.3f}s -- "
      f"{'keeps up with' if torch_mean < 1/ROS2_PUBLISH_RATE_HZ else 'falls behind'} the camera rate")
print(f"ONNX Runtime single-patch inference: {onnx_mean:.3f}s -- "
      f"{'keeps up with' if onnx_mean < 1/ROS2_PUBLISH_RATE_HZ else 'falls behind'} the camera rate")
print(
    "\nNote: this is a single 64^3 patch, not the sliding-window inference over a full CT "
    "volume that evaluate_spleen.py runs (which takes several seconds per full case -- see "
    "the API benchmark). The segmentation_bridge_node.py ROS2 node also does one patch-sized "
    "forward pass per camera frame, so this number is the actually relevant one for that node's "
    "per-frame latency budget, not the full-volume evaluation latency."
)
