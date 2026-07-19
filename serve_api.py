import base64
import io
import logging
import os
import tempfile
import threading
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, CropForeground, EnsureChannelFirst, EnsureType,
    LoadImage, Orientation, ScaleIntensityRange, Spacing,
)

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "checkpoints/best_spleen_model.pth")
PATCH_SIZE = (64, 64, 64)  # must match train_spleen.py / evaluate_spleen.py
DEVICE = torch.device("cpu")

# Real per-case intensity stats measured over the 33 training cases (post
# preprocessing, same pipeline as below) -- see commit history for how these
# were computed. Used as the drift-check reference, not guessed values.
TRAIN_INTENSITY_MEAN_OF_MEANS = 0.1757
TRAIN_INTENSITY_STD_OF_MEANS = 0.0456
DRIFT_Z_THRESHOLD = 3.0  # flag incoming volumes outside 3 sigma of training case means

logging.basicConfig(
    filename="api_requests.log", level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger("spleen_api")

app = FastAPI(title="Spleen Segmentation API", version="1.0.0")

model = None
_metrics_lock = threading.Lock()
metrics = {
    "request_count": 0,
    "success_count": 0,
    "error_count": 0,
    "drift_flag_count": 0,
    "total_latency_seconds": 0.0,
    "max_latency_seconds": 0.0,
}

preprocess = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    Orientation(axcodes="RAS"),
    Spacing(pixdim=(2.5, 2.5, 3.0), mode="bilinear"),
    ScaleIntensityRange(a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForeground(),
    EnsureType(),
])


@app.on_event("startup")
def load_model():
    global model
    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
    ).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    print(f"Loaded checkpoint: {CHECKPOINT_PATH}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "checkpoint": CHECKPOINT_PATH}


@app.get("/metrics")
def get_metrics():
    with _metrics_lock:
        success_count = metrics["success_count"]
        avg_latency = metrics["total_latency_seconds"] / success_count if success_count else 0.0
        return {
            "request_count": metrics["request_count"],
            "error_count": metrics["error_count"],
            "drift_flag_count": metrics["drift_flag_count"],
            "avg_latency_seconds": round(avg_latency, 3),
            "max_latency_seconds": round(metrics["max_latency_seconds"], 3),
        }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.filename.endswith(".nii.gz") or file.filename.endswith(".nii")):
        raise HTTPException(400, "Expected a .nii or .nii.gz CT volume (Task09_Spleen format)")

    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    start = time.time()
    try:
        image = preprocess(tmp_path)
        input_tensor = image.unsqueeze(0).to(DEVICE)

        intensity_mean = float(image.mean())
        drift_z = (intensity_mean - TRAIN_INTENSITY_MEAN_OF_MEANS) / TRAIN_INTENSITY_STD_OF_MEANS
        drift_flagged = abs(drift_z) > DRIFT_Z_THRESHOLD

        with torch.no_grad():
            output = sliding_window_inference(input_tensor, PATCH_SIZE, 1, model)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        elapsed = time.time() - start
        foreground_voxels = int((pred == 1).sum())
        total_voxels = int(pred.size)

        mid = pred.shape[-1] // 2
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image[0, :, :, mid].cpu().numpy(), cmap="gray")
        mask_slice = np.ma.masked_where(pred[:, :, mid] == 0, pred[:, :, mid])
        ax.imshow(mask_slice, cmap="autumn", alpha=0.5)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        preview_b64 = base64.b64encode(buf.getvalue()).decode()

        with _metrics_lock:
            metrics["request_count"] += 1
            metrics["success_count"] += 1
            metrics["total_latency_seconds"] += elapsed
            metrics["max_latency_seconds"] = max(metrics["max_latency_seconds"], elapsed)
            if drift_flagged:
                metrics["drift_flag_count"] += 1

        logger.info(
            "predict status=ok file=%s latency_s=%.3f intensity_mean=%.4f "
            "drift_z=%.2f drift_flagged=%s foreground_fraction=%.5f",
            file.filename, elapsed, intensity_mean, drift_z, drift_flagged,
            foreground_voxels / total_voxels,
        )

        return JSONResponse({
            "inference_seconds": round(elapsed, 3),
            "volume_shape": list(pred.shape),
            "predicted_foreground_voxels": foreground_voxels,
            "foreground_fraction": round(foreground_voxels / total_voxels, 5),
            "drift_check": {
                "intensity_mean": round(intensity_mean, 4),
                "reference_mean": TRAIN_INTENSITY_MEAN_OF_MEANS,
                "z_score": round(drift_z, 2),
                "flagged": drift_flagged,
            },
            "preview_png_base64": preview_b64,
        })
    except Exception as exc:
        with _metrics_lock:
            metrics["request_count"] += 1
            metrics["error_count"] += 1
        logger.info("predict status=error file=%s error=%s", file.filename, str(exc))
        raise
    finally:
        os.unlink(tmp_path)
