import base64
import io
import os
import tempfile
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

app = FastAPI(title="Spleen Segmentation API", version="1.0.0")

model = None

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


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.filename.endswith(".nii.gz") or file.filename.endswith(".nii")):
        raise HTTPException(400, "Expected a .nii or .nii.gz CT volume (Task09_Spleen format)")

    suffix = ".nii.gz" if file.filename.endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        start = time.time()
        image = preprocess(tmp_path)
        input_tensor = image.unsqueeze(0).to(DEVICE)

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

        return JSONResponse({
            "inference_seconds": round(elapsed, 3),
            "volume_shape": list(pred.shape),
            "predicted_foreground_voxels": foreground_voxels,
            "foreground_fraction": round(foreground_voxels / total_voxels, 5),
            "preview_png_base64": preview_b64,
        })
    finally:
        os.unlink(tmp_path)
