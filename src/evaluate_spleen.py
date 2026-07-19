import os
import csv
import datetime
import torch

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped, AsDiscrete,
)

DEVICE = torch.device("cpu")
ROOT_DIR = "./data"
PATCH_SIZE = (64, 64, 64)
CHECKPOINT_PATH = "checkpoints/best_spleen_model.pth"
LOG_FILE = "results/benchmark_log.csv"

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(2.5, 2.5, 3.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

val_ds = DecathlonDataset(
    root_dir=ROOT_DIR, task="Task09_Spleen", section="validation",
    transform=val_transforms, download=False, cache_rate=0.0, num_workers=0,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

print(f"Evaluating checkpoint: {CHECKPOINT_PATH}")
print(f"Validation cases: {len(val_ds)}")

with torch.no_grad():
    for val_batch in val_loader:
        val_inputs, val_labels = val_batch["image"].to(DEVICE), val_batch["label"].to(DEVICE)
        val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model)
        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
        val_labels_d = [post_label(i) for i in decollate_batch(val_labels)]
        dice_metric(y_pred=val_outputs, y=val_labels_d)

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

timestamp = datetime.datetime.now().isoformat(timespec="seconds")
print(f"[{timestamp}] Validation mean Dice: {mean_dice:.4f}")

# ---- Log result (creates the CSV with a header the first time it's run) ----
file_exists = os.path.isfile(LOG_FILE)
with open(LOG_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["timestamp", "checkpoint", "num_val_cases", "mean_dice"])
    writer.writerow([timestamp, CHECKPOINT_PATH, len(val_ds), f"{mean_dice:.4f}"])

print(f"Logged result to {LOG_FILE}")
