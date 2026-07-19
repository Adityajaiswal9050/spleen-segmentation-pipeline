import os
import time
import torch
import numpy as np

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, EnsureTyped, AsDiscrete,
)

DEVICE = torch.device("cpu")
ROOT_DIR = "./data"
NUM_EPOCHS = 50          # small on purpose -- CPU training on a low-RAM VM
PATCH_SIZE = (64, 64, 64)   # smaller patch -> less peak memory

print("=== Phase 3: MONAI Spleen Segmentation Training ===")
print(f"Device: {DEVICE}")

# ---- Transforms ----
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(2.5, 2.5, 3.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE,
        pos=1, neg=1, num_samples=1, image_key="image", image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(2.5, 2.5, 3.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

# ---- Datasets (cache_rate=0.0 -> load from disk on demand, not into RAM) ----
train_ds = DecathlonDataset(
    root_dir=ROOT_DIR, task="Task09_Spleen", section="training",
    transform=train_transforms, download=False, cache_rate=0.0, num_workers=0,
)
val_ds = DecathlonDataset(
    root_dir=ROOT_DIR, task="Task09_Spleen", section="validation",
    transform=val_transforms, download=False, cache_rate=0.0, num_workers=0,
)

print(f"Training cases: {len(train_ds)}  |  Validation cases: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# ---- Model, loss, optimizer, metric ----
model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
).to(DEVICE)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

best_dice = -1.0
os.makedirs("checkpoints", exist_ok=True)

# ---- Training loop ----
for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    epoch_loss = 0
    for step, batch in enumerate(train_loader, start=1):
        inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"  epoch {epoch+1}/{NUM_EPOCHS}  step {step}/{len(train_loader)}  loss={loss.item():.4f}")

    epoch_loss /= max(step, 1)
    print(f"Epoch {epoch+1} average loss: {epoch_loss:.4f}  ({time.time()-epoch_start:.1f}s)")

    # ---- Validation ----
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs, val_labels = val_batch["image"].to(DEVICE), val_batch["label"].to(DEVICE)
            val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels_d = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs, y=val_labels_d)

        mean_dice = dice_metric.aggregate().item()
        dice_metric.reset()

    print(f"Epoch {epoch+1} validation mean Dice: {mean_dice:.4f}")

    if mean_dice > best_dice:
        best_dice = mean_dice
        torch.save(model.state_dict(), "checkpoints/best_spleen_model.pth")
        print(f"  -> New best model saved (Dice: {best_dice:.4f})")

print(f"\nTraining complete. Best validation Dice: {best_dice:.4f}")
print("Model saved to: checkpoints/best_spleen_model.pth")
