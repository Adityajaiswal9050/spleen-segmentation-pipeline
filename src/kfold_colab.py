# ============================================================
# Spleen Segmentation - K-Fold Cross-Validation (run this in Google Colab)
# ============================================================
# HOW TO USE:
# 1. colab.research.google.com -> New notebook
# 2. Runtime -> Change runtime type -> T4 GPU -> Save
# 3. Paste this WHOLE file into a single cell -> Shift+Enter
# 4. It trains 3 folds x 50 epochs each (~10-20 min total on a T4, much
#    faster than the CPU run this mirrors). Keep the tab open/active.
# 5. At the end it zips everything and downloads it.
#    Bring that zip to your Ubuntu VM and unzip into
#    ~/cv-project/monai-project/experiments/kfold_gpu/
# ============================================================
#
# WHY THIS SCRIPT EXISTS: kfold_cross_validation.py (CPU version) had to cut
# corners -- 15 epochs/fold instead of the production run's 50 -- because 3x
# full training runs on this VM's CPU would take hours and previously caused
# a hard reset. This script keeps the exact same architecture, patch size,
# spacing, and hyperparameters as src/train_spleen.py (the CPU baseline that
# scored val Dice 0.4603) and just moves execution to GPU so each fold can
# run the FULL 50 epochs. The question being answered: was 0.4603 a lucky
# train/val split, or representative of what this architecture/recipe can
# do? Report the per-fold Dice and the std across folds honestly, whatever
# they are -- this does not produce a new production checkpoint, it's a
# robustness signal for the existing one.

get_ipython().system('pip install -q monai nibabel')

import os
import csv
import time
import shutil
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from monai.apps import DecathlonDataset
from monai.data import DataLoader, Dataset, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, Orientationd, RandCropByPosNegLabeld, RandFlipd,
    RandRotate90d, ScaleIntensityRanged, Spacingd,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
assert DEVICE.type == "cuda", "No GPU detected -- go to Runtime > Change runtime type > T4 GPU"

ROOT_DIR = "./data"
PATCH_SIZE = (64, 64, 64)     # same as src/train_spleen.py -- this is a robustness
                              # check on the existing recipe, not a new one
K_FOLDS = 3
NUM_EPOCHS = 50               # full budget, same as src/train_spleen.py (CPU
                              # k-fold had to cut this to 15 for time)

# ---- Pull all 41 labeled cases (train+val sections combined) via the
# Decathlon downloader, same source src/train_spleen.py and the CPU k-fold used ----
print("Downloading/loading Decathlon Task09_Spleen (first run only, ~1.5GB)...")
full_train_ds_raw = DecathlonDataset(
    root_dir=ROOT_DIR, task="Task09_Spleen", section="training",
    transform=None, download=True, cache_rate=0.0, num_workers=0,
)
data_dicts = list(full_train_ds_raw.data)
print(f"Total labeled cases: {len(data_dicts)}")

rng = np.random.RandomState(0)
indices = rng.permutation(len(data_dicts))
folds = np.array_split(indices, K_FOLDS)

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

fold_results = []
fold_histories = []

print("=== K-fold cross-validation start (full 50-epoch budget per fold, GPU) ===")
for fold_idx in range(K_FOLDS):
    val_idx = folds[fold_idx]
    train_idx = np.concatenate([folds[i] for i in range(K_FOLDS) if i != fold_idx])

    train_files = [data_dicts[i] for i in train_idx]
    val_files = [data_dicts[i] for i in val_idx]
    print(f"\n=== Fold {fold_idx + 1}/{K_FOLDS}: {len(train_files)} train / {len(val_files)} val cases ===")

    train_ds = Dataset(data=train_files, transform=train_transforms)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    # Same architecture as src/train_spleen.py / checkpoints/best_spleen_model.pth
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
    history = []
    fold_start = time.time()
    for epoch in range(NUM_EPOCHS):
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
        epoch_loss /= max(step, 1)

        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(DEVICE)
                val_labels = val_batch["label"].to(DEVICE)
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model)
                val_outputs_d = [post_pred(x) for x in decollate_batch(val_outputs)]
                val_labels_d = [post_label(x) for x in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs_d, y=val_labels_d)
            mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()

        if mean_dice > best_dice:
            best_dice = mean_dice
        history.append((epoch + 1, epoch_loss, mean_dice))
        print(f"  fold {fold_idx+1} epoch {epoch+1}/{NUM_EPOCHS}: loss={epoch_loss:.4f} val_dice={mean_dice:.4f}")

    fold_time = time.time() - fold_start
    print(f"Fold {fold_idx + 1} done in {fold_time:.0f}s -- best Dice: {best_dice:.4f}")
    fold_results.append(best_dice)
    fold_histories.append(history)

print("\n=== K-fold cross-validation summary ===")
for i, d in enumerate(fold_results):
    print(f"Fold {i+1}: best Dice = {d:.4f}")
mean_dice_all = float(np.mean(fold_results))
std_dice_all = float(np.std(fold_results))
print(f"Mean across folds: {mean_dice_all:.4f}")
print(f"Std across folds:  {std_dice_all:.4f}")
print(
    f"\n(Full 50-epoch budget per fold, matching src/train_spleen.py exactly. "
    f"Compare against the production checkpoint's single-split 0.4603 -- "
    f"this shows whether that number was representative or a lucky split.)"
)

# ---- Save fold results CSV ----
with open("kfold_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "best_val_dice"])
    for i, d in enumerate(fold_results):
        writer.writerow([i + 1, f"{d:.4f}"])
    writer.writerow(["mean", f"{mean_dice_all:.4f}"])
    writer.writerow(["std", f"{std_dice_all:.4f}"])

# ---- Save per-fold training curves CSV ----
with open("kfold_training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["fold", "epoch", "train_loss", "val_dice"])
    for fold_idx, history in enumerate(fold_histories):
        for epoch, loss, dice in history:
            writer.writerow([fold_idx + 1, epoch, loss, dice])

# ---- Plot: bar chart of best Dice per fold + baseline reference line ----
plt.figure(figsize=(7, 5))
folds_x = [f"Fold {i+1}" for i in range(K_FOLDS)]
plt.bar(folds_x, fold_results, color="steelblue")
plt.axhline(0.4603, color="darkorange", linestyle="--", label="Production checkpoint (single split): 0.4603")
plt.axhline(mean_dice_all, color="green", linestyle=":", label=f"Mean across folds: {mean_dice_all:.4f}")
for i, d in enumerate(fold_results):
    plt.annotate(f"{d:.4f}", (i, d), textcoords="offset points", xytext=(0, 5), ha="center")
plt.ylabel("Best validation Dice")
plt.title(f"3-fold CV, full 50-epoch budget (mean={mean_dice_all:.4f}, std={std_dice_all:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig("kfold_results.png", dpi=150)
print("Saved kfold_results.png, kfold_results.csv, kfold_training_log.csv")

# ---- Zip everything needed for the Ubuntu VM ----
with __import__("zipfile").ZipFile("kfold_results.zip", "w") as z:
    z.write("kfold_results.csv")
    z.write("kfold_training_log.csv")
    z.write("kfold_results.png")

print("\nZipped: kfold_results.zip")
print("Downloading now...")

from google.colab import files
files.download("kfold_results.zip")
