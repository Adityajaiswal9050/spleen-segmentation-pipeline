# ============================================================
# Spleen Segmentation - GPU Training (run this in Google Colab)
# ============================================================
# HOW TO USE:
# 1. Go to wandb.ai -> sign up free -> copy your API key (Settings -> API keys)
# 2. colab.research.google.com -> New notebook
# 3. Runtime -> Change runtime type -> T4 GPU -> Save
# 4. Paste this WHOLE file into a single cell -> Shift+Enter
# 5. It'll prompt for your W&B API key the first time -- paste it in.
# 6. It trains for ~1-2.5 hrs on a T4. Keep the tab open/active.
#    Watch live loss/Dice curves at wandb.ai under project "spleen-segmentation".
# 7. At the end it zips everything you need and downloads it.
#    Bring that zip to your Ubuntu VM and unzip into
#    ~/cv-project/monai-project/checkpoints/  (+ the plot/csv wherever you like)
# ============================================================

get_ipython().system('pip install -q monai nibabel wandb')

import os
import csv
import time
import shutil
import torch
import wandb
import matplotlib.pyplot as plt

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, RandGaussianNoised,
    EnsureTyped, AsDiscrete,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
assert DEVICE.type == "cuda", "No GPU detected -- go to Runtime > Change runtime type > T4 GPU"

ROOT_DIR = "./data"
NUM_EPOCHS = 300
PATCH_SIZE = (96, 96, 96)     # bigger than the CPU run -- GPU can afford it
VAL_EVERY = 5

os.makedirs("checkpoints", exist_ok=True)

# ---- W&B experiment tracking ----
wandb.login()
run = wandb.init(
    project="spleen-segmentation",
    config={
        "num_epochs": NUM_EPOCHS,
        "patch_size": PATCH_SIZE,
        "batch_size": 2,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "spacing": (1.5, 1.5, 2.0),
        "architecture": "UNet-5level",
        "channels": (16, 32, 64, 128, 256),
    },
)

# ---- Transforms (finer spacing + stronger augmentation than the CPU run) ----
train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE,
        pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
    RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
    EnsureTyped(keys=["image", "label"]),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    EnsureTyped(keys=["image", "label"]),
])

print("Downloading/loading Decathlon Task09_Spleen (first run only, ~1.5GB)...")
train_ds = DecathlonDataset(
    root_dir=ROOT_DIR, task="Task09_Spleen", section="training",
    transform=train_transforms, download=True, cache_rate=0.5, num_workers=2,
)
val_ds = DecathlonDataset(
    root_dir=ROOT_DIR, task="Task09_Spleen", section="validation",
    transform=val_transforms, download=False, cache_rate=1.0, num_workers=2,
)
print(f"Training cases: {len(train_ds)}  |  Validation cases: {len(val_ds)}")

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2,
).to(DEVICE)

loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
scaler = torch.cuda.amp.GradScaler()
dice_metric = DiceMetric(include_background=False, reduction="mean")

post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

best_dice = -1.0
history = []  # (epoch, train_loss, val_dice_or_blank)

print("=== Training start ===")
for epoch in range(NUM_EPOCHS):
    t0 = time.time()
    model.train()
    epoch_loss = 0.0
    for step, batch in enumerate(train_loader, start=1):
        inputs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
    epoch_loss /= max(step, 1)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    val_dice_str = ""
    if (epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == NUM_EPOCHS:
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_inputs = val_batch["image"].to(DEVICE)
                val_labels = val_batch["label"].to(DEVICE)
                val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 4, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels_d = [post_label(i) for i in decollate_batch(val_labels)]
                dice_metric(y_pred=val_outputs, y=val_labels_d)
            mean_dice = dice_metric.aggregate().item()
            dice_metric.reset()
        val_dice_str = f"{mean_dice:.4f}"
        is_best = mean_dice > best_dice
        if is_best:
            best_dice = mean_dice
            torch.save(model.state_dict(), "checkpoints/best_spleen_model.pth")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  loss={epoch_loss:.4f}  val_dice={mean_dice:.4f}  "
              f"best={best_dice:.4f}  ({time.time()-t0:.1f}s)")
        wandb.log({
            "epoch": epoch + 1, "train_loss": epoch_loss, "val_dice": mean_dice,
            "best_dice": best_dice, "lr": current_lr, "is_best": is_best,
        })
    else:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}  loss={epoch_loss:.4f}  ({time.time()-t0:.1f}s)")
        wandb.log({"epoch": epoch + 1, "train_loss": epoch_loss, "lr": current_lr})

    history.append((epoch + 1, epoch_loss, val_dice_str))

print(f"\nTraining complete. Best validation Dice: {best_dice:.4f}")
wandb.summary["best_dice"] = best_dice

# ---- Save training log CSV ----
with open("training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_dice"])
    writer.writerows(history)

# ---- Save training curve plot ----
epochs_list = [h[0] for h in history]
losses = [h[1] for h in history]
val_epochs = [h[0] for h in history if h[2] != ""]
val_dices = [float(h[2]) for h in history if h[2] != ""]

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(epochs_list, losses, color="tab:red", label="Train loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train loss", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.plot(val_epochs, val_dices, color="tab:blue", marker="o", label="Val Dice")
ax2.set_ylabel("Validation Dice", color="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title(f"Training curve (best Dice = {best_dice:.4f})")
fig.tight_layout()
plt.savefig("training_curve.png", dpi=130)
plt.close(fig)
print("Saved training_curve.png and training_log.csv")

# ---- Log final artifacts to W&B, then close the run ----
wandb.log({"training_curve": wandb.Image("training_curve.png")})
artifact = wandb.Artifact("best_spleen_model", type="model")
artifact.add_file("checkpoints/best_spleen_model.pth")
run.log_artifact(artifact)
print(f"W&B run URL: {run.url}")
wandb.finish()

# ---- Zip everything needed for the Ubuntu VM ----
shutil.make_archive("spleen_training_results", "zip", ".", "checkpoints/best_spleen_model.pth")
# also bundle the csv/png into the zip
import zipfile
with zipfile.ZipFile("spleen_training_results.zip", "a") as z:
    z.write("training_log.csv")
    z.write("training_curve.png")

print("\nZipped: spleen_training_results.zip")
print("Downloading now...")

from google.colab import files
files.download("spleen_training_results.zip")
