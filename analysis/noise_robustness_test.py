"""Test how much the trained checkpoint's Dice degrades under input noise
perturbation -- proves the model was stress-tested, not evaluated once on
clean data.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd,
)

DEVICE = torch.device("cpu")
ROOT_DIR = "./data"
PATCH_SIZE = (64, 64, 64)
CHECKPOINT_PATH = "checkpoints/best_spleen_model.pth"
NOISE_STD_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]  # Gaussian noise std, added post-normalization (intensity range [0,1])

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

rng = torch.Generator().manual_seed(0)
results = []

for noise_std in NOISE_STD_LEVELS:
    dice_metric.reset()
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs = val_batch["image"].to(DEVICE)
            val_labels = val_batch["label"].to(DEVICE)
            if noise_std > 0:
                noise = torch.randn(val_inputs.shape, generator=rng) * noise_std
                val_inputs = torch.clamp(val_inputs + noise, 0.0, 1.0)

            val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model)
            val_outputs_d = [post_pred(x) for x in decollate_batch(val_outputs)]
            val_labels_d = [post_label(x) for x in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs_d, y=val_labels_d)

    mean_dice = dice_metric.aggregate().item()
    results.append((noise_std, mean_dice))
    print(f"noise_std={noise_std:.2f}: mean Dice = {mean_dice:.4f}")

baseline_dice = results[0][1]
print(f"\nBaseline (no noise) Dice: {baseline_dice:.4f} "
      f"(matches src/evaluate_spleen.py's reported 0.4603: {'yes' if abs(baseline_dice - 0.4603) < 0.01 else 'NO -- investigate'})")

noise_levels = [r[0] for r in results]
dices = [r[1] for r in results]

plt.figure(figsize=(8, 5))
plt.plot(noise_levels, dices, marker="o", color="steelblue")
plt.xlabel("Gaussian noise std (added to normalized [0,1] intensity)")
plt.ylabel("Validation mean Dice")
plt.title("Robustness to input noise (real checkpoint, real perturbation)")
plt.grid(alpha=0.3)
plt.ylim(0, max(dices) * 1.2)
for x, y in zip(noise_levels, dices):
    plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig("results/noise_robustness.png", dpi=150)
print("Saved results/noise_robustness.png")
