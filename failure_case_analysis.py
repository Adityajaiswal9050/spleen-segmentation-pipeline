"""Honest failure-case analysis: run the real checkpoint on every validation
case, keep the actual per-case Dice (not just the aggregate mean reported by
evaluate_spleen.py), and save visualizations of the worst cases so they can
be inspected and explained honestly -- not cherry-picked good results.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from monai.apps import DecathlonDataset
from monai.data import decollate_batch
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
NUM_WORST_TO_VISUALIZE = 3

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

model = UNet(
    spatial_dims=3, in_channels=1, out_channels=2,
    channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

dice_metric = DiceMetric(include_background=False, reduction="mean")
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

results = []

print(f"Evaluating {len(val_ds)} validation cases individually...")
with torch.no_grad():
    for i in range(len(val_ds)):
        data = val_ds[i]
        image = data["image"].unsqueeze(0).to(DEVICE)
        label = data["label"].unsqueeze(0).to(DEVICE)

        output = sliding_window_inference(image, PATCH_SIZE, 1, model)
        pred_d = [post_pred(x) for x in decollate_batch(output)]
        label_d = [post_label(x) for x in decollate_batch(label)]
        dice_metric(y_pred=pred_d, y=label_d)
        case_dice = dice_metric.aggregate().item()
        dice_metric.reset()

        spleen_voxels = int((data["label"][0] > 0).sum().item())
        image_mean_intensity = float(data["image"][0][data["label"][0] > 0].mean().item()) if spleen_voxels > 0 else float("nan")

        results.append({
            "index": i,
            "dice": case_dice,
            "spleen_voxels": spleen_voxels,
            "spleen_mean_intensity": image_mean_intensity,
            "volume_shape": tuple(data["image"].shape[1:]),
        })
        print(f"  case {i}: Dice={case_dice:.4f}  spleen_voxels={spleen_voxels}  "
              f"mean_intensity_in_spleen={image_mean_intensity:.3f}")

results.sort(key=lambda r: r["dice"])
print("\nWorst cases (lowest Dice, real, not cherry-picked):")
for r in results[:NUM_WORST_TO_VISUALIZE]:
    print(f"  case {r['index']}: Dice={r['dice']:.4f}")

all_dice = [r["dice"] for r in results]
print(f"\nRecomputed per-case mean Dice: {np.mean(all_dice):.4f} "
      f"(evaluate_spleen.py's aggregate metric over the same 8 cases: 0.4603)")

for r in results[:NUM_WORST_TO_VISUALIZE]:
    i = r["index"]
    data = val_ds[i]
    image, label = data["image"], data["label"]
    input_tensor = image.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = sliding_window_inference(input_tensor, PATCH_SIZE, 1, model)
        pred = torch.argmax(output, dim=1)[0]

    fg = (label[0] > 0).nonzero(as_tuple=False)
    mid = int(fg[:, 2].float().mean().item()) if fg.numel() > 0 else image.shape[-1] // 2
    mid = max(0, min(mid, image.shape[-1] - 1))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(image[0, :, :, mid].numpy(), cmap="gray")
    axes[0].set_title(f"case {i} -- CT slice")
    axes[0].axis("off")

    axes[1].imshow(image[0, :, :, mid].numpy(), cmap="gray")
    axes[1].imshow(label[0, :, :, mid].numpy(), cmap="Reds", alpha=0.4)
    axes[1].set_title("Ground truth")
    axes[1].axis("off")

    axes[2].imshow(image[0, :, :, mid].numpy(), cmap="gray")
    axes[2].imshow(pred[:, :, mid].numpy(), cmap="Blues", alpha=0.4)
    axes[2].set_title(f"Prediction (Dice={r['dice']:.3f})")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = f"failure_case_{i}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
