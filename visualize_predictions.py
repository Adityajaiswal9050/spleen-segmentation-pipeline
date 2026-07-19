import torch
import matplotlib.pyplot as plt

from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped,
)

DEVICE = torch.device("cpu")
ROOT_DIR = "./data"
PATCH_SIZE = (64, 64, 64)
CHECKPOINT_PATH = "checkpoints/best_spleen_model.pth"
NUM_CASES_TO_SHOW = 3   # how many validation cases to visualize

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

print(f"Generating prediction visualisations for {NUM_CASES_TO_SHOW} validation cases...")

with torch.no_grad():
    for case_idx, val_batch in enumerate(val_loader):
        if case_idx >= NUM_CASES_TO_SHOW:
            break

        val_inputs = val_batch["image"].to(DEVICE)
        val_labels = val_batch["label"].to(DEVICE)
        val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model)
        pred = torch.argmax(val_outputs, dim=1)[0]   # (H, W, D), values 0 or 1

        image = val_inputs[0, 0]      # (H, W, D)
        label = val_labels[0, 0]      # (H, W, D)

        # Pick the slice with the most ground-truth spleen pixels -- the most informative one.
        slice_sums = label.sum(dim=(0, 1))
        best_slice = int(torch.argmax(slice_sums).item())
        if slice_sums[best_slice] == 0:
            best_slice = label.shape[-1] // 2  # fallback: just use the middle slice

        img_slice = image[:, :, best_slice].numpy()
        gt_slice = label[:, :, best_slice].numpy()
        pred_slice = pred[:, :, best_slice].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_slice, cmap="gray")
        axes[0].set_title("CT slice")
        axes[0].axis("off")

        axes[1].imshow(img_slice, cmap="gray")
        axes[1].imshow(gt_slice, cmap="Reds", alpha=0.4)
        axes[1].set_title("Ground truth spleen")
        axes[1].axis("off")

        axes[2].imshow(img_slice, cmap="gray")
        axes[2].imshow(pred_slice, cmap="Blues", alpha=0.4)
        axes[2].set_title("Model prediction")
        axes[2].axis("off")

        out_path = f"prediction_case{case_idx+1}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"  Saved {out_path} (slice {best_slice})")

print("Done. Open the PNG files to see how the predictions compare to ground truth.")
