"""Monte Carlo dropout uncertainty quantification.

HONEST CAVEAT (do not remove): checkpoints/best_spleen_model.pth was trained
with dropout=0.0 (see train_spleen.py) -- the original training run used no
dropout at all. Dropout has no learned parameters, so here we load the same
trained weights into a UNet built with dropout=0.2 and enable dropout only at
inference time ("test-time dropout injection"). This is a real, commonly used
post-hoc technique for retrofitting MC-dropout uncertainty onto an already
-trained deterministic model -- but it means the stochasticity comes from a
randomly-initialized-at-inference dropout mask, not from anything the network
learned to be robust to during training. Treat the resulting confidence maps
as a genuine but weaker uncertainty signal than a model trained with dropout
from scratch would give.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from monai.apps import DecathlonDataset
from monai.networks.nets import UNet
from monai.transforms import (
    Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, Orientationd, ScaleIntensityRanged, Spacingd,
)

DEVICE = torch.device("cpu")
ROOT_DIR = "./data"
PATCH_SIZE = (64, 64, 64)
CHECKPOINT_PATH = "checkpoints/best_spleen_model.pth"
INFERENCE_DROPOUT = 0.2
NUM_MC_PASSES = 20
CASE_INDICES = [0, 1, 2]

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
    dropout=INFERENCE_DROPOUT,
).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()  # everything (InstanceNorm, etc.) stays in eval mode...


def enable_dropout_only(m):
    for module in m.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def extract_patch_around_label(image, label, patch_size):
    fg = (label[0] > 0).nonzero(as_tuple=False)
    if fg.numel() == 0:
        center = [s // 2 for s in label.shape[1:]]
    else:
        center = fg.float().mean(dim=0).round().long().tolist()

    starts, ends, pads = [], [], []
    for c, sz, dim in zip(center, patch_size, image.shape[1:]):
        start = max(0, min(c - sz // 2, max(dim - sz, 0)))
        end = min(dim, start + sz)
        starts.append(start)
        ends.append(end)
        pads.append(sz - (end - start))

    img_patch = image[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]
    lbl_patch = label[:, starts[0]:ends[0], starts[1]:ends[1], starts[2]:ends[2]]

    if any(p > 0 for p in pads):
        pad_spec = (0, pads[2], 0, pads[1], 0, pads[0])
        img_patch = F.pad(img_patch, pad_spec)
        lbl_patch = F.pad(lbl_patch, pad_spec)

    return img_patch, lbl_patch


for case_idx in CASE_INDICES:
    data = val_ds[case_idx]
    image, label = data["image"], data["label"]
    img_patch, lbl_patch = extract_patch_around_label(image, label, PATCH_SIZE)
    input_tensor = img_patch.unsqueeze(0).to(DEVICE)

    enable_dropout_only(model)  # ...except Dropout, turned on for this loop
    probs = []
    with torch.no_grad():
        for _ in range(NUM_MC_PASSES):
            output = model(input_tensor)
            prob_fg = torch.softmax(output, dim=1)[0, 1]  # foreground probability
            probs.append(prob_fg)
    model.eval()  # back to fully deterministic afterward

    probs = torch.stack(probs)  # (T, D, H, W)
    mean_prob = probs.mean(dim=0)
    std_prob = probs.std(dim=0)
    mean_pred = (mean_prob > 0.5).float()

    mid = PATCH_SIZE[2] // 2
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    axes[0].imshow(img_patch[0, :, :, mid].numpy(), cmap="gray")
    axes[0].set_title("CT slice")
    axes[0].axis("off")

    axes[1].imshow(img_patch[0, :, :, mid].numpy(), cmap="gray")
    axes[1].imshow(lbl_patch[0, :, :, mid].numpy(), cmap="Reds", alpha=0.4)
    axes[1].contour(mean_pred[:, :, mid].numpy(), colors="cyan", linewidths=1)
    axes[1].set_title("Ground truth (red) vs. MC mean prediction (cyan)")
    axes[1].axis("off")

    axes[2].imshow(mean_prob[:, :, mid].numpy(), cmap="viridis", vmin=0, vmax=1)
    axes[2].set_title(f"Mean foreground probability ({NUM_MC_PASSES} passes)")
    axes[2].axis("off")

    im = axes[3].imshow(std_prob[:, :, mid].numpy(), cmap="magma")
    axes[3].set_title("Uncertainty (std across passes)")
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.046)

    plt.tight_layout()
    out_path = f"mc_dropout_case{case_idx + 1}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    boundary_std = std_prob[(mean_prob > 0.1) & (mean_prob < 0.9)].mean().item() if ((mean_prob > 0.1) & (mean_prob < 0.9)).any() else 0.0
    confident_std = std_prob[(mean_prob <= 0.1) | (mean_prob >= 0.9)].mean().item()
    print(f"case {case_idx + 1}: mean uncertainty at boundary={boundary_std:.4f}, "
          f"in confident regions={confident_std:.4f} -- saved {out_path}")
