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
CASE_INDICES = [0, 1, 2]  # first few validation cases

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

# Bottleneck ResidualUnit -- deepest, most downsampled feature map (128ch).
target_layer = model.model[1].submodule[1].submodule[1].submodule.conv.unit1

_activations = {}
_gradients = {}


def _forward_hook(module, inp, out):
    _activations["value"] = out.detach()


def _backward_hook(module, grad_input, grad_output):
    _gradients["value"] = grad_output[0].detach()


target_layer.register_forward_hook(_forward_hook)
target_layer.register_full_backward_hook(_backward_hook)


def extract_patch_around_label(image, label, patch_size):
    """Crop a fixed-size patch centered on the labeled spleen voxels (falls
    back to the volume center if the label is empty in this crop)."""
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
        pad_spec = (0, pads[2], 0, pads[1], 0, pads[0])  # F.pad is reverse-dim order
        img_patch = F.pad(img_patch, pad_spec)
        lbl_patch = F.pad(lbl_patch, pad_spec)

    return img_patch, lbl_patch


def compute_gradcam(img_patch):
    input_tensor = img_patch.unsqueeze(0).clone().to(DEVICE)
    input_tensor.requires_grad_(True)

    output = model(input_tensor)  # (1, 2, D, H, W)
    foreground_score = output[:, 1].sum()  # target: total foreground evidence
    model.zero_grad()
    foreground_score.backward()

    acts = _activations["value"][0]   # (C, d, h, w)
    grads = _gradients["value"][0]    # (C, d, h, w)
    weights = grads.mean(dim=(1, 2, 3))  # global-average-pooled gradients, per MONAI Seg-Grad-CAM convention

    cam = torch.einsum("c,cdhw->dhw", weights, acts)
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-8)

    cam_up = F.interpolate(cam[None, None], size=PATCH_SIZE, mode="trilinear", align_corners=False)[0, 0]
    return cam_up.detach(), output.detach()


for case_idx in CASE_INDICES:
    data = val_ds[case_idx]
    image, label = data["image"], data["label"]

    img_patch, lbl_patch = extract_patch_around_label(image, label, PATCH_SIZE)
    cam, output = compute_gradcam(img_patch)
    pred = torch.argmax(output, dim=1)[0]

    mid = PATCH_SIZE[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    axes[0].imshow(img_patch[0, :, :, mid].numpy(), cmap="gray")
    axes[0].set_title("CT slice")
    axes[0].axis("off")

    axes[1].imshow(img_patch[0, :, :, mid].numpy(), cmap="gray")
    axes[1].imshow(lbl_patch[0, :, :, mid].numpy(), cmap="Reds", alpha=0.4)
    pred_slice = pred[:, :, mid].numpy()
    axes[1].contour(pred_slice, colors="cyan", linewidths=1)
    axes[1].set_title("Ground truth (red) vs. prediction (cyan outline)")
    axes[1].axis("off")

    axes[2].imshow(img_patch[0, :, :, mid].numpy(), cmap="gray")
    axes[2].imshow(cam[:, :, mid].numpy(), cmap="jet", alpha=0.5)
    axes[2].set_title("Grad-CAM (bottleneck layer)")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = f"results/gradcam_case{case_idx + 1}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
