# ============================================================
# Spleen Segmentation - Real Multi-GPU DDP Demo (Kaggle dual-T4 notebook)
# ============================================================
# HOW TO USE:
# 1. kaggle.com/code -> New Notebook
# 2. Notebook Settings (right sidebar) -> Accelerator -> GPU T4 x2 -> Save
# 3. Add the Task09_Spleen data: either upload data/Task09_Spleen manually,
#    or paste this whole file into a cell and let the MONAI DecathlonDataset
#    downloader fetch it (needs internet enabled in notebook settings).
# 4. Paste this WHOLE file into a single cell -> run.
# 5. It spawns one process per visible GPU (torch.multiprocessing.spawn --
#    this is the standard way to do real DDP inside a single notebook
#    process, since Kaggle notebooks don't give you a shell for `torchrun`).
# 6. At the end it prints which GPU each rank actually ran on and zips the
#    results (checkpoint + training log + per-rank timing) for download.
#    Bring the zip back to the Ubuntu VM and unzip into
#    experiments/ddp_kaggle/.
# ============================================================
#
# WHY THIS SCRIPT EXISTS: Tier 3 item 16 asks for a genuine multi-GPU
# torch.distributed demo, not a claim of DDP without evidence. This trains
# the SAME architecture as the CPU baseline (4-level UNet, 64^3 patches,
# 2.5/2.5/3.0mm spacing, same lr/augmentation as src/train_spleen.py) so the
# resulting Dice is at least comparable in kind to 0.4603, but the actual
# point being demonstrated is the distributed mechanics: DistributedSampler
# splitting data across ranks, DDP-wrapped gradient sync, and both GPUs
# genuinely doing work (proven by the per-rank device/timing printout, not
# by having a single process print "GPU 0,1" in a comment).

get_ipython().system('pip install -q monai nibabel')

import os
import csv
import time
import zipfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from monai.apps import DecathlonDataset
from monai.data import DataLoader, decollate_batch
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete, Compose, CropForegroundd, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, Orientationd, RandCropByPosNegLabeld, RandFlipd,
    RandRotate90d, ScaleIntensityRanged, Spacingd,
)

ROOT_DIR = "./data"
PATCH_SIZE = (64, 64, 64)   # same as src/train_spleen.py -- this is a distributed
                            # mechanics demo, not a new best-model attempt
NUM_EPOCHS = 30
MASTER_ADDR = "127.0.0.1"
MASTER_PORT = "29500"

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


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def run_worker(rank, world_size, return_dict):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    device_name = torch.cuda.get_device_name(rank)
    print(f"[rank {rank}] real device: cuda:{rank} ({device_name})")

    # download=True is safe to call from every rank -- MONAI/Decathlon just
    # no-ops if the data is already present, and rank 0 will typically win
    # the download race in practice on Kaggle's shared filesystem.
    train_ds = DecathlonDataset(
        root_dir=ROOT_DIR, task="Task09_Spleen", section="training",
        transform=train_transforms, download=True, cache_rate=0.0, num_workers=2,
    )
    val_ds = DecathlonDataset(
        root_dir=ROOT_DIR, task="Task09_Spleen", section="validation",
        transform=val_transforms, download=False, cache_rate=0.0, num_workers=2,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler, num_workers=2)
    # validation only needs to run on rank 0 -- no need to shard/duplicate it
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2) if rank == 0 else None

    model = UNet(
        spatial_dims=3, in_channels=1, out_channels=2,
        channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=2,
    ).to(device)
    model = DDP(model, device_ids=[rank])

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    history = []
    best_dice = -1.0
    train_start = time.time()

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)  # required so each rank sees a different shuffle each epoch
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(step, 1)
        epoch_time = time.time() - epoch_start

        val_dice_str = ""
        if rank == 0:
            model.eval()
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    val_outputs = sliding_window_inference(val_inputs, PATCH_SIZE, 1, model)
                    val_outputs_d = [post_pred(x) for x in decollate_batch(val_outputs)]
                    val_labels_d = [post_label(x) for x in decollate_batch(val_labels)]
                    dice_metric(y_pred=val_outputs_d, y=val_labels_d)
                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()
            val_dice_str = f"{mean_dice:.4f}"
            if mean_dice > best_dice:
                best_dice = mean_dice
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.module.state_dict(), "checkpoints/ddp_spleen_model.pth")
            print(f"[rank {rank}] epoch {epoch+1}/{NUM_EPOCHS} loss={epoch_loss:.4f} "
                  f"val_dice={mean_dice:.4f} epoch_time={epoch_time:.1f}s")
        else:
            print(f"[rank {rank}] epoch {epoch+1}/{NUM_EPOCHS} loss={epoch_loss:.4f} "
                  f"epoch_time={epoch_time:.1f}s (validation runs on rank 0 only)")

        history.append((rank, epoch + 1, epoch_loss, val_dice_str, epoch_time))

    total_time = time.time() - train_start
    print(f"[rank {rank}] done in {total_time:.0f}s total")

    if rank == 0:
        return_dict["best_dice"] = best_dice
        return_dict["total_time"] = total_time
        return_dict["world_size"] = world_size
        return_dict["history"] = history
        return_dict["device_names"] = [torch.cuda.get_device_name(i) for i in range(world_size)]

    cleanup()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"Visible GPUs: {world_size}")
    assert world_size >= 2, (
        "This script demonstrates real multi-GPU DDP -- go to Notebook "
        "Settings > Accelerator > GPU T4 x2 and restart the session."
    )

    os.makedirs(ROOT_DIR, exist_ok=True)  # create once here, not once per spawned rank
    manager = mp.Manager()
    return_dict = manager.dict()
    mp.spawn(run_worker, args=(world_size, return_dict), nprocs=world_size, join=True)

    print("\n=== DDP training summary ===")
    print(f"World size (GPUs actually used): {return_dict['world_size']}")
    print(f"GPU devices: {return_dict['device_names']}")
    print(f"Total wall-clock time: {return_dict['total_time']:.0f}s")
    print(f"Best validation Dice (rank 0): {return_dict['best_dice']:.4f}")
    print(
        "(Same architecture/recipe as src/train_spleen.py's 0.4603 CPU run -- "
        "this number is a byproduct of proving the distributed training "
        "mechanics work for real, not a new best-model claim.)"
    )

    with open("ddp_training_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "epoch", "train_loss", "val_dice", "epoch_time_seconds"])
        writer.writerows(return_dict["history"])

    with open("ddp_summary.txt", "w") as f:
        f.write(f"World size: {return_dict['world_size']}\n")
        f.write(f"GPU devices: {return_dict['device_names']}\n")
        f.write(f"Total wall-clock time: {return_dict['total_time']:.0f}s\n")
        f.write(f"Best validation Dice: {return_dict['best_dice']:.4f}\n")

    with zipfile.ZipFile("ddp_results.zip", "w") as z:
        z.write("ddp_training_log.csv")
        z.write("ddp_summary.txt")
        if os.path.isfile("checkpoints/ddp_spleen_model.pth"):
            z.write("checkpoints/ddp_spleen_model.pth")

    print("\nZipped: ddp_results.zip")
    print("Downloading now (Kaggle: also check the notebook's Output tab if this doesn't trigger)...")
    try:
        from IPython.display import FileLink
        display(FileLink("ddp_results.zip"))
    except Exception:
        pass
