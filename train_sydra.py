import argparse
import json
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader

from data_loader_sydra import TrainDataset
from loss import FocalLoss, SSIM
from sydra_model import SyDRADiscriminativeStream, SyDRAReconstructiveStream


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def initialize_weights(module):
    name = module.__class__.__name__
    if "Conv" in name and getattr(module, "weight", None) is not None:
        module.weight.data.normal_(0.0, 0.02)
    elif "BatchNorm" in name:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.zero_()


def training_directory(args, class_name):
    if args.dataset_layout == "category":
        return os.path.join(args.data_path, class_name, "train")
    return os.path.join(args.data_path, "train")


def scheduler_factor(epoch, total_epochs):
    if epoch < 15:
        alpha = (epoch + 1) / 15.0
        return 0.1 + 0.9 * alpha
    if epoch >= total_epochs * 0.9:
        return 0.01
    if epoch >= total_epochs * 0.8:
        return 0.1
    return 1.0


def train_one_run(class_name, seed, args, device):
    setup_seed(seed)
    run_name = f"SyDRA_{class_name}_effbs{args.effective_bs}_lr{args.lr}_seed{seed}"
    print(f"[INFO] Starting {run_name}")

    reconstructor = SyDRAReconstructiveStream(in_channels=3, out_channels=3).to(device)
    reconstructor.apply(initialize_weights)
    discriminator = SyDRADiscriminativeStream(in_channels=6, out_channels=2).to(device)
    discriminator.apply(initialize_weights)

    optimizer = torch.optim.Adam(
        [
            {"params": reconstructor.parameters(), "lr": args.lr},
            {"params": discriminator.parameters(), "lr": args.lr},
        ]
    )
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: scheduler_factor(epoch, args.epochs)
    )

    recon_last = os.path.join(args.checkpoint_path, run_name + "_recon_last.pckl")
    seg_last = os.path.join(args.checkpoint_path, run_name + "_seg_last.pckl")
    dataset = TrainDataset(
        root_dir=training_directory(args, class_name),
        anomaly_source_path=args.anomaly_source_path,
        resize_shape=(args.height, args.width),
        split_file=args.train_split.format(class_name=class_name) if args.train_split else None,
        normal_only=args.normal_only,
    )
    if not dataset:
        raise RuntimeError("The training dataset is empty.")
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.workers,
        drop_last=args.drop_last == "true",
        pin_memory=True,
    )

    if len(loader) == 0:
        raise RuntimeError("No batches remain; check training size, --bs and --drop_last.")
    run_config = dict(vars(args), class_name=class_name, seed=seed,
                      actual_drop_last=loader.drop_last, num_train_images=len(dataset),
                      torch_version=torch.__version__, numpy_version=np.__version__)
    with open(os.path.join(args.checkpoint_path, run_name + "_config.json"), "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)
    with open(os.path.join(args.checkpoint_path, run_name + "_train_split.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(os.path.relpath(p, dataset.root_dir).replace("\\", "/")
                               for p in dataset.image_paths) + "\n")

    accumulation_steps = max(1, args.effective_bs // args.bs)
    l2_loss = torch.nn.MSELoss(reduction="none")
    ssim_loss = SSIM()
    focal_loss = FocalLoss()
    for epoch in range(args.epochs):
        reconstructor.train()
        discriminator.train()
        optimizer.zero_grad()
        reconstruction_total = 0.0
        segmentation_total = 0.0

        for batch_index, batch in enumerate(loader):
            clean = batch["image"].to(device, non_blocking=True)
            augmented = batch["augmented_image"].to(device, non_blocking=True)
            mask = batch["anomaly_mask"].to(device, non_blocking=True)
            is_labeled = batch["is_labeled"].to(device, non_blocking=True).view(-1, 1, 1, 1)

            reconstruction = reconstructor(augmented)
            logits = discriminator(torch.cat((reconstruction, augmented), dim=1))
            probabilities = torch.softmax(logits, dim=1)

            synthetic_mask = 1.0 - is_labeled
            pixel_loss = l2_loss(reconstruction, clean)
            normalizer = synthetic_mask.sum() * clean.shape[1] * clean.shape[2] * clean.shape[3]
            reconstruction_l2 = (pixel_loss * synthetic_mask).sum() / (normalizer + 1e-6)
            reconstruction_ssim = ssim_loss(reconstruction, clean) * synthetic_mask.mean()
            reconstruction_loss = reconstruction_l2 + args.ssim_weight * reconstruction_ssim
            segmentation_loss = focal_loss(probabilities, mask)
            ((reconstruction_loss + args.seg_weight * segmentation_loss) / accumulation_steps).backward()

            if (batch_index + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            reconstruction_total += reconstruction_loss.item()
            segmentation_total += segmentation_loss.item()
        if len(loader) % accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        torch.save(reconstructor.state_dict(), recon_last)
        torch.save(discriminator.state_dict(), seg_last)
        completed_epoch = epoch + 1
        if completed_epoch % args.save_every == 0:
            torch.save(
                reconstructor.state_dict(),
                os.path.join(args.checkpoint_path, f"{run_name}_recon_ep{completed_epoch}.pckl"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(args.checkpoint_path, f"{run_name}_seg_ep{completed_epoch}.pckl"),
            )

        print(
            f"[EPOCH {completed_epoch}/{args.epochs}] lr={optimizer.param_groups[0]['lr']:.6g} "
            f"recon={reconstruction_total / len(loader):.4f} "
            f"seg={segmentation_total / len(loader):.4f}"
        )
def parse_args():
    parser = argparse.ArgumentParser(description="Train the SyDRA model.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--dataset_layout", choices=("category", "direct"), default="category")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--train_split", help="Paths relative to train/. Use {class_name} for multi-class runs.")
    parser.add_argument("--normal_only", action="store_true", help="Use only defect-free real images (0% protocol).")
    parser.add_argument("--drop_last", choices=("true", "false"), default="false")
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--anomaly_source_path", required=True)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--effective_bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ssim_weight", type=float, default=2.0)
    parser.add_argument("--seg_weight", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5)
    args = parser.parse_args()
    if args.effective_bs < args.bs or args.effective_bs % args.bs:
        parser.error("--effective_bs must be an integer multiple of --bs")
    if args.dataset_layout == "direct" and len(args.classes) != 1:
        parser.error("The direct layout accepts exactly one class label per invocation.")
    if min(args.bs, args.effective_bs, args.epochs, args.save_every) <= 0:
        parser.error("Batch sizes, epochs and save_every must be positive")
    if args.height < 32 or args.width < 32 or args.height % 32 or args.width % 32:
        parser.error("Height and width must be positive multiples of 32")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires a CUDA-enabled PyTorch environment.")
    os.makedirs(args.checkpoint_path, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    for seed in args.seeds:
        for class_name in args.classes:
            train_one_run(class_name, seed, args, device)


if __name__ == "__main__":
    main()
