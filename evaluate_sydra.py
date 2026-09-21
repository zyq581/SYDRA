import argparse
import json
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from skimage import measure
from torch.utils.data import DataLoader

from data_loader_sydra import TestDataset
from sydra_model import SyDRADiscriminativeStream, SyDRAReconstructiveStream


def checkpoint_state(path, device):
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


def f1_max(labels, scores):
    precision, recall, _ = precision_recall_curve(labels, scores)
    values = 2.0 * precision * recall / (precision + recall + 1e-12)
    return float(np.max(values))


def pro_auc(masks, anomaly_maps, num_thresholds=100, max_fpr=0.3):
    masks = np.asarray(masks, dtype=bool)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
    background_count = int(np.sum(~masks))
    if not masks.any() or background_count == 0:
        return float("nan")

    regions_by_image = []
    for mask in masks:
        labeled = measure.label(mask)
        regions_by_image.append(
            [(labeled == region.label, region.area) for region in measure.regionprops(labeled)]
        )

    thresholds = np.linspace(float(anomaly_maps.min()), float(anomaly_maps.max()), num_thresholds)
    fprs = []
    pros = []
    for threshold in sorted(thresholds, reverse=True):
        predictions = anomaly_maps > threshold
        overlaps = []
        for prediction, regions in zip(predictions, regions_by_image):
            for region_mask, area in regions:
                overlaps.append(np.sum(prediction & region_mask) / area)
        fprs.append(np.sum(predictions & ~masks) / background_count)
        pros.append(np.mean(overlaps) if overlaps else 0.0)

    fprs = np.asarray(fprs)
    pros = np.asarray(pros)
    valid = fprs < max_fpr
    if np.sum(valid) < 2:
        return 0.0
    x = fprs[valid]
    y = pros[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(auc(x, y) / max_fpr)


def safe_ranking_metric(metric, labels, scores):
    if np.unique(labels).size < 2:
        return float("nan")
    return float(metric(labels, scores))


def compute_boundary_f1(masks, anomaly_maps, ring_width=5, num_thresholds=100):
    """Maximum pixel F1 within pooled ground-truth contour rings."""
    masks = np.asarray(masks).astype(bool)
    anomaly_maps = np.asarray(anomaly_maps)
    boundary_gt, boundary_pred = [], []
    structure = np.ones((3, 3), dtype=bool)
    for mask, score in zip(masks, anomaly_maps):
        if mask.sum() == 0:
            continue
        dilated = mask.copy()
        eroded = mask.copy()
        for _ in range(ring_width):
            dilated = binary_dilation(dilated, structure=structure)
            eroded = binary_erosion(eroded, structure=structure)
        ring = np.logical_xor(dilated, eroded)
        if ring.sum() == 0:
            continue
        boundary_gt.append(mask[ring].astype(np.uint8))
        boundary_pred.append(score[ring])
    if not boundary_gt:
        return 0.0
    gt = np.concatenate(boundary_gt)
    pred = np.concatenate(boundary_pred)
    if gt.max() == gt.min():
        return 0.0
    thresholds = np.linspace(float(pred.min()), float(pred.max()), num_thresholds)
    best = 0.0
    for th in thresholds:
        binary = pred > th
        tp = np.logical_and(binary, gt == 1).sum()
        fp = np.logical_and(binary, gt == 0).sum()
        fn = np.logical_and(~binary, gt == 1).sum()
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        best = max(best, 2 * precision * recall / (precision + recall + 1e-12))
    return float(best)


def resolve_test_path(data_path):
    candidate = os.path.join(data_path, "test")
    return candidate if os.path.isdir(candidate) else data_path


def evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    reconstructor = SyDRAReconstructiveStream(in_channels=3, out_channels=3, base_width=128).to(device)
    discriminator = SyDRADiscriminativeStream(
        in_channels=6, out_channels=2, base_channels=64
    ).to(device)
    reconstructor.load_state_dict(checkpoint_state(args.recon_checkpoint, device), strict=True)
    discriminator.load_state_dict(checkpoint_state(args.seg_checkpoint, device), strict=True)
    reconstructor.eval()
    discriminator.eval()

    test_path = resolve_test_path(args.data_path)
    dataset = TestDataset(test_path, resize_shape=[args.height, args.width], split_file=args.test_split)
    if not dataset:
        raise RuntimeError(f"No test images were found under {test_path}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    image_labels = []
    image_scores = []
    masks = []
    anomaly_maps = []
    boundary_maps = []
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            reconstruction = reconstructor(image)
            logits = discriminator(torch.cat((reconstruction, image), dim=1))
            raw_map = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
            anomaly_map = raw_map
            if args.sigma > 0:
                anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma)
            if args.boundary_f1:
                boundary_maps.append(gaussian_filter(raw_map, sigma=args.boundary_sigma)
                                     if args.boundary_sigma > 0 else raw_map)

            mask = (batch["mask"][0, 0].numpy() > 0).astype(np.uint8)
            masks.append(mask)
            anomaly_maps.append(anomaly_map)
            image_labels.append(int(batch["has_anomaly"].item()))
            image_scores.append(float(anomaly_map.max()))

    image_labels = np.asarray(image_labels, dtype=np.uint8)
    image_scores = np.asarray(image_scores, dtype=np.float32)
    masks = np.asarray(masks, dtype=np.uint8)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
    pixel_labels = masks.reshape(-1)
    pixel_scores = anomaly_maps.reshape(-1)

    metrics = {
        "num_images": int(len(dataset)),
        "sigma": float(args.sigma),
        "image_auroc": safe_ranking_metric(roc_auc_score, image_labels, image_scores),
        "image_ap": safe_ranking_metric(average_precision_score, image_labels, image_scores),
        "image_f1_max": f1_max(image_labels, image_scores),
        "pixel_auroc": safe_ranking_metric(roc_auc_score, pixel_labels, pixel_scores),
        "pixel_ap": safe_ranking_metric(average_precision_score, pixel_labels, pixel_scores),
        "pixel_f1_max": f1_max(pixel_labels, pixel_scores),
        "pro_0.3": pro_auc(masks, anomaly_maps, num_thresholds=args.pro_thresholds),
    }
    if args.boundary_f1:
        metrics["boundary_f1"] = compute_boundary_f1(masks, boundary_maps)
    metrics["protocol"] = {
        "height": args.height, "width": args.width,
        "boundary_sigma": args.boundary_sigma if args.boundary_f1 else None,
        "boundary_ring_width": 5 if args.boundary_f1 else None,
        "boundary_thresholds": 100 if args.boundary_f1 else None,
        "pro_thresholds": args.pro_thresholds, "pro_max_fpr": 0.3,
        "gt_resize": "bilinear_then_greater_than_zero", "image_score": "max_smoothed_map",
        "num_images": int(len(dataset)),
    }
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one SyDRA checkpoint pair.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--test_split", help="Image paths relative to test/.")
    parser.add_argument("--recon_checkpoint", required=True)
    parser.add_argument("--seg_checkpoint", required=True)
    parser.add_argument("--output")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--boundary_f1", action="store_true", help="Report the boundary-ring F1 metric.")
    parser.add_argument("--boundary_sigma", type=float, default=1.0)
    parser.add_argument("--pro_thresholds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    if args.sigma < 0 or args.boundary_sigma < 0 or args.pro_thresholds < 2:
        parser.error("Smoothing sigma must be nonnegative and PRO thresholds must be >= 2")
    if args.height < 32 or args.width < 32 or args.height % 32 or args.width % 32:
        parser.error("Height and width must be positive multiples of 32")
    return args


def main():
    args = parse_args()
    metrics = evaluate(args)
    text = json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True)
    print(text)
    if args.output:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"[INFO] Results saved to {args.output}")


if __name__ == "__main__":
    main()
