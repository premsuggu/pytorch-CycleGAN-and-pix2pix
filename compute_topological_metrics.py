import os
import sys
import argparse
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import cv2
from skimage.morphology import skeletonize
from skimage.measure import euler_number
from scipy.ndimage import label
from rtnls_inference import SegmentationEnsemble

def parse_args():
    parser = argparse.ArgumentParser(description="Compute vascular and topological preservation metrics (clDice, Vessel Dice, Betti numbers)")
    parser.add_argument("--results_dir", type=str, required=True, help="Directory containing model test images (e.g. results/dual_encoder_cyclegan/test_latest/images)")
    parser.add_argument("--gt_vessel_dir", type=str, default="./datasets/dlri/aligned/testB_vessel", help="Directory containing ground truth confocal vessel maps")
    parser.add_argument("--model_path", type=str, default="./segmentation/models/vessels/vessels_july24.pt", help="Path to VascX torchscript ensemble")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--output_file", type=str, default=None, help="Optional text file to save results")
    return parser.parse_args()

def compute_cldice(v_p, v_l):
    """
    Computes clDice (connectivity-preserving Dice) between binary masks.
    v_p: predicted binary mask (boolean or {0, 1})
    v_l: ground truth binary mask (boolean or {0, 1})
    """
    v_p = (v_p > 0).astype(bool)
    v_l = (v_l > 0).astype(bool)
    
    if not v_p.any() and not v_l.any():
        return 1.0, 1.0, 1.0
    if not v_p.any() or not v_l.any():
        return 0.0, 0.0, 0.0

    s_p = skeletonize(v_p)
    s_l = skeletonize(v_l)

    tprec = (s_p & v_l).sum() / (s_p.sum() + 1e-7)
    tsens = (s_l & v_p).sum() / (s_l.sum() + 1e-7)

    if (tprec + tsens) == 0:
        cldice = 0.0
    else:
        cldice = 2.0 * tprec * tsens / (tprec + tsens)

    return float(cldice), float(tprec), float(tsens)

def compute_dice_and_iou(pred, gt):
    p = (pred > 0).astype(bool)
    g = (gt > 0).astype(bool)
    intersection = (p & g).sum()
    total = p.sum() + g.sum()
    union = (p | g).sum()
    
    dice = (2.0 * intersection) / (total + 1e-7) if total > 0 else 1.0
    iou = intersection / (union + 1e-7) if union > 0 else 1.0
    return float(dice), float(iou)

def compute_betti_numbers(mask):
    """
    Computes Betti 0 (connected components) and Betti 1 (loops/cycles).
    Euler char: chi = beta_0 - beta_1  =>  beta_1 = beta_0 - chi
    """
    b = (mask > 0).astype(np.uint8)
    if not b.any():
        return 0, 0
    _, n_components = label(b)
    chi = euler_number(b, connectivity=2)
    beta_0 = n_components
    beta_1 = max(0, beta_0 - chi)
    return beta_0, beta_1

def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    gt_vessel_dir = Path(args.gt_vessel_dir)
    model_path = Path(args.model_path)

    print("=" * 75)
    print("  VASCULAR & TOPOLOGICAL PRESERVATION BENCHMARK")
    print(f"  Target images dir: {results_dir.resolve()}")
    print(f"  Ground truth dir:  {gt_vessel_dir.resolve()}")
    print("=" * 75)

    fake_b_files = sorted(list(results_dir.glob("*_fake_B.png")))
    if not fake_b_files:
        raise FileNotFoundError(f"No *_fake_B.png images found in {results_dir}")

    print(f"Found {len(fake_b_files)} fake_B synthesis outputs.")

    # Check for paired ground truth vessel masks
    valid_samples = []
    for f in fake_b_files:
        # Base sample ID
        sample_id = f.name.replace("_fake_B.png", "")
        gt_mask_p = gt_vessel_dir / f"{sample_id}.png"
        if gt_mask_p.exists():
            valid_samples.append((sample_id, f, gt_mask_p))

    print(f"Matching ground-truth vessel masks found: {len(valid_samples)}/{len(fake_b_files)}")
    if not valid_samples:
        raise FileNotFoundError(f"No matching GT vessel masks in {gt_vessel_dir}")

    # Set up VascX segmentation model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading VascX ensemble on device: {device}...")
    vessel_ensemble = SegmentationEnsemble.from_torchscript(model_path).to(device)

    # Temporary directory for predicted fake_B vessel maps
    pred_vessel_dir = results_dir / "vascx_vessel_eval"
    pred_vessel_dir.mkdir(parents=True, exist_ok=True)

    # Check which fake_B already need prediction
    todo_list = []
    for sid, fake_p, _ in valid_samples:
        pred_p = pred_vessel_dir / f"{sid}.png"
        if not pred_p.exists() or pred_p.stat().st_size == 0:
            todo_list.append({"id": sid, "image": str(fake_p)})

    if todo_list:
        print(f"Running VascX segmentation on {len(todo_list)} fake_B images...")
        vessel_ensemble.predict(
            todo_list,
            dest_path=pred_vessel_dir,
            batch_size=args.batch_size,
            num_workers=2
        )
        # Ensure 0-255 scaling
        for item in todo_list:
            mp = pred_vessel_dir / f"{item['id']}.png"
            if mp.exists():
                with Image.open(mp) as im:
                    if im.getextrema()[1] <= 1:
                        im_scaled = Image.eval(im, lambda p: 255 if p > 0 else 0)
                        im_scaled.save(mp)
        print("VascX extraction on fake_B completed!")
    else:
        print("All fake_B vessel segmentations already extracted.")

    # Compute metrics across all samples
    dices = []
    ious = []
    cldices = []
    tprecs = []
    tsenses = []
    delta_beta0 = []
    delta_beta1 = []

    for sid, fake_p, gt_p in valid_samples:
        pred_mask_p = pred_vessel_dir / f"{sid}.png"
        pred_img = cv2.imread(str(pred_mask_p), cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)

        # Match dimensions if needed
        if pred_img.shape != gt_img.shape:
            pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        pred_bin = (pred_img > 127).astype(np.uint8)
        gt_bin = (gt_img > 127).astype(np.uint8)

        # Dice & IoU
        d, u = compute_dice_and_iou(pred_bin, gt_bin)
        dices.append(d)
        ious.append(u)

        # clDice
        cd, tp, ts = compute_cldice(pred_bin, gt_bin)
        cldices.append(cd)
        tprecs.append(tp)
        tsenses.append(ts)

        # Betti numbers
        p_b0, p_b1 = compute_betti_numbers(pred_bin)
        g_b0, g_b1 = compute_betti_numbers(gt_bin)
        delta_beta0.append(abs(p_b0 - g_b0))
        delta_beta1.append(abs(p_b1 - g_b1))

    # Summarize
    report = []
    report.append(f"Model Results: {results_dir.parent.parent.name} ({results_dir.parent.name})")
    report.append(f"Evaluated Samples: {len(valid_samples)}")
    report.append(f"  Vessel Dice        : {np.mean(dices):.4f} +/- {np.std(dices):.4f}")
    report.append(f"  Vessel IoU (Jaccard): {np.mean(ious):.4f} +/- {np.std(ious):.4f}")
    report.append(f"  clDice (Topology)  : {np.mean(cldices):.4f} +/- {np.std(cldices):.4f}")
    report.append(f"  Topo Precision     : {np.mean(tprecs):.4f} +/- {np.std(tprecs):.4f}")
    report.append(f"  Topo Sensitivity   : {np.mean(tsenses):.4f} +/- {np.std(tsenses):.4f}")
    report.append(f"  Delta Betti 0 (Conn): {np.mean(delta_beta0):.2f} +/- {np.std(delta_beta0):.2f}")
    report.append(f"  Delta Betti 1 (Loop): {np.mean(delta_beta1):.2f} +/- {np.std(delta_beta1):.2f}")

    summary_text = "\n".join(report)
    print("\n" + "=" * 75)
    print(summary_text)
    print("=" * 75)

    if args.output_file:
        Path(args.output_file).write_text(summary_text)
        print(f"Summary written to {args.output_file}")

if __name__ == "__main__":
    main()
