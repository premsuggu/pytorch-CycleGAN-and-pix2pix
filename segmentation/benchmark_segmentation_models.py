import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from skimage.morphology import skeletonize

# Setup imports
sys.path.append(str(Path(__file__).parent / 'lwnet_src'))
sys.path.append(str(Path(__file__).parent / 'sgl_src'))

from rtnls_inference import SegmentationEnsemble

# Metrics helpers
def compute_cldice(v_p, v_l):
    v_p = (v_p > 0).astype(bool)
    v_l = (v_l > 0).astype(bool)
    if not v_p.any() and not v_l.any(): return 1.0, 1.0, 1.0
    if not v_p.any() or not v_l.any(): return 0.0, 0.0, 0.0
    s_p = skeletonize(v_p)
    s_l = skeletonize(v_l)
    tprec = (s_p & v_l).sum() / (s_p.sum() + 1e-7)
    tsens = (s_l & v_p).sum() / (s_l.sum() + 1e-7)
    cldice = 2.0 * tprec * tsens / (tprec + tsens + 1e-7)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, default="./datasets/dlri/aligned/testB")
    parser.add_argument("--gt_vessel_dir", type=str, default="./datasets/dlri/aligned/testB_vessel")
    parser.add_argument("--num_samples", type=int, default=15)
    parser.add_argument("--output_file", type=str, default="./results/segmentation_benchmark.txt")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 70)
    print("  RETINAL VESSEL SEGMENTATION MODEL BENCHMARK")
    print(f"  Device: {device}")
    print("=" * 70)

    test_dir = Path(args.test_dir)
    gt_dir = Path(args.gt_vessel_dir)
    images = sorted(list(test_dir.glob("*.png")))[:args.num_samples]
    print(f"Benchmarking on {len(images)} test samples...")

    # 1. Load VascX
    print("\n[1/3] Loading VascX Ensemble...")
    vascx_weights = Path("./segmentation/models/vessels/vessels_july24.pt")
    vascx_model = SegmentationEnsemble.from_torchscript(vascx_weights).to(device)

    # 2. Load LWNet
    print("[2/3] Loading LWNet...")
    from models.get_model import get_arch
    from utils.model_saving_loading import load_model
    from predict_one_image import get_fov, crop_to_fov, create_pred
    from utils import paired_transforms_tv04 as p_tr

    lwnet_model = get_arch('wnet').to(device)
    lwnet_model.mode = 'eval'
    lwnet_model, _ = load_model(lwnet_model, 'segmentation/models/lwnet', device)
    lwnet_model.eval()

    # 3. Load SGL
    print("[3/3] Loading SGL...")
    from option import args as sgl_args
    from model import Model as SGLModel
    class DummyCheckpoint:
        def __init__(self): self.log_file = sys.stdout
        def get_path(self, *a, **kw): return '.'
    sgl_args.model = 'CON'
    sgl_args.pre_train = 'segmentation/models/sgl/drive_k8.pth'
    sgl_args.test_only = True
    sgl_args.cpu = not torch.cuda.is_available()
    sgl_model = SGLModel(sgl_args, DummyCheckpoint())
    sgl_model.eval()

    results = {"VascX": {"dice": [], "iou": [], "cldice": [], "latency": []},
               "LWNet": {"dice": [], "iou": [], "cldice": [], "latency": []},
               "SGL":   {"dice": [], "iou": [], "cldice": [], "latency": []}}

    for img_p in images:
        gt_p = gt_dir / img_p.name
        if not gt_p.exists(): continue
        gt_img = cv2.imread(str(gt_p), cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_img > 0).astype(np.uint8)

        # Eval VascX
        t0 = time.time()
        chunk = [{"id": img_p.stem, "image": str(img_p)}]
        out_vascx = vascx_model.predict_tensors(chunk)[0] if hasattr(vascx_model, 'predict_tensors') else None
        if out_vascx is None:
            # fallback predict
            tmp_dir = Path("/tmp/vascx_bench")
            tmp_dir.mkdir(exist_ok=True)
            vascx_model.predict(chunk, dest_path=tmp_dir, batch_size=1, num_workers=0)
            vascx_pred = cv2.imread(str(tmp_dir / f"{img_p.stem}.png"), cv2.IMREAD_GRAYSCALE)
        else:
            vascx_pred = (out_vascx.cpu().numpy() * 255).astype(np.uint8)
        lat_vascx = (time.time() - t0) * 1000
        if vascx_pred.shape != gt_bin.shape:
            vascx_pred = cv2.resize(vascx_pred, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
        v_bin = (vascx_pred > 0).astype(np.uint8)
        d_v, u_v = compute_dice_and_iou(v_bin, gt_bin)
        cd_v, _, _ = compute_cldice(v_bin, gt_bin)
        results["VascX"]["dice"].append(d_v)
        results["VascX"]["iou"].append(u_v)
        results["VascX"]["cldice"].append(cd_v)
        results["VascX"]["latency"].append(lat_vascx)

        # Eval LWNet
        t0 = time.time()
        pil_im = Image.open(img_p).convert('RGB')
        mask = get_fov(pil_im)
        mask_arr = np.array(mask).astype(bool)
        img_cr, coords = crop_to_fov(pil_im, mask_arr)
        orig_sz = img_cr.size[1], img_cr.size[0]
        tr = p_tr.Compose([p_tr.Resize((512, 512)), p_tr.ToTensor()])
        im_t = tr(img_cr).to(device)
        with torch.no_grad():
            _, lwnet_bin = create_pred(lwnet_model, im_t, mask_arr, coords, orig_sz, 0.5)
        lat_lwnet = (time.time() - t0) * 1000
        if lwnet_bin.shape != gt_bin.shape:
            lwnet_bin = cv2.resize(lwnet_bin.astype(np.uint8), (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
        d_l, u_l = compute_dice_and_iou(lwnet_bin, gt_bin)
        cd_l, _, _ = compute_cldice(lwnet_bin, gt_bin)
        results["LWNet"]["dice"].append(d_l)
        results["LWNet"]["iou"].append(u_l)
        results["LWNet"]["cldice"].append(cd_l)
        results["LWNet"]["latency"].append(lat_lwnet)

        # Eval SGL
        t0 = time.time()
        im_sgl = Image.open(img_p).convert('RGB').resize((512, 512))
        sgl_np = np.array(im_sgl).transpose(2,0,1)[np.newaxis, ...]
        sgl_t = torch.tensor(sgl_np).float().to(device)
        with torch.no_grad():
            _, y = sgl_model(sgl_t, 0)
            pred_sgl = torch.sigmoid(y).squeeze().cpu().numpy()
        lat_sgl = (time.time() - t0) * 1000
        sgl_bin = (pred_sgl > 0.5).astype(np.uint8)
        if sgl_bin.shape != gt_bin.shape:
            sgl_bin = cv2.resize(sgl_bin, (gt_bin.shape[1], gt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
        d_s, u_s = compute_dice_and_iou(sgl_bin, gt_bin)
        cd_s, _, _ = compute_cldice(sgl_bin, gt_bin)
        results["SGL"]["dice"].append(d_s)
        results["SGL"]["iou"].append(u_s)
        results["SGL"]["cldice"].append(cd_s)
        results["SGL"]["latency"].append(lat_sgl)

    out_lines = []
    out_lines.append("| Model Architecture | Dice Score ↑ | IoU (Jaccard) ↑ | clDice (Topology) ↑ | Latency (ms) ↓ |")
    out_lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for m in ["LWNet", "SGL", "VascX"]:
        d_m = np.mean(results[m]["dice"])
        u_m = np.mean(results[m]["iou"])
        cd_m = np.mean(results[m]["cldice"])
        lat_m = np.mean(results[m]["latency"])
        out_lines.append(f"| **{m}** | {d_m:.4f} | {u_m:.4f} | {cd_m:.4f} | {lat_m:.1f} ms |")

    table = "\n".join(out_lines)
    print("\n" + table)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_file).write_text(table)
    print(f"\nSaved benchmark table to {args.output_file}")

if __name__ == "__main__":
    main()
