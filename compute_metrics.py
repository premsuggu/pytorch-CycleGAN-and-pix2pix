import os
import glob
import torch
import lpips
import argparse
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms
from collections import defaultdict
from tqdm import tqdm

def load_image_np(path, target_shape=None):
    img = Image.open(path).convert("RGB")
    if target_shape is not None:
        # target_shape is (H, W, C), PIL needs (W, H)
        img = img.resize((target_shape[1], target_shape[0]), Image.BICUBIC)
    return np.array(img)

def load_image_tensor(path, target_shape=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    img = Image.open(path).convert("RGB")
    if target_shape is not None:
        img = img.resize((target_shape[1], target_shape[0]), Image.BICUBIC)
    return transform(img).unsqueeze(0)

def main():
    parser = argparse.ArgumentParser(description="Compute evaluation metrics for CycleGAN outputs.")
    parser.add_argument("--dir", type=str, default="results/cyclegan_paired_ssim/test_200/images",
                        help="Directory containing the test images.")
    args = parser.parse_args()

    BASE = args.dir
    if not os.path.exists(BASE):
        print(f"Error: Directory {BASE} does not exist.")
        return

    # Find all unique prefixes
    all_files = glob.glob(os.path.join(BASE, "*_real_A.png"))
    prefixes = [os.path.basename(f).replace("_real_A.png", "") for f in all_files]

    if not prefixes:
        print(f"No valid image sets found in {BASE}. Make sure files end with '_real_A.png' etc.")
        return

    print(f"Found {len(prefixes)} image sets in '{BASE}'. Computing metrics...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    def compute_lpips(t1, t2):
        return loss_fn(t1, t2).item()

    metrics = defaultdict(list)

    for prefix in tqdm(prefixes):
        paths = {
            "real_A": os.path.join(BASE, f"{prefix}_real_A.png"),
            "real_B": os.path.join(BASE, f"{prefix}_real_B.png"),
            "fake_A": os.path.join(BASE, f"{prefix}_fake_A.png"),
            "fake_B": os.path.join(BASE, f"{prefix}_fake_B.png"),
            "rec_A": os.path.join(BASE, f"{prefix}_rec_A.png"),
            "rec_B": os.path.join(BASE, f"{prefix}_rec_B.png"),
        }

        # Check if all needed files exist
        if not all(os.path.exists(p) for p in paths.values()):
            print(f"Warning: Missing files for prefix {prefix}. Skipping.")
            continue

        real_A = load_image_np(paths["real_A"])
        real_B = load_image_np(paths["real_B"])
        
        target_shape_A = real_A.shape
        target_shape_B = real_B.shape

        fake_A = load_image_np(paths["fake_A"], target_shape_A)
        fake_B = load_image_np(paths["fake_B"], target_shape_B)
        rec_A = load_image_np(paths["rec_A"], target_shape_A)
        rec_B = load_image_np(paths["rec_B"], target_shape_B)

        # SSIM
        metrics["ssim_A"].append(ssim(real_A, rec_A, channel_axis=2))
        metrics["ssim_B"].append(ssim(real_B, rec_B, channel_axis=2))
        metrics["ssim_AB"].append(ssim(real_B, fake_B, channel_axis=2))
        metrics["ssim_BA"].append(ssim(real_A, fake_A, channel_axis=2))

        # PSNR
        metrics["psnr_A"].append(psnr(real_A, rec_A))
        metrics["psnr_B"].append(psnr(real_B, rec_B))
        metrics["psnr_AB"].append(psnr(real_B, fake_B))
        metrics["psnr_BA"].append(psnr(real_A, fake_A))

        # LPIPS
        t_real_A = load_image_tensor(paths["real_A"]).to(device)
        t_real_B = load_image_tensor(paths["real_B"]).to(device)
        t_fake_A = load_image_tensor(paths["fake_A"], target_shape_A).to(device)
        t_fake_B = load_image_tensor(paths["fake_B"], target_shape_B).to(device)
        t_rec_A = load_image_tensor(paths["rec_A"], target_shape_A).to(device)
        t_rec_B = load_image_tensor(paths["rec_B"], target_shape_B).to(device)

        metrics["lpips_A"].append(compute_lpips(t_real_A, t_rec_A))
        metrics["lpips_B"].append(compute_lpips(t_real_B, t_rec_B))
        metrics["lpips_AB"].append(compute_lpips(t_real_B, t_fake_B))
        metrics["lpips_BA"].append(compute_lpips(t_real_A, t_fake_A))

    # Average metrics
    avg = {k: np.mean(v) for k, v in metrics.items()}

    print("\n===== Cycle Reconstruction Metrics =====")
    print("Cycle A (A -> B -> A)")
    print(f"SSIM  : {avg['ssim_A']:.4f}")
    print(f"PSNR  : {avg['psnr_A']:.4f}")
    print(f"LPIPS : {avg['lpips_A']:.4f}\n")

    print("Cycle B (B -> A -> B)")
    print(f"SSIM  : {avg['ssim_B']:.4f}")
    print(f"PSNR  : {avg['psnr_B']:.4f}")
    print(f"LPIPS : {avg['lpips_B']:.4f}\n")

    print("===== Translation Quality (Fake vs Real) =====")
    print("A -> B (fake_B vs real_B)  <-- *Important!*")
    print(f"SSIM  : {avg['ssim_AB']:.4f}")
    print(f"PSNR  : {avg['psnr_AB']:.4f}")
    print(f"LPIPS : {avg['lpips_AB']:.4f}\n")

    print("B -> A (fake_A vs real_A)")
    print(f"SSIM  : {avg['ssim_BA']:.4f}")
    print(f"PSNR  : {avg['psnr_BA']:.4f}")
    print(f"LPIPS : {avg['lpips_BA']:.4f}")

if __name__ == "__main__":
    main()