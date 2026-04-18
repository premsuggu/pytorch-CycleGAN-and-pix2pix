import os
import torch
import lpips
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torchvision.transforms as transforms

# -----------------------
# Paths (based on your repo)
# -----------------------

BASE = "results/conventional2confocal_cyclegan/test_100/images"

paths = {
    "real_A": os.path.join(BASE, "16_real_A.png"),
    "real_B": os.path.join(BASE, "16_real_B.png"),
    "fake_A": os.path.join(BASE, "16_fake_A.png"),
    "fake_B": os.path.join(BASE, "16_fake_B.png"),
    "rec_A": os.path.join(BASE, "16_rec_A.png"),
    "rec_B": os.path.join(BASE, "16_rec_B.png"),
}

# -----------------------
# Load images
# -----------------------

def load_image_np(path):
    return np.array(Image.open(path).convert("RGB"))

def load_image_tensor(path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

real_A = load_image_np(paths["real_A"])
real_B = load_image_np(paths["real_B"])
fake_A = load_image_np(paths["fake_A"])
fake_B = load_image_np(paths["fake_B"])
rec_A = load_image_np(paths["rec_A"])
rec_B = load_image_np(paths["rec_B"])

# -----------------------
# SSIM
# -----------------------

ssim_A = ssim(real_A, rec_A, channel_axis=2)
ssim_B = ssim(real_B, rec_B, channel_axis=2)

# -----------------------
# PSNR
# -----------------------

psnr_A = psnr(real_A, rec_A)
psnr_B = psnr(real_B, rec_B)

# -----------------------
# LPIPS
# -----------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = lpips.LPIPS(net='vgg').to(device)

def compute_lpips(img1, img2):
    t1 = load_image_tensor(img1).to(device)
    t2 = load_image_tensor(img2).to(device)
    return loss_fn(t1, t2).item()

lpips_A = compute_lpips(paths["real_A"], paths["rec_A"])
lpips_B = compute_lpips(paths["real_B"], paths["rec_B"])

# perceptual translation quality
lpips_AB = compute_lpips(paths["real_B"], paths["fake_B"])
lpips_BA = compute_lpips(paths["real_A"], paths["fake_A"])

# -----------------------
# Print results
# -----------------------

print("\n===== Cycle Reconstruction Metrics =====")

print(f"Cycle A (A -> B -> A)")
print(f"SSIM  : {ssim_A:.4f}")
print(f"PSNR  : {psnr_A:.4f}")
print(f"LPIPS : {lpips_A:.4f}")

print()

print(f"Cycle B (B -> A -> B)")
print(f"SSIM  : {ssim_B:.4f}")
print(f"PSNR  : {psnr_B:.4f}")
print(f"LPIPS : {lpips_B:.4f}")

print("\n===== Translation Perceptual Distance =====")

print(f"A -> B LPIPS (fake_B vs real_B): {lpips_AB:.4f}")
print(f"B -> A LPIPS (fake_A vs real_A): {lpips_BA:.4f}")