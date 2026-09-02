import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

def get_test_images(data_dir, num_images=3, seed=42):
    domain_A_dir = Path(data_dir) / 'trainA'
    domain_B_dir = Path(data_dir) / 'trainB'
    all_A_imgs = sorted(list(domain_A_dir.glob('*.png')))
    random.seed(seed)
    chosen_A = random.sample(all_A_imgs, num_images)
    chosen_B = [domain_B_dir / p.name for p in chosen_A]
    return chosen_A, chosen_B

def load_image_for_plot(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def plot_comparison(chosen_A, chosen_B, preds_A, preds_B, save_path):
    fig, axes = plt.subplots(len(chosen_A), 4, figsize=(16, 4 * len(chosen_A)))
    cols = ['Original A', 'Vessel Map A', 'Original B', 'Vessel Map B']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col, fontsize=14)
        
    for i in range(len(chosen_A)):
        img_A = load_image_for_plot(chosen_A[i])
        img_B = load_image_for_plot(chosen_B[i])
        
        axes[i, 0].imshow(img_A)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(preds_A[i], cmap='gray')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(img_B)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(preds_B[i], cmap='gray')
        axes[i, 3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

