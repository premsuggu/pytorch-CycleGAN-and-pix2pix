# Detailed Work Log: Phase 2 CycleGAN Development & Debugging

This document provides a deep, technical breakdown of the issues encountered, the root causes discovered, and the exact architectural and script-level changes made to transition the project to **Phase 2 (Supervised Aligned Training)**.

---

## 1. Resolving the Blurry Artifacts via Batch Size Restrictions
### The Problem
During recent training runs, the generated confocal images (`fake_B`) exhibited severe blurriness, losing the sharp, high-frequency structural details that are critical in retinal imaging.

### Root Cause Analysis
Upon reviewing the training scripts, it was discovered that the `--batch_size` had been increased from `1` to a higher value (e.g., `8`). The CycleGAN architecture relies fundamentally on `InstanceNorm2d` layers (Instance Normalization) rather than Batch Normalization. 
Instance Normalization computes the mean and variance across the spatial dimensions independently for each channel and each image. However, when the batch size is increased, depending on the specific PyTorch implementation or how the gradients aggregate across diverse medical images, the network struggles to maintain localized feature contrast, leading to heavily blurred outputs.

### The Fix
- We strictly enforced `--batch_size 1` in all SLURM submission scripts (e.g., `python-scripts/submit_train_unaligned.sh`).
- We submitted a safe, isolated smoke test to the cluster which immediately confirmed that restricting the batch size to 1 restored the sharp structural integrity of the generated images.

---

## 2. Transition to Paired Supervised Learning
### The Problem
The standard CycleGAN operates on unpaired datasets (originally 15 images per domain), relying entirely on adversarial loss and cycle-consistency. This makes it very difficult for the network to perfectly align fine blood vessels and optic disc boundaries.

### The Enhancement
The dataset was expanded to a pre-aligned dataset (`datasets/fundus_aligned`) containing **213 structurally paired images**, mapped using SIFT feature matching and RANSAC affine transformations. Because the data was now structurally aligned pixel-for-pixel, we could safely inject supervised constraints into the training loop.

### Codebase Modifications
We heavily modified `models/cycle_gan_model.py` to support supervised gating:
- **L1 Loss:** We added a `--lambda_L1` argument. Inside `backward_G()`, we computed a direct pixel-wise `L1Loss(fake_B, real_B)` and `L1Loss(fake_A, real_A)`. This forced the generator to physically match the ground truth image, not just match the domain style.
- **SSIM Loss:** We added a `--lambda_SSIM` argument to directly optimize the Structural Similarity Index between the generated image and the ground truth during the backward pass, pushing the network to preserve perceptual structures.
- **Backward Compatibility:** All modifications were gated behind `if self.opt.lambda_L1 > 0.0:` to ensure that standard, unpaired CycleGAN training would still function flawlessly if the dataset was swapped back.

---

## 3. Training Loop Early-Exit & Bash Script Debugging
### The Problem
When attempting to resume training from an existing checkpoint using `--continue_train` and `--epoch_count 201`, the submitted SLURM job (`590453`) finished instantly in under 3 minutes with an Exit Code of `0`, without performing a single iteration of training.

### Root Cause Analysis
CycleGAN's `train.py` calculates the absolute maximum number of epochs dynamically: `total_epochs = n_epochs + n_epochs_decay`. 
The submission script had `--n_epochs 100` and `--n_epochs_decay 100` (yielding a maximum of 200 epochs). Because we passed `--epoch_count 201`, the internal training loop evaluated `for epoch in range(201, 200 + 1)`—which is an empty range. The script correctly assumed training was completely finished and cleanly exited.

### The Fix
- **Epoch Adjustments:** We updated the script to `--n_epochs 150` and `--n_epochs_decay 150` (total 300 epochs), allowing it to successfully resume from 201 and train for another 100 epochs.
- **Bash Syntax Fixes:** While refactoring the script for a fresh model run (`cyclegan_unaligned_fresh` with `200 + 200` epochs), we discovered and fixed a critical bash syntax error. Commented lines (`# --continue_train \`) had been placed in the middle of a line-continuation block, which inherently breaks the bash interpreter. 
- **Safe Smoke Testing:** Before overwriting the actual model weights, we duplicated the entire `checkpoints/cyclegan_unaligned` directory to a temporary `_test` directory. We ran a bounded SLURM test that successfully loaded the weights, iterated past epoch 225, and saved. Once verified, the test files were purged, and the real job (`594534`) was submitted.

---

## 4. Total Overhaul of Evaluation Metrics Pipeline
### The Problem
The original `compute_metrics.py` was hardcoded to evaluate a single specific image (e.g., `16_real_A.png`). When we attempted to scale it to evaluate an entire folder, the script violently crashed with:
`ValueError: Input images must have the same dimensions.`

### Root Cause Analysis
The raw input fundus images had a spatial resolution of **778 × 1024**. The CycleGAN generator network uses strided convolutions (downsampling and upsampling) which require the spatial dimensions to be cleanly divisible by the scaling factor (usually 4). Because 778 is not perfectly divisible by 4, the generator's reflection padding outputs a dimension of **780 × 1024**.
When `skimage.metrics.ssim` attempted to compare the original `real_A` (778) to the generated `fake_B` (780), the 2-pixel dimension mismatch triggered a fatal error.

### The Fix
We completely rewrote the `compute_metrics.py` file from the ground up:
- **Dynamic Dimension Matching:** We introduced a `target_shape` parameter to the image loading functions. The script now intercepts the shape of the `real` ground truth image (778), and strictly forces the generated `fake` and `rec` images to resize/crop back to exactly 778 × 1024 before calculating any metrics.
- **Full Directory Iteration:** The script now accepts a `--dir` argument (defaulting to `results/cyclegan_paired_ssim/test_200/images`), uses `glob` to find all unique image prefixes, and iteratively calculates metrics for the entire test set.
- **Metric Paradigm Shift:** The original script only calculated metrics for Cycle Reconstruction (`rec_A` vs `real_A`). We fundamentally shifted the evaluation to calculate SSIM, PSNR, and LPIPS directly between the generated translation and the ground truth (`fake_B` vs `real_B`).
- **Performance:** Integrated `tqdm` progress bars, as evaluating the LPIPS VGG network sequentially across 50 high-resolution images is highly compute-intensive on CPU.

---

## 5. Decoding the "Inflated" Reconstruction Metrics (Steganography)
### The Observation
When we ran the new metrics pipeline, we noticed a massive discrepancy:
- Cycle Reconstruction (`A -> B -> A`): **SSIM 0.90**
- Direct Translation (`A -> B`): **SSIM 0.61**

### The Explanation
This massive gap was flagged and documented as a known exploit in CycleGAN architectures, commonly referred to as **Steganography**.
- The Cycle-Consistency loss is weighted extremely heavily during training (`lambda=10.0`), meaning the network is massively penalized if it fails to reconstruct the original image perfectly.
- To bypass this difficult constraint, Generator A learns to "cheat". It hides the exact structural details of the original conventional image (`real_A`) inside mathematically imperceptible, high-frequency noise within the generated confocal image (`fake_B`).
- Generator B then largely ignores the actual visual "confocal" properties of the image. Instead, it reads the hidden steganographic noise and perfectly decodes it back into the reconstructed image (`rec_A`).

### The Conclusion
This finding definitively proved that **Cycle Reconstruction metrics (0.90) are completely unreliable** for evaluating how well the model is actually translating domains. Moving forward, the project relies exclusively on the direct Translation Quality metrics (`fake` vs `real` = 0.61) introduced in our overhauled `compute_metrics.py` script.
