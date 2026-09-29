# Work Log: Phase 2 CycleGAN Improvements

This document serves as a comprehensive log of the findings, debugging steps, and architectural improvements made to transition the project into **Phase 2 (Supervised Aligned Training)**.

---

## 1. The Blurry Artifacts Issue & Batch Size Discovery
**The Problem:** The generated confocal images began exhibiting severe blurriness and degraded quality after adjusting some training hyperparameters.
**Investigation & Finding:** It was discovered that the `batch_size` had been increased to `8` in the training script. CycleGAN uses `InstanceNorm` by default, which normalizes activations across spatial dimensions per image. However, when combined with a larger batch size in this specific implementation, it resulted in aggregated statistics or gradient conflicts that caused the network to output heavily blurred translations.
**The Fix:** We permanently reverted and enforced `--batch_size 1`. A live SLURM smoke test confirmed that reverting to a batch size of 1 immediately restored the sharp, distinct structural details of the images.

## 2. Transition to Paired Supervised Learning
**Context:** The original dataset was small (15 unpaired images). The project evolved to use a heavily expanded, pre-aligned dataset (`datasets/fundus_aligned`) containing 213 structurally paired images (aligned using SIFT + RANSAC).
**Implementation:** Since the dataset was now perfectly aligned, we could explicitly teach the network to generate the *exact* target image rather than just matching domain statistics.
- Edited `models/cycle_gan_model.py` to inject **supervised constraints**.
- Added `--lambda_L1` to compute a pixel-wise L1 loss between the generated image (`fake_B`) and the ground truth (`real_B`).
- Added `--lambda_SSIM` to directly optimize for Structural Similarity between the generated image and ground truth.
- These additions were cleanly gated by `if lambda_X > 0.0:` so the original standard CycleGAN functionality remained intact.

## 3. Job Submission Script & Training Loop Fixes
**The Problem:** When attempting to resume a training run (`--continue_train`), the SLURM job finished instantly in under 3 minutes with an Exit Code of 0 without doing any training.
**Investigation & Finding:** The `train.py` loop limits total epochs to `n_epochs + n_epochs_decay`. The script was trying to resume from `--epoch_count 201`, but the epochs were still set to `100 + 100 = 200`. Since `201 > 200`, the training loop gracefully exited immediately.
**The Fix:** 
- Updated `python-scripts/submit_train_unaligned.sh` to `--n_epochs 150` and `--n_epochs_decay 150` (300 total) to allow continuation. 
- Created an isolated dummy checkpoint directory to run a safe smoke test to confirm it worked before running it on the real checkpoints.
- Later, we configured a fresh start script (`cyclegan_unaligned_fresh`) for 400 epochs (`200 + 200`). Fixed a critical bash syntax error where commented lines (`# --continue_train \`) broke the bash line continuation (`\`).

## 4. Total Overhaul of Evaluation Metrics Pipeline
**The Problem:** `compute_metrics.py` was hardcoded to a single image and crashed with a `ValueError: Input images must have the same dimensions.` when computing SSIM on the full test dataset.
**Investigation & Finding:** The generator network internally uses downsampling and upsampling layers that expect inputs to be divisible by 4. Because the original images were `(778, 1024)`, the generator zero-padded the output to `(780, 1024)`. This 2-pixel mismatch broke `skimage.metrics`.
**The Fix:** 
- Completely rewrote `compute_metrics.py`.
- Added dynamic directory parsing (`--dir`) using `glob` to evaluate an entire test set automatically.
- Implemented automatic target-shape resizing/cropping to perfectly align the generated `(780, 1024)` images back to the ground truth `(778, 1024)` dimensions before metric calculation.
- Shifted the evaluation focus: Added SSIM, PSNR, and LPIPS calculations for direct **Translation Quality (Fake vs Real)** rather than just Cycle Reconstruction.
- Integrated `tqdm` progress bars for the slow LPIPS VGG network evaluation.

## 5. Decoding the "Inflated" Reconstruction Metrics (Steganography)
**The Observation:** The A -> B -> A cycle reconstruction yielded a massive SSIM of **0.90**, but the actual A -> B translation quality was only **0.61**.
**The Analysis:** We documented and explained this phenomenon, which is a known exploit in CycleGAN architectures. 
- Because the cycle-consistency loss is weighted so heavily (`lambda=10.0`), the generator (`G_A`) learns to "cheat" the loss.
- It hides the exact structural details of the original image (`real_A`) inside invisible, high-frequency noise within the translated image (`fake_B`).
- The reverse generator (`G_B`) ignores the visible image and reads the hidden steganographic noise to perfectly decode the original image.
- **Conclusion:** Cycle reconstruction scores (0.90) are largely unreliable for evaluating the true model quality. The direct Fake vs Real translation metrics (0.61) introduced in our new pipeline are the only true measure of translation success.

## 6. Comprehensive Documentation Updates
- Updated `AGENT.md` to mandate `batch_size=1` and document the new loss parameters and scripts.
- Updated `docs/topodlri/PROJECT_REPORT.md` and `docs/topodlri/train.md` with a "Phase 2" addendum, ensuring anyone reading the previous history understands how the codebase evolved to solve the problems they were facing.
