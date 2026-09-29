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

---

## 6. Dual-Encoder CycleGAN Architecture for Retinal Structure Preservation
### The Motivation & Concept
In standard CycleGAN image-to-image translation between conventional fundus photography (Domain A) and confocal scanning ophthalmoscopy (Domain B), fine anatomical structures—particularly the retinal vascular tree—are susceptible to hallucination, blurring, or discontinuous vessel breakage. 

To enforce explicit structural guidance, we formulated a **Dual-Encoder Generator** architecture:
1. **Stream 1 (Image Encoder):** Takes the full RGB retinal image (3 channels), learning domain-specific photometric features, illumination gradients, and tissue texture.
2. **Stream 2 (Vessel Encoder):** Takes the segmented binary retinal vessel map (1 channel), learning dedicated vascular geometry, vessel bifurcation topology, and caliber variations.
3. **Bottleneck Fusion:** Feature maps from both streams at the lowest spatial resolution ($H/4 \times W/4$, 256 channels each) are concatenated along the channel dimension (512 channels) and projected back to 256 channels via a $1 \times 1$ convolution, followed by 9 residual blocks and standard transposed convolution decoding.

### Generator Symmetry: Why Both $G_A$ and $G_B$ Must Have Dual Encoders
A fundamental architectural design question arose: should only the forward generator $G_A$ ($A \to B$) have dual encoders, or must both generators ($G_A$ and $G_B$) share the identical dual-encoder topology?

We established both mathematical and physical requirements for strict generator symmetry:
- **Cycle-Consistency Constraint:** CycleGAN relies on bidirectional closed-loop cycle consistency: $A \to G_A(A, \text{vessel}_A) \to \text{fake\_}B \to G_B(\text{fake\_}B, \text{vessel}_B) \to \text{rec\_}A$, and symmetrically for $B \to \text{fake\_}A \to \text{rec\_}B$. If $G_B$ lacked a vessel encoder, it would be starved of structural prior when reconstructing Domain B, producing degraded reconstructions that destabilize the cycle-consistency loss $\mathcal{L}_{\text{cyc}}$ and corrupt the gradients flowing back into $G_A$.
- **Physical Domain Reality:** Retinal blood vessels are anatomical structures physically present in both conventional and confocal modalities. Providing vessel priors to both directions ensures topological consistency throughout both the translation and identity mappings.

### Code Implementation
We implemented the dual-encoder pipeline across the codebase:
1. **`models/networks.py`:**
   - Implemented `DualEncoderResnetGenerator`: contains `encoder_img` (7x7 conv + 2 strided downsampling convs), `encoder_vessel` (parallel downsampling branch with `vessel_nc=1`), `fusion` ($1 \times 1$ conv: $2 \times \text{ngf} \to \text{ngf}$ with InstanceNorm and ReLU), 9 ResNet blocks, and 2 upsampling transposed convolutions to output the 3-channel translated image.
   - Updated `define_G` with `--netG dual_encoder_resnet_9blocks` and `--netG dual_encoder_resnet_6blocks` supporting custom `vessel_nc`.
2. **`options/base_options.py`:**
   - Added `--vessel_nc` flag (default: 1) and registered dual-encoder network options.
3. **`models/cycle_gan_model.py`:**
   - Integrated dual-encoder forward routing in `forward()`: `self.fake_B = self.netG_A(self.real_A, self.real_A_vessel)` and `self.fake_A = self.netG_B(self.real_B, self.real_B_vessel)`.
   - Updated cycle-consistency reconstructions: `self.rec_A = self.netG_B(self.fake_B, self.real_A_vessel)` and `self.rec_B = self.netG_A(self.fake_A, self.real_B_vessel)`.
   - Updated identity loss mapping: `self.netG_A(self.real_B, self.real_B_vessel)` and `self.netG_B(self.real_A, self.real_A_vessel)`.
   - Bound input unpacking in `set_input()` to read `A_vessel` and `B_vessel` onto the target device when available.
4. **`data/paired_cyclegan_dataset.py` & `data/unaligned_dataset.py`:**
   - Added automatic detection of corresponding vessel directories (`<split>_vessel/`).
   - Synchronized random data augmentations (`transform_params`: crop position and horizontal flips) so every geometric transformation applied to an image is identically applied to its companion vessel mask.

---

## 7. Retinal Vessel Map Extraction with VascX Ensemble & HPC Optimization
### Model Selection: VascX
Following empirical evaluation of multiple segmentation models (SGL, LWNet, and VascX), VascX was selected as the state-of-the-art vessel segmentation backbone. Its TorchScript ensemble weights (`segmentation/models/vessels/vessels_july24.pt`, 337 MB) were utilized for offline batch extraction across the complete dataset.

### Critical Finding: Dynamic Range Scaling ([0, 1] vs [0, 255])
During initial inspection of raw VascX predictions, masks appeared completely black under standard image viewers. Investigation revealed that VascX outputs raw binary pixel values $\{0, 1\}$ (not $\{0, 255\}$):
- When loaded via standard `transforms.ToTensor()`, integer $\{0, 1\}$ was divided by 255, producing floating point values $0.0$ and $0.00392$.
- Subsequent `transforms.Normalize((0.5,), (0.5,))` shifted these values to $-1.0$ and $-0.99216$, destroying image contrast and starving the neural network of gradient signal.
- **The Fix:** In `segmentation/extract_all_vessel_maps.py` and both dataset loaders (`paired_cyclegan_dataset.py` and `unaligned_dataset.py`), we incorporated dynamic range checks: if `vessel_img.getextrema()[1] <= 1`, the image is automatically rescaled using `Image.eval(vessel_img, lambda p: 255 if p > 0 else 0)`. Extracted masks on disk and tensors fed to the network now possess the full $[0, 255]$ dynamic range (normalized to $[-1.0, +1.0]$).

### SLURM Resource & Partition Optimization (`sgpu` vs `gpu`)
When attempting to run batch extraction on the HPC GPU cluster, the initial job was blocked in state `PENDING (Reason=QOSMinGRES)`:
- **Root Cause:** The default SLURM partition `gpu` is configured with `QoS=gpu_qos`, which strictly enforces `MinTRES = gres/gpu=2` (a minimum of 2 GPUs per job). Because single-GPU jobs requested `--gres=gpu:1`, SLURM refused scheduling under `gpu_qos`.
- **Solution:** Investigation of cluster partitions revealed the dedicated `sgpu` partition (`PartitionName=sgpu`), whose default QOS is `sgpu_qos` (`MinTRES = gres/gpu=1`, `MaxTRES = gres/gpu=1`, walltime up to 2 days). Submitting single-GPU jobs with `--partition=sgpu` immediately dispatched the job to an idle NVIDIA A100 80GB PCIe node (`rsgpu022`).

### End-to-End Pipeline Execution
- **Extraction Job (`Job 629265`):** Running on `rsgpu022` with batch size 4 across `trainA` (395 images), `trainB` (395 images), `testB` (71 images), and `testA` (1160 images) at ~0.44s/image.
- **Chained Training Job (`Job 629268`):** Submitted with SLURM dependency `--dependency=afterok:629265`. As soon as vessel extraction concludes with ExitCode 0, the Dual-Encoder CycleGAN training automatically initializes with 150 regular + 150 decay epochs on the full dataset.

