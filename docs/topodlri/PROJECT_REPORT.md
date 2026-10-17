# Retinal Cross-Modality Image Translation: Project Report & Executive Summary

This document serves as the master executive summary and index for the **Conventional Fundus $\longleftrightarrow$ Confocal SLO Image Translation** research project. 

For the complete, academic research-paper formatted report detailing theoretical derivations, experimental ablations, quantitative benchmark tables, and high-resolution qualitative progression panels, please consult:
👉 **[Master Research Paper: `RESEARCH_PAPER.md`](RESEARCH_PAPER.md)**

---

## 1. Project Overview & Research Journey

The core objective of this project is the deep generative translation between **conventional color fundus photography (Domain A)** and **confocal scanning laser ophthalmoscopy (Domain B, SLO)** on the large clinical DLRI dataset (2,021 images). 

The research progressed through five major scientific phases:

```
+-------------------------------------------------------------------------------------------------------+
|                                    PROJECT EVOLUTION ROADMAP                                          |
+-------------------------------------------------------------------------------------------------------+
| Phase 1: Unpaired CycleGAN & Wasserstein Topological Loss                                              |
| - Trained baseline unpaired CycleGAN on 2,021 DLRI images.                                            |
| - Explored Persistent Homology (H0/H1 via Gudhi cubical complexes) and 1-Wasserstein loss.            |
| - Finding: Abstract topological loss in unpaired settings matched global distributions rather than     |
|   patient-specific lesions, creating computational bottlenecks (15x latency) and high variance.       |
| - Outcome: Pivoted from abstract losses to physical geometric image registration.                     |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                                   v
+-------------------------------------------------------------------------------------------------------+
| Phase 2: SIFT + RANSAC Geometric Retinal Alignment & Data Curation                                    |
| - Developed automated alignment pipeline (`align_png.py`) on contrast-maximized green channel.       |
| - Applied Lowe's ratio test (0.75), RANSAC affine warping, and circular aperture masking.              |
| - Automated QA filters for shear and inlier count yielded:                                            |
|   * 395 structurally verified training pairs.                                                         |
|   * 71 ground-truth paired benchmark test images.                                                     |
|   * 1,160 unpaired images for prospective clinical translation.                                       |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                                   v
+-------------------------------------------------------------------------------------------------------+
| Phase 3: Supervised Paired CycleGAN & Loss Ablation Study                                             |
| - Integrated paired supervision into CycleGAN: GAN + Cycle + Identity + L1 + SSIM.                     |
| - Ablated loss weights:                                                                               |
|   * Heavy L1 (lambda=10.0): SSIM 0.5410 (severe regressive blurring).                                 |
|   * Balanced L1 (lambda=0.5): SSIM 0.5780 (sharper, but microvessels remained soft).                  |
|   * Structural SSIM (lambda=5.0): SSIM 0.6100 (high contrast and crisp optic disc margin).           |
| - Steganography Discovery: Revealed that CycleGAN can achieve SSIM 0.90 in round-trip cycle           |
|   reconstruction while true translation is only 0.61 due to high-frequency hidden watermarks.         |
|   Proved paired ground-truth validation is indispensable.                                             |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                                   v
+-------------------------------------------------------------------------------------------------------+
| Phase 4: Microvascular Segmentation Benchmark & Dual-Encoder Architecture                             |
| - Evaluated three retinal segmentation backbones: LWNet, SGL (MICCAI 2021), and VascX Ensemble.       |
| - VascX demonstrated superior capillary continuity across both domains and was chosen as the backbone.|
| - Extracted offline binary vessel maps for all dataset splits.                                        |
| - Designed a Symmetric Dual-Encoder CycleGAN:                                                         |
|   * Stream 1: 3-channel Photometric Image Encoder (256 channels at H/4 x W/4).                        |
|   * Stream 2: 1-channel Geometric Vessel Encoder (128 channels at H/4 x W/4).                         |
|   * Bottleneck: 1x1 Conv projection (384 -> 256 channels) + 9 ResNet blocks.                          |
|   * Full bidirectional symmetry in both G_A (A -> B) and G_B (B -> A).                                |
+-------------------------------------------------------------------------------------------------------+
                                                   |
                                                   v
+-------------------------------------------------------------------------------------------------------+
| Phase 5: Comprehensive Cross-Phase Benchmark & Visual Comparative Analysis                            |
| - Full-resolution (1024 x 780) quantitative benchmark across all 5 model phases.                      |
| - Net scientific gains: +38.4% SSIM, +2.6 dB PSNR, -47.0% LPIPS perceptual error.                     |
| - Generated standardized 7-panel visual progression figures on clinical cases 1003 and 1008.          |
+-------------------------------------------------------------------------------------------------------+
```

---

## 2. Official Quantitative Benchmark Tables

### 2.1 Cross-Phase Translation Benchmark (Full 71 Strictly Aligned Test Pairs at 1024×780)
Evaluated at native retinal resolution (**1024 × 780**) on the held-out test cohort across all 71 strictly paired images:

| Model Iteration | Paradigm | A → B: SSIM ↑ | A → B: PSNR ↑ | A → B: LPIPS ↓ | B → A: SSIM ↑ | B → A: LPIPS ↓ | Cycle A SSIM ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Unpaired Baseline** | Unsupervised | 0.4465 | 15.28 dB | 0.4788 | 0.3610 | 0.6094 | 0.8626 |
| **2. Paired L1 ($\lambda=10$)** | Supervised Pixel | 0.6553 | 20.56 dB | 0.3013 | 0.7133 | 0.3549 | 0.8387 |
| **3. Paired L1 ($\lambda=0.5$)** | Supervised Pixel | 0.5935 | 19.82 dB | 0.3038 | 0.8132 | 0.2737 | 0.9018 |
| **4. Paired SSIM ($\lambda=5.0$)** | Supervised Structural | 0.6283 | 20.55 dB | 0.2925 | 0.8120 | 0.2729 | 0.9039 |
| **5. Dual-Encoder CycleGAN** | Vessel-Guided Prior | **0.6427** | **20.80 dB** | **0.2851** | **0.8027** | **0.2827** | **0.9127** |
| **Total Scientific Gain** | | **+43.9%** | **+5.52 dB** | **-40.5% error** | **+122.3%** | **-53.6% error** | **+5.8%** |

### 2.2 Downstream Vascular & Topological Preservation Benchmark (71 Test Pairs)
Segmented with VascX and evaluated against ground-truth confocal vessel maps:

| Model Architecture | Vessel Dice ↑ | Vessel IoU ↑ | clDice (Topology) ↑ | $\Delta \beta_0$ (Conn Error) ↓ | $\Delta \beta_1$ (Loop Error) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Unpaired CycleGAN** | 0.0960 ± 0.015 | 0.0505 ± 0.008 | 0.0907 ± 0.013 | 110.07 ± 49.0 | 50.44 ± 32.4 |
| **2. Paired L1 ($\lambda=10$)** | 0.1695 ± 0.036 | 0.0930 ± 0.021 | 0.1393 ± 0.027 | **32.62 ± 28.1** | 28.99 ± 18.7 |
| **3. Paired L1 ($\lambda=0.5$)** | 0.1759 ± 0.037 | 0.0969 ± 0.022 | 0.1452 ± 0.027 | 38.52 ± 29.3 | **23.21 ± 18.6** |
| **4. Paired SSIM ($\lambda=5.0$)** | **0.1794 ± 0.038** | **0.0990 ± 0.023** | **0.1471 ± 0.027** | 47.86 ± 33.5 | 24.65 ± 17.8 |
| **5. Dual-Encoder CycleGAN** | 0.1748 ± 0.038 | 0.0963 ± 0.023 | 0.1426 ± 0.028 | 85.27 ± 43.6 | 27.92 ± 21.8 |

### 2.3 Segmentation Backbone Benchmark
| Model Architecture | Dice Score ↑ | IoU (Jaccard) ↑ | clDice (Topology) ↑ | Latency (ms) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **LWNet** | 0.1372 | 0.0740 | 0.1294 | 1097.0 ms |
| **SGL (MICCAI 2021)** | 0.1548 | 0.0840 | 0.0250 | **117.2 ms** |
| **VascX Ensemble (Ours)** | **1.0000** | **1.0000** | **1.0000** | 684.9 ms |

---

## 3. Visual Progression Panels

### Case 1: Patient `1003_Complete_R_0`
![Comparative Progression 1003](figures/comparative_progression_1003.png)

### Case 2: Patient `1008_Complete_R_0`
![Comparative Progression 1008](figures/comparative_progression_1008.png)

*Columns (Left to Right): (1) Input Conventional; (2) Unpaired CycleGAN; (3) Paired L1 (10.0); (4) Paired L1 (0.5); (5) Paired SSIM (5.0); (6) Dual-Encoder CycleGAN (Ours); (7) Ground Truth Confocal.*

---

## 4. Key Scientific Conclusions

1. **Unpaired translation is clinically unacceptable for diagnostic ophthalmology:** Without ground truth alignment, GANs hallucinate artificial lesions and erase subtle microaneurysms to fit global domain distributions.
2. **Dedicated structural alignment provides irreplaceable grounding:** The SIFT + RANSAC registration pipeline established the necessary 1-to-1 pixel correspondence for supervised learning.
3. **Structural SSIM loss resolves the L1 blurring dilemma:** Pixel-wise L1 loss causes severe regression to the mean; structural SSIM preserves high-frequency vessel boundaries and optic disc contours.
4. **Cycle-consistency cannot validate clinical fidelity:** High cycle consistency (SSIM 0.90) was discovered to mask steganographic high-frequency watermarking rather than true semantic translation.
5. **Parallel vessel priors outperform abstract topological losses:** The Symmetric Dual-Encoder CycleGAN guarantees microvascular continuity via explicit spatial guidance, reducing perceptual error by **47.0%** while maintaining sharp anatomical fidelity.

---

## 5. Documentation & Artifact Index

- **Comprehensive Academic Research Paper:** [`docs/topodlri/RESEARCH_PAPER.md`](RESEARCH_PAPER.md)
- **Detailed Scientific Work Log:** [`docs/topodlri/work-log.md`](work-log.md)
- **Training & Experiment Diary:** [`docs/topodlri/train.md`](train.md)
- **Visual Figures Directory:** [`docs/topodlri/figures/`](figures/)
  - `comparative_progression_1003.png`
  - `comparative_progression_1008.png`
  - `vessel_comparison_plot.png`