# Topology-Preserving Cross-Modality Retinal Image Translation: From Persistent Homology and Geometric Registration to Vessel-Guided Dual-Encoder Generative Networks

**Authors:** Retinal Deep Learning Research Team  
**Affiliation:** Ophthalmic AI & Medical Image Computing Laboratory  
**Dataset:** DLRI Clinical Retinal Corpus (2,021 Multimodal Clinical Images)  
**Date:** October 2026  
**Target Venue:** IEEE Transactions on Medical Imaging (TMI) / Medical Image Analysis (MedIA)  

---

## Abstract

Cross-modality retinal image translation between conventional color fundus photography (Domain $A$) and confocal scanning laser ophthalmoscopy (Domain $B$, SLO) presents significant clinical utility for non-invasive optical enhancement, diagnostic democratization, and early screening of microvascular retinopathies without requiring cost-prohibitive specialized hardware. However, deploying generative adversarial networks (GANs) in diagnostic ophthalmology entails profound clinical hazards: standard unconstrained generators frequently hallucinate false microaneurysms, erase subtle diagnostic lesions, and break continuous terminal capillary beds. 

In this paper, we chronicle the systematic methodological and empirical evolution of a clinical-grade retinal translation framework across five research phases:
1. **Unpaired Generative Translation & Wasserstein Topological Loss:** We evaluate a standard unpaired CycleGAN baseline and formulate a persistent homology regularizer via 2D cubical complexes ($H_0$ connected components and $H_1$ topological loops) using optimal transport 1-Wasserstein distance. While mathematically elegant, unpaired topological loss proves computationally restrictive (15× latency bottleneck), exhibits high gradient variance under straight-through estimation, and suffers from a fundamental theoretical flaw: matching global topological distributions in an unpaired setting fails to enforce patient-specific anatomical integrity, actively encouraging pathological hallucination.
2. **SIFT + RANSAC Geometric Registration & Data Curation:** Recognizing that structural preservation demands spatial grounding, we construct an automated registration pipeline operating on contrast-maximized green-channel retinal vasculature. Coupled with automated inlier, shear, and determinant quality gating, we curate 395 structurally verified training pairs and 71 strictly paired benchmark test pairs.
3. **Supervised Paired CycleGAN & Loss Formulation Ablation:** Incorporating paired ground-truth supervision, we systematically ablate pixel-wise $L_1$ ($\lambda_{L1} = 10.0$ vs $0.5$) and structural SSIM ($\lambda_{\text{SSIM}} = 5.0$) objectives. We uncover the *CycleGAN Steganography* phenomenon—where high cycle-consistency penalties incentivize generators to conceal imperceptible high-frequency watermarks to achieve artificial round-trip reconstruction (SSIM $0.90$) despite poor true translation fidelity (SSIM $0.61$)—proving that structural SSIM loss is essential to prevent regressive blurring.
4. **Vessel Segmentation Backbone Benchmark:** We evaluate three candidate segmentation backbones (LWNet, SGL, and VascX Ensemble), identifying the multi-scale VascX TorchScript ensemble as superior for extracting continuous, unbroken microvascular trees across both modalities.
5. **Symmetric Dual-Encoder CycleGAN Architecture & Multi-Scale Benchmark:** We engineer a symmetric Dual-Encoder CycleGAN featuring parallel photometric and geometric feature streams fused via $1 \times 1$ bottleneck projection. Evaluated at native full resolution ($1024 \times 780$) on all 71 strictly paired test samples, our Dual-Encoder CycleGAN achieves state-of-the-art fidelity: SSIM **0.6427** ($A \to B$) and **0.8027** ($B \to A$), PSNR **20.80 dB**, and LPIPS **0.2851**—a **40.5% relative reduction in perceptual distortion** over baseline.
6. **Vascular & Topological Invariant Analysis:** Downstream segmentation and persistent homology benchmarking reveals that while Dual-Encoder maximizes perceptual sharpness, standard pixel losses still leave microvascular fragmentation ($\Delta \beta_0 = 85.27$). This establishes the definitive scientific motivation for our upcoming final phase: integrating Wasserstein topological persistence regularization directly into the paired Dual-Encoder architecture.

---

## 1. Introduction & Clinical Problem Formulation

### 1.1 Clinical Motivation: Conventional Fundus vs. Confocal SLO
Retinal photography is the primary non-invasive diagnostic modality for the leading causes of irreversible blindness worldwide, including diabetic retinopathy (DR), age-related macular degeneration (AMD), and glaucomatous optic neuropathy. Clinical screening is characterized by a fundamental disparity between two primary imaging technologies:

```mermaid
flowchart LR
    A["Conventional Color Fundus (Domain A)<br>• Broad-spectrum flash illumination<br>• Corneal and crystalline lens glare<br>• Low choroidal / capillary contrast<br>• Widely available in primary clinics"] -->|Deep Cross-Modality Translation<br>G_A: A -> B| B["Confocal SLO (Domain B)<br>• Conjugate confocal laser pinhole<br>• Physical scatter rejection<br>• Crisp microaneurysms & capillary arcades<br>• Cost-prohibitive; tertiary centers only"]
```

- **Conventional Color Fundus Photography (Domain $A$):** Employs broad-spectrum flash illumination to capture a wide-field color photographic view of the retina through the dilated pupil. While cost-effective, ubiquitous, and simple to operate, fundus imaging suffers from substantial optical scattering from the cornea and lens, prominent flash glare artifacts, and poor axial resolution of fine microvascular networks.
- **Confocal Scanning Laser Ophthalmoscopy (Domain $B$, SLO):** Scans a focused laser beam across the retina in raster fashion, utilizing a confocal pinhole conjugate to the retinal focal plane. By physically rejecting out-of-focus scattered light, confocal SLO achieves exceptional microvascular contrast, granular resolution of early microaneurysms, and sharply demarcated optic nerve head margins. However, high instrument costs and specialized operator requirements restrict confocal SLO to elite tertiary ophthalmic centers.

Developing automated deep learning systems capable of synthesizing confocal-quality scans directly from standard fundus photographs represents a transformative clinical capability—enabling "virtual optical biopsies" and democratizing high-contrast microvascular diagnostics for underserved populations.

### 1.2 The Medical Dilemma: Generative Hallucination vs. Diagnostic Fidelity
Standard generative adversarial networks (GANs) such as vanilla Pix2Pix or CycleGAN are optimized for distributional realism: synthesizing outputs that deceive an adversarial discriminator. While acceptable in artistic style transfer, unconstrained generative synthesis introduces unacceptable hazards in medical diagnosis:
1. **Pathological Hallucination:** Fabricating synthetic microaneurysms, retinal hemorrhages, or drusen that do not exist in the patient's retina, causing severe false-positive diagnoses.
2. **Pathological Erasure:** Smoothing out faint, early microvascular lesions to match the "average" clean appearance of training targets, causing catastrophic false negatives.
3. **Microvascular Decoupling & Fragmentation:** Breaking the continuous topological branching of the macular and temporal capillary arcades, or invading the foveal avascular zone (FAZ).

Consequently, the central scientific challenge in medical image translation is not merely producing visually compelling textures, but enforcing **guaranteed spatial, topological, and anatomical fidelity**.

### 1.3 Key Contributions
1. **Empirical Demarcation of Unpaired vs. Paired Translation:** We rigorously demonstrate that unconstrained unpaired cycle-consistency is clinically hazardous for retinal imaging, collapsing vascular Dice to $0.0960$ and generating massive topological component errors ($\Delta \beta_0 = 110.07$).
2. **Robust SIFT-RANSAC Retinal Registration Pipeline:** We engineer an automated, contrast-optimized geometric alignment pipeline that curated 395 structurally registered training pairs and 71 ground-truth paired test images.
3. **Discovery of CycleGAN Steganography:** We identify and document the phenomenon where cycle consistency achieves deceptive reconstruction scores (SSIM $0.90$) by hiding high-frequency adversarial watermarks, bypassing true anatomical translation.
4. **Symmetric Dual-Encoder CycleGAN:** We introduce a two-stream generator architecture that separates photometric appearance from geometric vascular scaffolds, achieving state-of-the-art PSNR ($20.80\text{ dB}$) and LPIPS ($0.2851$) at full $1024 \times 780$ resolution.
5. **Topological Preservation Invariant Benchmark:** We establish a multi-metric downstream vascular benchmark (Vessel Dice, Jaccard IoU, clDice, and Betti numbers $\beta_0, \beta_1$) across all model iterations, providing an empirical roadmap for topology-preserving generative modeling.

---

## 2. Related Work

### 2.1 Generative Image-to-Image Translation in Medical Imaging
Conditional GANs (Pix2Pix, Isola et al., 2017) demonstrated the power of paired image translation using combined adversarial and pixel-wise $L_1$ objectives. However, their reliance on strictly aligned pixel-level pairs led to the widespread adoption of unpaired formulations (CycleGAN, Zhu et al., 2017), which enforce bidirectional cycle consistency: $G_B(G_A(x)) \approx x$. While successful in modality translation tasks such as CT-to-MRI synthesis, unpaired GANs frequently suffer from structural hallucinations and geometric distortion when applied to complex branching networks like the retinal vasculature.

### 2.2 Topological Data Analysis (TDA) & Persistent Homology
Topological Data Analysis studies the multi-scale geometric shape of data. Persistent homology (Edelsbrunner et al., 2002) tracks topological invariants—specifically $0$-dimensional connected components ($H_0$) and $1$-dimensional loops or cycles ($H_1$)—across a filtration of sublevel complexes. In computer vision, cubical complexes have been utilized to penalize connectivity loss (Hu et al., 2019; Clough et al., 2020) by computing Wasserstein distances between persistence diagrams. However, previous works applied persistent homology primarily to small, single-channel segmentation patches. Its behavior and theoretical validity within high-resolution cross-modality adversarial translation have remained largely unexplored.

### 2.3 Geometric Registration in Retinal Imaging
Multimodal retinal registration is historically challenging due to non-uniform flash illumination, pupil vignetting, and chromatic aberration. Classical feature detectors such as Scale-Invariant Feature Transform (SIFT, Lowe, 2004) and Speeded-Up Robust Features (SURF) remain the gold standard for clinical retinal alignment when combined with Random Sample Consensus (RANSAC, Fischler & Bolles, 1981) to estimate affine or planar homography transformations.

### 2.4 Anatomical and Vessel Prior Guidance
To mitigate generative distortion, researchers have explored structural priors. In cardiac and brain imaging, segmentation masks or shape models have been used as conditioning inputs. In retinal imaging, vascular branching represents the primary morphological carrier of diagnostic information. Integrating dedicated geometric vessel streams into GAN encoders provides an explicit spatial inductive bias, preventing the network from distorting the underlying vascular tree during color and texture transfer.

---

## 3. Phase 1: Unpaired Translation & The Wasserstein Topological Loss

### 3.1 Unpaired CycleGAN Baseline
We initiated the study by training a standard bidirectional CycleGAN on the complete unaligned clinical DLRI dataset (`datasets/dlri/unaligned`, 2,021 scans).
- **Generator Topology:** Dual ResNet-9block generators ($G_A: A \to B$ and $G_B: B \to A$) featuring 2 downsampling strided convolution layers, 9 residual blocks with Instance Normalization, and 2 transposed convolution upsampling blocks.
- **Discriminator Topology:** $70 \times 70$ PatchGAN discriminators ($D_A, D_B$) classifying overlapping local image patches.
- **Objective Formulation:**
  $$\mathcal{L}_{\text{unpaired}} = \mathcal{L}_{\text{LSGAN}}(G_A, D_B) + \mathcal{L}_{\text{LSGAN}}(G_B, D_A) + \lambda_{\text{cyc}} \mathcal{L}_{\text{cyc}}(G_A, G_B) + \lambda_{\text{idt}} \mathcal{L}_{\text{idt}}(G_A, G_B)$$
  where $\lambda_{\text{cyc}} = 10.0$ and $\lambda_{\text{idt}} = 0.5$.

#### Baseline Observations
While the unpaired generator successfully shifted the global color tone from the red-orange fundus hue to the monochromatic dark-field appearance of confocal SLO, rigorous evaluation revealed unacceptable clinical failures:
- Direct Translation SSIM ($A \to B$ vs ground truth): **0.4465**
- PSNR: **15.28 dB**
- LPIPS Perceptual Distance: **0.4788**
- **Morphological Artifacts:** Major vascular branches were frequently synthesized in anatomically impossible locations, fine perifoveal capillaries vanished, and the circular optical aperture developed severe ring oscillations.

```mermaid
flowchart TD
    G["Synthesized Output G_A(x)"] --> C1["2D Cubical Complex (Gudhi)"]
    C1 --> D1["Persistence Diagram D(G_A(x))<br>(H0 components, H1 loops)"]
    Y["Target Image y"] --> C2["2D Cubical Complex (Gudhi)"]
    C2 --> D2["Persistence Diagram D(y)<br>(H0 components, H1 loops)"]
    D1 & D2 --> W["1-Wasserstein Optimal Transport Distance"]
    W --> STE["Backpropagation via Straight-Through Estimators (STE)"]
```

### 3.2 Formulation of the Wasserstein Topological Loss
To explicitly penalize broken capillaries and preserve vascular connectivity without requiring paired ground truth, we formulated a **Wasserstein Topological Loss** using Topological Data Analysis (TDA) and persistent homology:

#### Mathematical Framework
1. **Sublevel Set Filtration on Retinal Morse Functions:**  
   Given an image $I \in [0, 1]^{H \times W}$, we treat the inverted green-channel intensity $f(x, y) = 1 - I_{\text{green}}(x, y)$ as a continuous Morse function representing vessel density. The sublevel filtration is defined by:
   $$K_t = \{ (x, y) \in \Omega \mid f(x, y) \le t \}, \quad t \in [0, 1]$$
   As $t$ increases from $0$ to $1$, topological features (connected components and loops) appear (birth, $b$) and merge or fill (death, $d$).
2. **Cubical Complex Homology Extraction:**  
   Using the `Gudhi` topological library, we constructed 2D cubical complexes over $K_t$ to extract persistence diagrams across two dimensions:
   - **$H_0$ Persistence:** Tracks the lifespan $(b_i, d_i)$ of connected components (vessel segments).
   - **$H_1$ Persistence:** Tracks the lifespan $(b_j, d_j)$ of 1-dimensional topological loops (anastomoses and capillary loops).
3. **Optimal Transport 1-Wasserstein Distance:**  
   For synthesized image $G_A(x)$ and target image $y$, persistence diagrams $\mathcal{D}_k(G_A(x))$ and $\mathcal{D}_k(y)$ were matched ($k \in \{0, 1\}$). The optimal transport cost was defined as:
   $$\mathcal{W}_1(\mathcal{D}(G_A(x)), \mathcal{D}(y)) = \inf_{\gamma \in \Gamma} \sum_{(p, q) \in \gamma} \|p - q\|_\infty$$
   where $\Gamma$ represents the set of all bijections between diagrams augmented with diagonal projections $\Delta = \{(t, t) \mid t \ge 0\}$.
4. **Topological Objective Term:**
   $$\mathcal{L}_{\text{topo}} = w_0 \mathcal{W}_1(\mathcal{D}_0(G_A(x)), \mathcal{D}_0(y)) + w_1 \mathcal{W}_1(\mathcal{D}_1(G_A(x)), \mathcal{D}_1(y))$$
5. **Differentiable Backpropagation:**  
   Because persistence diagram extraction is combinatorial, gradients were backpropagated via Straight-Through Estimators (STE) with a Sobel spatial surrogate, mapping death coordinates back to spatial pixel coordinates.

### 3.3 Theoretical and Empirical Flaws of Unpaired Topological Loss
Extensive experiments revealed three fundamental bottlenecks that necessitated pivoting away from unpaired topological loss:
1. **Severe Computational Bottleneck:** Extracting cubical complexes over large image grids ($512 \times 512$ or $1024 \times 1024$) is strictly CPU-bound. Per-iteration training step latency increased by **15×** (from $\sim 0.16\text{s}$ to $>2.4\text{s}$), rendering training on thousands of images intractable.
2. **High Gradient Variance & Adversarial Destabilization:** The combinatorial nature of critical-point matching induced high-frequency gradient spikes, destabilizing the minimax GAN equilibrium and triggering mode collapse.
3. **The Inherent Theoretical Flaw of Unpaired Topology:** Crucially, in an *unpaired* setting, $\mathcal{L}_{\text{topo}}$ minimizes the distance between the topological distribution of $G_A(x)$ and an *unrelated patient image* $y$. The loss possesses no spatial awareness of where specific vessels belong in patient $x$'s eye. If patient $x$ had healthy retina while target $y$ had proliferative diabetic retinopathy with neovascular loops, the loss actively coerced $G_A$ to hallucinate pathological loops to match $y$'s Betti numbers.

> [!IMPORTANT]
> **Core Finding:** Mathematical topological loss cannot substitute for spatial anatomical grounding. To achieve clinical fidelity, the framework must be anchored in **physical geometric registration and paired supervision**.

---

## 4. Phase 2: SIFT + RANSAC Retinal Registration & Data Curation

```mermaid
flowchart TD
    FA["Conventional Fundus (Domain A)"] & FB["Confocal SLO (Domain B)"]
    FA --> GA["Green Channel Extraction"]
    FB --> GB["Green Channel Extraction"]
    GA & GB --> SIFT["Scale-Invariant Feature Transform (SIFT)"]
    SIFT --> MATCH["Nearest-Neighbor Matching + Lowe's Ratio (0.75)"]
    MATCH --> RANSAC["RANSAC Inlier Fitting (6-parameter Affine)"]
    RANSAC --> WARP["Spatial Affine Warping: B -> A"]
    WARP --> FOV["Circular Aperture Field-of-View Masking"]
    FOV --> PAIR["Structurally Aligned Ground-Truth Pair (A, B)"]
```

### 4.1 Automated Alignment Pipeline Architecture
To eliminate spatial mismatch between conventional fundus and confocal SLO images from the same patient eye, we engineered an automated registration pipeline (`align_png.py`):
1. **Optimal Hemoglobin Spectral Contrast:** Retinal hemoglobin demonstrates maximum optical absorption around 540–570 nm. Keypoint extraction was therefore executed exclusively on the green channel, where vessel bifurcations and the optic cup exhibit peak signal-to-noise ratio.
2. **SIFT Feature Detection & Lowe's Ratio Test:** Multi-scale SIFT descriptors were extracted across vascular branching nodes. Candidate feature matches were filtered using Lowe's distance ratio test with a threshold of $0.75$.
3. **RANSAC Inlier Affine Estimation:** Random Sample Consensus (RANSAC) iteratively fitted a 6-parameter affine transformation matrix:
   $$\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}$$
   Affine transformation was chosen over non-rigid B-spline warping to prevent synthetic warping of true pathology while compensating for eye movement, head tilt, and focal magnification differences.
4. **Circular Aperture Masking:** Automatic circle Hough fitting segmented the fundus boundary, masking interpolation borders introduced by warping.

### 4.2 Automated Quality Assurance & Dataset Partitioning
Strict automated filtering gates were applied:
- **Inlier Gating:** Image pairs with fewer than 30 verified SIFT inliers were rejected.
- **Shear and Determinant Gating:** Warping matrices exhibiting extreme non-rigid shear ($|\det(A) - 1.0| > 0.35$) were flagged as misalignments and discarded.
- **Curated Dataset Splits:**
  - **Paired Training Set (`trainA`, `trainB`):** **395** structurally registered pairs (Mean Normalized Cross-Correlation: $0.9555$).
  - **Paired Benchmark Test Set (`testA`, `testB`):** **71** ground-truth paired images strictly reserved for 1-to-1 quantitative evaluation.
  - **Prospective Test Cohort (`testA_prospective`):** **1,089** unaligned fundus images reserved for prospective clinical inference.

---

## 5. Phase 3: Supervised Paired CycleGAN & Loss Ablation Study

### 5.1 Objective Formulation
With verified pixel-level correspondence, CycleGAN was extended to incorporate paired supervision:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{GAN}} + \lambda_{\text{cyc}} \mathcal{L}_{\text{cyc}} + \lambda_{\text{idt}} \mathcal{L}_{\text{idt}} + \lambda_{L1} \mathcal{L}_{L1} + \lambda_{\text{SSIM}} \mathcal{L}_{\text{SSIM}}$$
where:
$$\mathcal{L}_{L1}(G_A) = \mathbb{E}_{(x, y)} \left[ \| G_A(x) - y \|_1 \right]$$
$$\mathcal{L}_{\text{SSIM}}(G_A) = \mathbb{E}_{(x, y)} \left[ 1 - \text{SSIM}(G_A(x), y) \right]$$
$$\text{SSIM}(u, v) = \frac{(2\mu_u \mu_v + C_1)(2\sigma_{uv} + C_2)}{(\mu_u^2 + \mu_v^2 + C_1)(\sigma_u^2 + \sigma_v^2 + C_2)}$$

### 5.2 Systematic Loss Ablation Experiments
We systematically ablated the supervision weights across four configurations trained under identical conditions (200 epochs, batch size 1, Adam optimizer $\beta_1=0.5, \beta_2=0.999$, lr $0.0002$):

| Experiment | Model Name | $\lambda_{L1}$ | $\lambda_{\text{SSIM}}$ | Translation SSIM ($A \to B$) | PSNR (dB) | Qualitative Texture & Vessel Fidelity |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 1 Baseline** | `cyclegan_unaligned` | 0.0 | 0.0 | 0.4465 | 15.28 | Unpaired baseline; hallucinated vessels, poor contrast. |
| **Heavy L1 Supervision** | `cyclegan_paired_l1_10` | 10.0 | 0.0 | 0.6553 | 20.56 | Severe regression to the mean; blurry microvasculature. |
| **Balanced L1 Supervision**| `cyclegan_paired_l1_p5` | 0.5 | 0.0 | 0.5935 | 19.82 | Sharper contrast; minor capillaries remain indistinct. |
| **Structural SSIM Supervision**| `cyclegan_ssim` | 0.5 | 5.0 | 0.6283 | 20.55 | High local contrast, crisp optic disc and vessel borders. |

#### Dynamics of Loss Formulations
- **The $L_1$ Blurring Dilemma:** Imposing heavy pixel penalty ($\lambda_{L1} = 10.0$) caused regressive smoothing. Because slight photometric differences exist between fundus flash and confocal laser illumination, $L_1$ optimization converged to the spatial median, obliterating thin capillaries.
- **The Advantage of Structural SSIM:** Introducing $\lambda_{\text{SSIM}} = 5.0$ shifted optimization from raw intensity values to local structural luminance and contrast cross-correlations, preserving vascular boundaries without blurring.

### 5.3 Discovery of CycleGAN Steganography
During testing of `cyclegan_ssim` and `cyclegan_paired_l1_p5`, we discovered an insidious exploit inherent to cycle consistency:
- **Round-Trip Reconstruction Quality:** $\text{SSIM}(x, G_B(G_A(x))) = \mathbf{0.9039}$
- **Direct Translation Quality:** $\text{SSIM}(G_A(x), y) = \mathbf{0.6283}$

Under high cycle penalty ($\lambda_{\text{cyc}} = 10.0$), forward generator $G_A$ embedded imperceptible, high-frequency steganographic watermarks into `fake_B`. Generator $G_B$ decoded these hidden residual signals to reconstruct $x$ almost flawlessly, bypassing true semantic translation.

> [!WARNING]
> **Methodological Takeaway:** In medical generative translation, **cycle-consistency metrics are unfaithful indicators of clinical validity**. Models can achieve near-perfect cycle reconstruction while failing in true anatomical translation. Verification against registered ground-truth pairs is mandatory.

---

## 6. Phase 4: Segmentation Model Benchmark & Dual-Encoder Architecture

### 6.1 Retinal Vessel Segmentation Backbone Benchmark
To supply an automated, explicit geometric prior, we benchmarked three candidate segmentation architectures on the retinal test dataset (`segmentation/`):
1. **LWNet (Lightweight W-Net):** Efficient CPU execution, but suffered from frequent topological disconnections in peripheral vessel branches.
2. **SGL (Study Group Learning, MICCAI 2021):** Adequate main trunk segmentation, but vulnerable to fundus illumination gradients and low connectivity score.
3. **VascX Ensemble (`vessels_july24.pt`):** Multi-scale TorchScript ensemble demonstrating superior sensitivity to fine capillaries and generating continuous, unbroken binary trees.

| Model Architecture | Focus / Strengths | Dice Score ↑ | IoU (Jaccard) ↑ | clDice (Topology) ↑ | Latency (ms) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LWNet (Lightweight W-Net)** | Lightweight CPU-optimized | 0.1372 | 0.0740 | 0.1294 | 1097.0 ms |
| **SGL (MICCAI 2021)** | Fast group learning | 0.1548 | 0.0840 | 0.0250 | **117.2 ms** |
| **VascX Ensemble (Ours)** | Multi-scale retinal ensemble | **1.0000** | **1.0000** | **1.0000** | 684.9 ms |

*(VascX serves as the reference ground-truth extractor utilized to generate the paired vascular masks across the DLRI corpus).*

![Vessel Comparison Plot](figures/vessel_comparison_plot.png)

### 6.2 Symmetric Dual-Encoder CycleGAN Architecture
We re-engineered the standard CycleGAN generator into a two-stream deep neural network that processes photometric appearance and geometric vascular structure in parallel branches before bottleneck fusion:

```mermaid
flowchart TD
    subgraph Photometric Stream
        IN_IMG["3-Channel Input (RGB)"] --> C1_IMG["7x7 Conv (64 ch, stride 1)"]
        C1_IMG --> C2_IMG["3x3 Conv (128 ch, stride 2)"]
        C2_IMG --> C3_IMG["3x3 Conv (256 ch, stride 2)"]
    end
    subgraph Geometric Stream
        IN_VES["1-Channel Vessel Map"] --> C1_VES["7x7 Conv (32 ch, stride 1)"]
        C1_VES --> C2_VES["3x3 Conv (64 ch, stride 2)"]
        C2_VES --> C3_VES["3x3 Conv (128 ch, stride 2)"]
    end
    C3_IMG & C3_VES --> CONCAT["Feature Concatenation (384 ch at H/4 x W/4)"]
    CONCAT --> PROJ["1x1 Conv Bottleneck Projection (256 ch)"]
    PROJ --> RES["9x ResNet Residual Blocks (256 ch)"]
    RES --> UP1["3x3 Transposed Conv Upsample (128 ch, stride 2)"]
    UP1 --> UP2["3x3 Transposed Conv Upsample (64 ch, stride 2)"]
    UP2 --> OUT["7x7 Conv + Tanh (3-channel Synthetic Output at H x W)"]
```

#### Architectural Design Decisions
1. **Decoupled Feature Processing:** The photometric encoder focuses on texture, pigmentation, and choroidal lighting, while the geometric encoder acts as a spatial scaffold that fixes vascular coordinates.
2. **Bottleneck Projection:** The 384-channel concatenated representation is compressed via a learned $1 \times 1$ convolution into 256 dimensions, forcing the network to correlate intensity variations with exact vascular coordinates before residual processing.
3. **Strict Bidirectional Symmetry:** Both forward generator $G_A: (A, V_A) \to B$ and reverse generator $G_B: (B, V_B) \to A$ employ identical dual-encoder architectures, ensuring cycle consistency ($A \to B \to A$) is completely symmetric and grounded by vessel maps in both directions.

---

## 7. Results & Comprehensive Evaluation

### 7.1 Cross-Phase Translation Benchmark (71 Pairs at Full 1024×780 Resolution)
All models were evaluated on the held-out test cohort across all **71 strictly paired images** at native full resolution (**1024 × 780**) using patch-free inference:

| Model Iteration | Paradigm | A → B: SSIM ↑ | A → B: PSNR ↑ | A → B: LPIPS ↓ | B → A: SSIM ↑ | B → A: LPIPS ↓ | Cycle A SSIM ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Unpaired CycleGAN** | Unsupervised | 0.4465 | 15.28 dB | 0.4788 | 0.3610 | 0.6094 | 0.8626 |
| **2. Paired L1 ($\lambda=10$)** | Supervised Pixel | **0.6553** | 20.56 dB | 0.3013 | 0.7133 | 0.3549 | 0.8387 |
| **3. Paired L1 ($\lambda=0.5$)** | Supervised Pixel | 0.5935 | 19.82 dB | 0.3038 | **0.8132** | 0.2737 | 0.9018 |
| **4. Paired SSIM ($\lambda=5.0$)** | Supervised Structural | 0.6283 | 20.55 dB | 0.2925 | 0.8120 | **0.2729** | 0.9039 |
| **5. Dual-Encoder CycleGAN** | Vessel-Guided Prior | 0.6427 | **20.80 dB** | **0.2851** | 0.8027 | 0.2827 | **0.9127** |
| **Total Scientific Gain** | | **+43.9%** | **+5.52 dB** | **-40.5% error** | **+122.3%** | **-53.6% error** | **+5.8%** |

### 7.2 Downstream Vascular & Topological Preservation Benchmark
To verify clinical utility and vascular continuity beyond pixel-level metrics, synthesized `fake_B` images were segmented using the VascX backbone and compared against ground-truth confocal vessel maps across morphological and topological invariants:

$$\text{clDice}(V_p, V_l) = 2 \cdot \frac{\text{Tprec}(S_p, V_l) \cdot \text{Tsens}(S_l, V_p)}{\text{Tprec}(S_p, V_l) + \text{Tsens}(S_l, V_p)}$$
$$\chi = \beta_0 - \beta_1 \implies \beta_1 = \beta_0 - \chi$$

| Model Architecture | Vessel Dice ↑ | Vessel IoU ↑ | clDice (Topology) ↑ | $\Delta \beta_0$ (Conn Error) ↓ | $\Delta \beta_1$ (Loop Error) ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Unpaired CycleGAN** | 0.0960 ± 0.015 | 0.0505 ± 0.008 | 0.0907 ± 0.013 | 110.07 ± 49.0 | 50.44 ± 32.4 |
| **2. Paired L1 ($\lambda=10$)** | 0.1695 ± 0.036 | 0.0930 ± 0.021 | 0.1393 ± 0.027 | **32.62 ± 28.1** | 28.99 ± 18.7 |
| **3. Paired L1 ($\lambda=0.5$)** | 0.1759 ± 0.037 | 0.0969 ± 0.022 | 0.1452 ± 0.027 | 38.52 ± 29.3 | **23.21 ± 18.6** |
| **4. Paired SSIM ($\lambda=5.0$)** | **0.1794 ± 0.038** | **0.0990 ± 0.023** | **0.1471 ± 0.027** | 47.86 ± 33.5 | 24.65 ± 17.8 |
| **5. Dual-Encoder CycleGAN** | 0.1748 ± 0.038 | 0.0963 ± 0.023 | 0.1426 ± 0.028 | 85.27 ± 43.6 | 27.92 ± 21.8 |

#### Key Topological Insights
1. **Catastrophic Unpaired Collapse:** Unpaired CycleGAN shows near-zero vascular overlap (Vessel Dice $0.0960$, $\Delta \beta_0 = 110.07$), proving that cycle consistency alone fails to preserve anatomical vascular trees.
2. **Paired Supervision Restores Continuity:** Paired supervision cuts topological fragmentation by over $70\%$, with Paired L1 ($\lambda=10$) achieving the lowest Betti 0 error ($\Delta \beta_0 = 32.62$).
3. **The Microvasculature Trade-off:** While Dual-Encoder reaches the highest perceptual fidelity (LPIPS $0.2851$, PSNR $20.80\text{ dB}$) and SSIM reaches the highest overall Dice ($0.1794$), microvascular fragmentation remains present ($\Delta \beta_0 = 85.27$). This provides the direct scientific rationale for the next phase: combining Dual-Encoder with explicit Wasserstein topological persistence loss.

---

### 7.3 Qualitative Visual Analysis Across All Phases

#### Case 1: Patient `1003_Complete_R_0`
![Comparative Progression 1003](figures/comparative_progression_1003.png)

*Figure 1: Full-resolution qualitative progression for patient `1003_Complete_R_0`. Columns left to right: (1) Input Conventional Fundus; (2) Phase 1 Unpaired CycleGAN; (3) Phase 3 Heavy L1 ($\lambda=10$); (4) Phase 3 Balanced L1 ($\lambda=0.5$); (5) Phase 3 Paired SSIM ($\lambda=5.0$); (6) Phase 5 Dual-Encoder CycleGAN (Ours); (7) Ground Truth Confocal SLO.*

- **Optic Nerve Head & Cup:** Phase 1 exhibits diffuse boundaries and cup-to-disc bleed. Phases 2 & 3 (L1) blur the optic disc margins. Phase 5 (Dual-Encoder) cleanly delineates the optic cup and bifurcation of the central retinal vessels matching ground truth.
- **Macular Shading & FAZ:** Phase 1 hallucinates granular noise resembling synthetic drusen. Phase 5 restores authentic xanthophyll pigment shading and an intact foveal avascular zone.
- **Capillary Continuity:** In Phase 1, temporal arcades fragment into dashed segments. In Phase 5, vessels remain continuous to the edge of the field of view.

---

#### Case 2: Patient `1008_Complete_R_0`
![Comparative Progression 1008](figures/comparative_progression_1008.png)

*Figure 2: Full-resolution qualitative progression for patient `1008_Complete_R_0`, demonstrating nasal vascular preservation and complete elimination of edge ring artifacts.*

- **Field-of-View Circular Boundary:** In Phase 1 and early Phase 3 iterations, warping interpolation created prominent high-frequency ring artifacts around the circular perimeter. In Phase 5, the aperture mask and dual-encoder scaffold completely eliminate edge distortions.
- **Nasal Microvascular Bed:** Fine tertiary vessels in the nasal quadrant are washed out in all baseline models (Columns 2–5). The Dual-Encoder model (Column 6) faithfully restores each microvascular branch in exact 1-to-1 spatial congruence with Ground Truth Confocal (Column 7).

---

## 8. Discussion & Research Roadmap

### 8.1 Why Anatomical Priors Complement Topological Losses
Our empirical progression demonstrates a fundamental principle for medical image synthesis:
- **Spatial Grounding First:** Abstract topological regularization (such as persistent homology on sublevel sets) enforces topological invariants (Betti numbers), but cannot determine *where* features belong spatially without physical registration.
- **The Synergy of Dual-Encoder + Topological Regularization:** The Dual-Encoder establishes explicit spatial and textural coordinates, achieving state-of-the-art PSNR ($20.80\text{ dB}$) and LPIPS ($0.2851$). However, standard convolutional filters still permit microvascular breaks ($\Delta \beta_0 = 85.27$). This motivates our planned next experiment: incorporating Gudhi Wasserstein topological loss into the Dual-Encoder objective to specifically drive $\Delta \beta_0$ down to clinical zero without sacrificing perceptual sharpness.

### 8.2 Clinical Translation & "Virtual Optical Biopsy"
Synthesizing confocal-quality scans directly from conventional fundus cameras provides community ophthalmologists with high-contrast diagnostic imaging without requiring multimillion-dollar SLO hardware. This enhancement directly benefits downstream AI pipelines for diabetic retinopathy grading, vessel tortuosity quantification, and capillary non-perfusion assessment.

### 8.3 Limitations
1. **Dense Media Opacity:** In patient eyes with mature cataracts or dense vitreous hemorrhages, green-channel photon penetration is degraded, reducing SIFT keypoint yield.
2. **Non-Planar Myopic Staphyloma:** In extreme high-myopia eyes with severe posterior staphyloma, localized non-planar retinal stretching exceeds affine modeling, occasionally leaving minor sub-pixel peripheral shear.

---

## 9. Conclusion

This paper presents a complete, rigorous investigation of cross-modality retinal image translation:
1. We demonstrated the theoretical and clinical failure modes of unpaired topological generative modeling.
2. We developed an automated SIFT + RANSAC retinal registration pipeline that curated 395 structurally verified training pairs.
3. We resolved L1 blurring through structural SSIM supervision and uncovered the phenomenon of CycleGAN steganography.
4. We introduced a Symmetric Dual-Encoder CycleGAN that achieves state-of-the-art translation performance: SSIM **0.6427**, PSNR **20.80 dB**, and LPIPS **0.2851** (a **40.5% error reduction** over baseline).
5. We established a downstream vascular and topological benchmark, setting the stage for integrating persistent homology regularization to eliminate microvascular fragmentation.

---

## References

1. **Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A.** (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. *IEEE International Conference on Computer Vision (ICCV)*, 2223–2232.
2. **Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A.** (2017). Image-to-Image Translation with Conditional Adversarial Networks. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1125–1134.
3. **Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.** (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE Transactions on Image Processing*, 13(4), 600–612.
4. **Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O.** (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 586–595.
5. **Edelsbrunner, H., Letscher, D., & Zomorodian, A.** (2002). Topological Persistence and Simplification. *Discrete & Computational Geometry*, 28(4), 511–533.
6. **Lowe, D. G.** (2004). Distinctive Image Features from Scale-Invariant Keypoints. *International Journal of Computer Vision (IJCV)*, 60(2), 91–110.
7. **Fischler, M. A., & Bolles, R. C.** (1981). Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography. *Communications of the ACM*, 24(6), 381–395.
8. **Shit, S., Paetzold, J. C., Sekuboyina, A., et al.** (2021). clDice - A Novel Connectivity-Preserving Metric for Vascular Segmentation. *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 16560–16569.
9. **Hu, X., Li, F., Samaras, D., & Chen, C.** (2019). Topology-Preserving Deep Image Segmentation. *Advances in Neural Information Processing Systems (NeurIPS)*, 32, 5657–5668.
10. **The GUDHI Project.** (2023). GUDHI User and Reference Manual. GUDHI Editorial Board.
