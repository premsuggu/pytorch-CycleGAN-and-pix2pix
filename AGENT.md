# Project Rules — pytorch-CycleGAN-and-pix2pix (fundus retinal imaging)

## What This Project Is

This is a deep-learning research project for **retinal fundus image translation**
between two imaging modalities using CycleGAN:
- **Domain A** — Conventional fundus photographs (Topcon camera, 2576×1958 px)
- **Domain B** — Confocal scanning laser ophthalmoscopy (iCare Eidon, 3680×3288 px)

The clinical goal is to synthesise high-quality confocal images from conventional
fundus photos (and vice versa) to reduce the need for expensive equipment.

---

## Cluster Environment

| Item | Value |
|------|-------|
| Cluster type | HPC (SLURM) |
| Login node | current shell (no GPU here) |
| GPU partition | `gpu` — 30 nodes (`rsgpu001`–`rsgpu030`), each with **2 GPUs**, 192 GB RAM, 48 CPUs |
| GPU per node | 2× (type not labelled in sinfo but CUDA 12.8 compatible) |
| Walltime limit | 6 days on `gpu`, 4 days on `standard`/`cpu` |
| Module system | Lmod |

### Activating the environment

**Always** load the environment like this:
```bash
module load miniconda
conda activate retina-cnn
```

Never use `conda run` for interactive work. In SLURM scripts use `source activate retina-cnn`.

### retina-cnn conda environment — key packages

| Package | Version |
|---------|---------|
| Python | 3.14 |
| torch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| opencv-python | 4.13.0 |
| numpy | 2.4.3 |
| Pillow | 11.3.0 |
| matplotlib | 3.10.8 |
| scikit-image | 0.26.0 |
| lpips | latest |
| dominate | 2.9.1 |
| tqdm | 4.67.3 |
| wandb | 0.25.1 |

> **Python 3.14 pickling issue**: `torchvision.transforms.Lambda` cannot be
> pickled by the forkserver. Always set `num_threads=0` when running DataLoader
> on the login node. Use `num_threads=4` inside SLURM GPU jobs.

---

## Dataset Layout

```
datasets/
├── fundus_cyclegan/        ← RAW dataset (original, do not modify)
│   ├── trainA/             237 PNGs — conventional (2576×1958)
│   ├── trainB/             241 PNGs — confocal (3680×3288)
│   ├── testA/              68 PNGs
│   └── testB/              63 PNGs
│
└── fundus_aligned/         ← ALIGNED dataset (use this for training)
    ├── trainA/             213 matched pairs — conventional (unchanged)
    ├── trainB/             213 matched pairs — confocal (SIFT-aligned to conv space)
    ├── testA/              51 matched pairs
    ├── testB/              51 matched pairs
    ├── flagged.txt         ← unmatched / failed pairs logged here
    └── alignment_preview.png
```

### Pairing convention
Images are paired **by identical filename** across A and B directories.
e.g. `trainA/1005_Complete_0.png` ↔ `trainB/1005_Complete_0.png`

Filenames follow the pattern: `{patientID}_{class}_{index}.png`
- Classes: `Complete`, `Anomalous`
- 220 train pairs + 60 test pairs aligned successfully

---

## Key Scripts

| Script | Purpose | Run with |
|--------|---------|----------|
| `python-scripts/align_png.py` | Align confocal → conventional space (SIFT+RANSAC) | `python python-scripts/align_png.py --workers 32` |
| `python-scripts/train_cyclegan.py` | CycleGAN training (all config inside) | `python python-scripts/train_cyclegan.py` or via SLURM |
| `python-scripts/submit_train.sh` | SLURM job script for GPU training | `sbatch python-scripts/submit_train.sh` |
| `python-scripts/test_alignment.py` | Quick 2-image alignment smoke test | `python python-scripts/test_alignment.py` |

---

## Alignment Pipeline (`align_png.py`)

- **Input**: `datasets/fundus_cyclegan/{trainA,trainB,testA,testB}/`
- **Output**: `datasets/fundus_aligned/{trainA,trainB,testA,testB}/`
- **Algorithm**: SIFT keypoints → BFMatcher + Lowe's ratio test (0.75) → RANSAC `estimateAffinePartial2D` → `warpAffine`
- **Critical fix**: Both images are resized to **1024×1024** before SIFT matching to normalise the resolution difference between modalities. The affine matrix is then scaled back to full resolution for warping.
- **Optimal workers on this cluster**: `--workers 32` (48 CPUs free, ~70 MB RAM/worker, 140 GB available)
- **Circular mask**: applied after warping to remove black border artefacts (`min(h,w)//2 - 50` radius)
- Failed/unmatched pairs → `flagged.txt` (never crash the pipeline)

---

## CycleGAN Training Config (from `dlri.ipynb`)

All parameters live in `TrainConfig` dataclass inside `train_cyclegan.py`.
Key values **must not be changed without good reason**:

| Parameter | Value | Reason |
|-----------|-------|--------|
| `model` | `cycle_gan` | unpaired GAN |
| `dataset_mode` | `unaligned` | CycleGAN data loader |
| `netG` | `resnet_9blocks` | 9-block ResNet generator (11.4M params) |
| `netD` | `basic` | 70×70 PatchGAN discriminator (2.8M params) |
| `norm` | `instance` | instance norm standard for CycleGAN |
| `batch_size` | `4` | set in dlri.ipynb |
| `load_size` | `512` | scale images to this before crop |
| `crop_size` | `256` | final training resolution |
| `preprocess` | `scale_width_and_crop` | |
| `n_epochs` | `100` | constant LR phase |
| `n_epochs_decay` | `100` | linear LR decay → total **200 epochs** |
| `lr` | `2e-4` | Adam LR |
| `beta1` | `0.5` | Adam β₁ |
| `gan_mode` | `lsgan` | least-squares GAN loss |
| `pool_size` | `50` | replay buffer size |
| `lambda_A/B` | `10.0` | cycle consistency weight |
| `lambda_identity` | `0.5` | identity loss weight |
| `save_epoch_freq` | `50` | save checkpoint every 50 epochs |

### Training outputs
```
checkpoints/fundus_aligned_cyclegan/    ← model weights (.pth files)
results/cyclegan/fundus_aligned_cyclegan/
    loss_history.json                   ← per-epoch loss values
    loss_curves.png                     ← plotted loss curves
    train_samples/                      ← 2×2 visual grids per epoch
        epoch_0050_sample_1.png
        ...
```

### Losses to monitor
8 losses are tracked per step:
- `D_A`, `D_B` — discriminator losses (should stay ~0.5–1.5 when training well)
- `G_A`, `G_B` — generator adversarial losses
- `cycle_A`, `cycle_B` — cycle-consistency losses (most important; should decrease)
- `idt_A`, `idt_B` — identity losses

---

## SLURM Job Management

```bash
sbatch python-scripts/submit_train.sh       # submit
squeue --me                                 # check status
tail -f logs/train_<JOBID>.out              # live log
scancel <JOBID>                             # cancel

# Resume from checkpoint (edit python-scripts/submit_train.sh first):
# add: --continue_train --epoch_count <last_saved_epoch + 1>
sbatch python-scripts/submit_train.sh
```

### Smoke test (login node, CPU, 1 epoch, 5 images)
```bash
module load miniconda && conda activate retina-cnn
python python-scripts/train_cyclegan.py --smoke_test
```

---

## Project Files Reference

| File | Description |
|------|-------------|
| `dlri.ipynb` | Original Jupyter notebook with full training pipeline |
| `retinal_alignment.ipynb` | Original alignment notebook (DICOM-based, superseded) |
| `align_png.py` | PNG alignment script (adapted from above) |
| `train_cyclegan.py` | Standalone training script (adapted from dlri.ipynb) |
| `submit_train.sh` | SLURM GPU job script |
| `models/cycle_gan_model.py` | CycleGAN model definition |
| `models/networks.py` | Generator + Discriminator architectures |
| `data/unaligned_dataset.py` | Dataset loader |
| `compute_metrics.py` | SSIM / LPIPS evaluation |
| `docs/topodlri/` | Project documentation |

---

## Coding Rules

1. **Never modify `datasets/fundus_cyclegan/`** — it's the raw source. Always work with `fundus_aligned/`.
2. **Always use `retina-cnn`** conda env. Do not install packages into `base`.
3. **Login node = CPU only**. All GPU training must go through `sbatch submit_train.sh`.
4. **Set `num_threads=0`** when running DataLoader on the login node (Python 3.14 pickling bug).
5. **Checkpoints saved every 50 epochs** + `latest` always. Use `--continue_train` to resume.
6. **Do not use visdom** — `no_html=True` is set. Visualisations are saved as PNG files.
7. When writing new training experiments, copy `submit_train.sh` and change `--name` to avoid overwriting checkpoints.
