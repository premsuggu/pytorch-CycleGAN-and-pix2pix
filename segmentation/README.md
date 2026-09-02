# Vessel Segmentation Models

This directory contains evaluation notebooks and source code for multiple State-of-the-Art retinal vessel segmentation models, used to evaluate our generated CycleGAN domain adaptations (Confocal <-> Conventional).

## Structure
- `eval_utils.py`: Shared utilities for data loading and plotting comparison grids.
- `models/`: Directory containing pre-trained weights for the models (ignored in Git).
- `results/`: Directory containing output plots (ignored in Git).

## Models
1. **VascX** (`vascx_seg.ipynb`)
   - The baseline segmentation model.

2. **LWNet (Lightweight W-Net)** (`lwnet_seg.ipynb` / `lwnet_src/`)
   - A highly efficient segmentation architecture. Tested on CPU.
   - Ref: [agaldran/lwnet](https://github.com/agaldran/lwnet)

3. **SGL (Study Group Learning)** (`sgl_seg.ipynb` / `sgl_src/`)
   - MICCAI 2021 model with robust pre-trained weights (`drive_k8.pth`).
   - SGL operates directly on `0-255` raw pixel inputs (no 0-1 normalization).
   - Ref: [SHI-Labs/SGL-Retinal-Vessel-Segmentation](https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation)

## How to use
Each model has its own dedicated Jupyter Notebook that:
1. Loads 3 random images from both Domain A (Confocal) and Domain B (Conventional).
2. Processes them through the pre-trained models.
3. Generates a visual comparison grid saved in `results/`.
