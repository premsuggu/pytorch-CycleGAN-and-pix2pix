# Project Report: Conventional <-> Confocal Image Translation with CycleGAN

## 1. Project Summary

This project trains and evaluates a CycleGAN model to translate between two microscopy domains:
- Domain A: Conventional fluorescence images
- Domain B: Confocal microscopy images

The work is built on the official PyTorch implementation in this repository and adapted for a very small custom dataset.

Primary goals:
- Learn A -> B translation (conventional to confocal)
- Learn B -> A translation (confocal to conventional)
- Preserve semantic content during cross-domain translation
- Validate outputs with both visual inspection and quantitative metrics

## 2. Problem Context

Paired microscopy data is often hard to collect. CycleGAN is useful because it supports unpaired training via adversarial + cycle-consistency constraints.

Key challenge in this project:
- Very small dataset size (15 training images per domain)

Design response:
- Use extensive patch-based sampling through resize + random crop preprocessing
- Train for longer schedule (200 total epochs)
- Track both reconstruction and translation quality

## 3. Repository Components Used

Core scripts:
- train.py: model training
- test.py: model inference and result generation
- compute_metrics.py: metric computation (SSIM/LPIPS/PSNR depending on script version)

Core modules:
- models/cycle_gan_model.py: CycleGAN training logic
- models/networks.py: generators/discriminators/losses
- data/unaligned_dataset.py: unpaired data loading
- options/train_options.py and options/test_options.py: runtime options

Documentation already present:
- train.md: detailed training and evaluation notes for this project

## 4. Dataset Details

Path:
- datasets/conventional2confocal/

Structure:
- trainA: conventional training images
- trainB: confocal training images
- testA: conventional test images
- testB: confocal test images

Observed stats:
- Training set: 15 images in trainA, 15 images in trainB
- Test set: 1 image in testA, 1 image in testB
- Original image size: 2056 x 2056 RGB

Important note:
- Training is unpaired
- Test set can still be used for paired-style evaluation if testA/testB correspond to the same scene/sample

## 5. Model and Loss Configuration

Architecture:
- Generators: 2 x ResNet-9blocks
  - G_A: A -> B
  - G_B: B -> A
- Discriminators: 2 x PatchGAN (basic)
  - D_A for domain A realism
  - D_B for domain B realism

Losses:
- GAN loss: least-squares GAN (LSGAN)
- Cycle-consistency losses:
  - A -> B -> A with lambda_A = 10.0
  - B -> A -> B with lambda_B = 10.0
- Identity loss:
  - lambda_identity = 0.5

## 6. Training Configuration

Training command profile used in this project:

```bash
python train.py \
  --dataroot ./datasets/conventional2confocal \
  --name conventional2confocal_cyclegan \
  --model cycle_gan \
  --batch_size 1 \
  --load_size 512 \
  --crop_size 256 \
  --preprocess scale_width_and_crop \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --save_epoch_freq 100 \
  --lambda_A 10.0 \
  --lambda_B 10.0 \
  --lambda_identity 0.5 \
  --netG resnet_9blocks \
  --netD basic \
  --pool_size 50
```

Why this setup:
- Batch size 1 is standard for CycleGAN and memory-safe for large image inputs
- 512 -> 256 patch pipeline increases sample diversity from tiny dataset
- 200 total epochs gives model enough time to converge with limited data

## 7. Checkpoints and Artifacts

Checkpoints are saved at:
- checkpoints/conventional2confocal_cyclegan/

Key files:
- 100_net_G_A.pth, 100_net_G_B.pth, 100_net_D_A.pth, 100_net_D_B.pth
- 200_net_G_A.pth, 200_net_G_B.pth, 200_net_D_A.pth, 200_net_D_B.pth
- latest_net_*.pth
- loss_log.txt
- train_opt.txt

Test outputs are saved at:
- results/conventional2confocal_cyclegan/test_100/images/
- results/conventional2confocal_cyclegan/test_200/images/

Typical image outputs per test sample:
- real_A, fake_B, rec_A
- real_B, fake_A, rec_B

## 8. Evaluation Approach

Two complementary evaluation views are used.

1) Cycle consistency (reconstruction behavior)
- Compare real_A vs rec_A
- Compare real_B vs rec_B
- Measures information preservation through round-trip mapping

2) Translation quality (direct mapping quality)
- Compare fake_B vs real_B for A -> B
- Compare fake_A vs real_A for B -> A
- Measures direct generation quality versus available target reference

Metrics used in the project workflow:
- SSIM: structural similarity
- LPIPS: perceptual distance (lower is better)
- PSNR: pixel-wise fidelity (in some script versions)
- SIFID: single-image FID style signal (in some summaries)

## 9. Current Reported Metrics Snapshot

From results/metrics_summary.txt currently in workspace:

Conventional2Confocal - Test 100
- ssim_cycle_A: 0.6746
- lpips_cycle_A: 0.4334
- ssim_cycle_B: 0.6523
- lpips_cycle_B: 0.4557
- sifid_translation_AtoB: 342.2574
- sifid_translation_BtoA: 594.9921

Conventional2Confocal - Test 200
- ssim_cycle_A: 0.6241
- lpips_cycle_A: 0.4299
- ssim_cycle_B: 0.5298
- lpips_cycle_B: 0.4608
- sifid_translation_AtoB: 399.3778
- sifid_translation_BtoA: 604.0095

Interpretation caution:
- Different metric scripts/versions were used during experimentation.
- Keep metric comparisons consistent by using one script and one metric set across both epochs.
- Visual quality can improve even when some pixel-structural metrics decrease.

## 10. Reproducibility Commands

Run full bidirectional testing for epoch 100:

```bash
python test.py \
  --dataroot ./datasets/conventional2confocal \
  --name conventional2confocal_cyclegan \
  --model cycle_gan \
  --epoch 100 \
  --preprocess scale_width \
  --load_size 1024
```

Run full bidirectional testing for epoch 200:

```bash
python test.py \
  --dataroot ./datasets/conventional2confocal \
  --name conventional2confocal_cyclegan \
  --model cycle_gan \
  --epoch 200 \
  --preprocess scale_width \
  --load_size 1024
```

Run one-way inference (if needed):

```bash
python test.py \
  --dataroot ./datasets/conventional2confocal/testB \
  --name conventional2confocal_cyclegan \
  --model test \
  --no_dropout \
  --epoch 200
```

Compute metrics:

```bash
python compute_metrics.py
```

## 11. Practical Findings

- The model converges on this small dataset with extended training.
- Resize + crop preprocessing is essential for memory and effective patch sampling.
- Quantitative scores should be interpreted together with visual inspection.
- For GAN outputs, perceptual metrics and qualitative realism are often more aligned with user judgment than strict pixel-level alignment.

## 12. Risks and Limitations

- Extremely small dataset risks overfitting and unstable metric trends.
- Single test pair is not enough for statistically robust conclusions.
- Differences in metric implementations can lead to conflicting interpretations.

## 13. Recommended Next Steps

1. Expand test set to include multiple paired samples.
2. Standardize one evaluation script and freeze metric definitions.
3. Report per-image and averaged metrics for each epoch.
4. Add side-by-side visual paneling for real/fake/reconstructed across epochs.
5. Optionally compare against pix2pix if paired training data is available.

## 14. File Index (Project-Specific)

- train.md: full experiment diary and explanations
- PROJECT_REPORT.md: this consolidated project report
- compute_metrics.py: current metrics script
- results/metrics_summary.txt: latest metric summary dump
- checkpoints/conventional2confocal_cyclegan/: trained weights and logs
- results/conventional2confocal_cyclegan/: generated inference outputs

---

This report is intended as a complete, shareable summary of the current project state in this workspace.