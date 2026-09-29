import os
import sys
import time
import argparse
from pathlib import Path
import torch
from PIL import Image
from rtnls_inference import SegmentationEnsemble

def parse_args():
    parser = argparse.ArgumentParser(description="Extract VascX vessel maps for retinal datasets")
    parser.add_argument("--dataroot", type=str, default="./datasets/dlri/aligned", help="Path to dataset root")
    parser.add_argument("--model_path", type=str, default="./segmentation/models/vessels/vessels_july24.pt", help="Path to VascX torchscript weights")
    parser.add_argument("--splits", type=str, default="trainA,trainB,testB,testA", help="Comma-separated splits to process")
    parser.add_argument("--batch_size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    return parser.parse_args()

def main():
    args = parse_args()
    dataroot = Path(args.dataroot)
    model_path = Path(args.model_path)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print("=" * 70)
    print("VascX Vessel Map Extraction Pipeline")
    print(f"Data root:   {dataroot.resolve()}")
    print(f"Model path:  {model_path.resolve()}")
    print(f"Splits:      {splits}")
    print("=" * 70)

    if not model_path.exists():
        raise FileNotFoundError(f"VascX model weights not found at: {model_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    print(f"\nLoading VascX ensemble from {model_path}...")
    t0 = time.time()
    vessel_ensemble = SegmentationEnsemble.from_torchscript(model_path).to(device)
    print(f"Model loaded in {time.time() - t0:.2f}s")

    for split in splits:
        src_dir = dataroot / split
        dest_dir = dataroot / f"{split}_vessel"

        if not src_dir.exists():
            print(f"\n[SKIP] Split directory {src_dir} does not exist. Skipping.")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)

        all_images = sorted([
            p for p in src_dir.iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ])

        # Filter out images that already have extracted vessel maps
        todo = []
        for img_path in all_images:
            mask_path = dest_dir / f"{img_path.stem}.png"
            if not mask_path.exists() or mask_path.stat().st_size == 0:
                todo.append({"id": img_path.stem, "image": str(img_path)})

        print(f"\n--- Processing split: {split} ---")
        print(f"Total images:     {len(all_images)}")
        print(f"Already extracted: {len(all_images) - len(todo)}")
        print(f"Remaining to run: {len(todo)}")

        if not todo:
            print(f"All vessel maps for {split} already exist!")
            continue

        # Process in batches to avoid RAM ballooning and log periodic progress
        chunk_size = 50
        num_chunks = (len(todo) + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            chunk = todo[i * chunk_size : (i + 1) * chunk_size]
            t_chunk = time.time()
            print(f"[{split}] Running chunk {i+1}/{num_chunks} ({len(chunk)} images)...")
            vessel_ensemble.predict(
                chunk,
                dest_path=dest_dir,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            # Ensure saved masks have dynamic range [0, 255] instead of [0, 1]
            for item in chunk:
                mask_p = dest_dir / f"{item['id']}.png"
                if mask_p.exists():
                    with Image.open(mask_p) as im:
                        if im.getextrema()[1] <= 1:
                            im_scaled = Image.eval(im, lambda p: 255 if p > 0 else 0)
                            im_scaled.save(mask_p)

            elapsed = time.time() - t_chunk
            print(f"[{split}] Chunk {i+1}/{num_chunks} completed in {elapsed:.1f}s ({elapsed/len(chunk):.2f}s/img)")

        # Verify
        completed = len(list(dest_dir.glob("*.png")))
        print(f"Done split {split}! Total masks in {dest_dir}: {completed}")

    print("\n" + "=" * 70)
    print("Vessel Map Extraction Complete for all requested splits!")
    print("=" * 70)

if __name__ == "__main__":
    main()
