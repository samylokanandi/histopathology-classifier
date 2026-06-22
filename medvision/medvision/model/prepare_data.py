"""
prepare_data.py
The raw "Breast Histopathology Images" Kaggle dataset is organized by patient ID,
each with subfolders "0" (non-IDC) and "1" (IDC):

    raw_data/
      10253/
        0/  *.png
        1/  *.png
      10254/
        0/
        1/
      ...

This script collapses that into a flat train/val split that torchvision's
ImageFolder can read directly:

    data/
      train/class_0/*.png   train/class_1/*.png
      val/class_0/*.png     val/class_1/*.png

Usage:
    python prepare_data.py --raw_dir ./raw_data --out_dir ./data --val_split 0.15

How to get the raw data:
    1. Create a free Kaggle account, go to:
       https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
    2. Download + unzip it (or use `kagglehub` in a Colab notebook, which handles
       auth automatically: `import kagglehub; path = kagglehub.dataset_download(
       "paultimothymooney/breast-histopathology-images")`)
    3. Point --raw_dir at the unzipped folder.

Note: this dataset is ~3GB and has ~277k small (50x50) patches. For a first pass,
consider capping the number of patches per class with --max_per_class to keep
training fast while you're validating the pipeline end-to-end.
"""

import argparse
import random
import shutil
from pathlib import Path


def collect_patches(raw_dir, max_per_class=None):
    raw_dir = Path(raw_dir)
    patches = {"0": [], "1": []}

    for patient_dir in raw_dir.iterdir():
        if not patient_dir.is_dir():
            continue
        for label in ("0", "1"):
            label_dir = patient_dir / label
            if label_dir.exists():
                patches[label].extend(label_dir.glob("*.png"))

    if max_per_class:
        for label in patches:
            random.shuffle(patches[label])
            patches[label] = patches[label][:max_per_class]

    return patches


def split_and_copy(patches, out_dir, val_split, seed=42):
    random.seed(seed)
    out_dir = Path(out_dir)

    for label, files in patches.items():
        class_name = f"class_{label}"
        random.shuffle(files)
        n_val = int(len(files) * val_split)
        val_files, train_files = files[:n_val], files[n_val:]

        for split_name, split_files in [("train", train_files), ("val", val_files)]:
            dest_dir = out_dir / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy(f, dest_dir / f.name)

        print(f"class_{label}: {len(train_files)} train, {len(val_files)} val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./data")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--max_per_class", type=int, default=None,
                         help="Cap patches per class for a faster first run, e.g. 5000")
    args = parser.parse_args()

    patches = collect_patches(args.raw_dir, args.max_per_class)
    print(f"Found {len(patches['0'])} class_0 and {len(patches['1'])} class_1 patches")
    split_and_copy(patches, args.out_dir, args.val_split)
