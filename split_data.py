import os
import shutil
import random
from glob import glob

random.seed(42)

SRC_IMAGES = "dataset/Snake.v1i.yolov8/train/images"
SRC_LABELS = "dataset/Snake.v1i.yolov8/train/labels"
DST = "dataset"
RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}

image_paths = sorted(glob(os.path.join(SRC_IMAGES, "*")))
random.shuffle(image_paths)

n = len(image_paths)
train_end = int(n * RATIOS["train"])
val_end = train_end + int(n * RATIOS["val"])

splits = {
    "train": image_paths[:train_end],
    "val": image_paths[train_end:val_end],
    "test": image_paths[val_end:],
}

for split_name, paths in splits.items():
    img_dir = os.path.join(DST, split_name, "images")
    lbl_dir = os.path.join(DST, split_name, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_path in paths:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(SRC_LABELS, base + ".txt")

        if os.path.exists(label_path):
            shutil.copy2(img_path, img_dir)
            shutil.copy2(label_path, lbl_dir)
        else:
            print(f"Warning: missing label for {img_path}")

print(f"Done: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")
