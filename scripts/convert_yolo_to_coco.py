#!/usr/bin/env python3
"""Convert COCO128 YOLO-format labels to COCO JSON format for MMDetection training."""

import json
import os
import random
from pathlib import Path

from PIL import Image

# COCO 2017 class names (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

def yolo_to_coco(
    img_dir: str,
    label_dir: str,
    output_json: str,
    train_val_split: float = 0.85,
    seed: int = 42,
):
    """Convert YOLO format labels to COCO JSON format."""
    random.seed(seed)

    img_dir = Path(img_dir)
    label_dir = Path(label_dir)

    image_files = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_val_split)
    train_files = sorted(image_files[:split_idx])
    val_files = sorted(image_files[split_idx:])

    for split, files in [("train", train_files), ("val", val_files)]:
        images = []
        annotations = []
        categories = [
            {"id": i, "name": name} for i, name in enumerate(COCO_CLASSES)
        ]

        ann_id = 0
        for img_id, img_path in enumerate(files, start=1):
            img = Image.open(img_path)
            w, h = img.size

            images.append({
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            })

            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cx = float(parts[1]) * w
                        cy = float(parts[2]) * h
                        bw = float(parts[3]) * w
                        bh = float(parts[4]) * h
                        x = cx - bw / 2
                        y = cy - bh / 2

                        annotations.append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls_id,
                            "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                            "area": round(bw * bh, 2),
                            "iscrowd": 0,
                        })
                        ann_id += 1

        coco_dict = {
            "images": images,
            "annotations": annotations,
            "categories": categories,
        }

        out_path = Path(output_json).parent / f"{split}.json"
        with open(out_path, "w") as f:
            json.dump(coco_dict, f, indent=2)
        print(f"Saved {out_path}: {len(images)} images, {len(annotations)} annotations")

if __name__ == "__main__":
    base = Path(__file__).parent
    coco128_root = base / "coco128"

    yolo_to_coco(
        img_dir=str(coco128_root / "images" / "train2017"),
        label_dir=str(coco128_root / "labels" / "train2017"),
        output_json=str(base / "annotations" / "instances"),
    )
