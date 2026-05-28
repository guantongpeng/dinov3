#!/usr/bin/env python3
"""Convert DIOR dataset (VOC XML format) to COCO JSON format for MMDetection."""

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

DIOR_CLASSES = [
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship",
    "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
]

def parse_xml(xml_path):
    """Parse a VOC-format XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find("filename").text
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
        objects.append({"name": name, "bbox": [xmin, ymin, w, h]})
    return filename, width, height, objects


def convert_split(split_name, file_list, ann_dir, img_dir, output_path, class_to_id):
    images = []
    annotations = []
    ann_id = 0

    for img_id, img_name in enumerate(file_list, start=1):
        img_stem = img_name.replace(".jpg", "")
        xml_path = os.path.join(ann_dir, f"{img_stem}.xml")

        if not os.path.exists(xml_path):
            continue

        filename, width, height, objects = parse_xml(xml_path)

        images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })

        for obj in objects:
            cls_name = obj["name"]
            if cls_name not in class_to_id:
                continue
            x, y, w, h = obj["bbox"]
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_to_id[cls_name],
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco_dict = {
        "info": {"description": "DIOR Dataset", "version": "1.0", "year": 2019},
        "licenses": [{"id": 0, "name": "Unknown", "url": ""}],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": cname} for cname, cid in class_to_id.items()],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco_dict, f, indent=2)
    print(f"Saved {output_path}: {len(images)} images, {len(annotations)} annotations")


def main():
    base = Path("data/DIOR/DIOR")
    ann_dir = str(base / "Annotations")
    img_dir = str(base / "JPEGImages")
    output_dir = str(Path("data/DIOR/annotations"))

    class_to_id = {name: i + 1 for i, name in enumerate(DIOR_CLASSES)}  # 1-indexed

    splits_dir = base / "ImageSets" / "Main"
    for split in ["train", "val", "test"]:
        with open(splits_dir / f"{split}.txt") as f:
            file_list = [line.strip() + ".jpg" for line in f.read().strip().split()]

        convert_split(
            split, file_list, ann_dir, img_dir,
            os.path.join(output_dir, f"instances_{split}2019.json"),
            class_to_id,
        )


if __name__ == "__main__":
    main()
