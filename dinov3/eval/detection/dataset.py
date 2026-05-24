# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
COCO-format detection dataset and data transforms for training.
"""
import json
import logging
import os
import random
from typing import Any

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

logger = logging.getLogger("dinov3")


class RandomResize:
    def __init__(self, sizes, max_size=None):
        self.sizes = sorted(sizes)
        self.max_size = max_size

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        orig_size = img.size  # (w, h)
        img = T.functional.resize(img, size, max_size=self.max_size)
        if target is not None:
            ratio_w = img.size[0] / orig_size[0]
            ratio_h = img.size[1] / orig_size[1]
            if "boxes" in target:
                target["boxes"][:, 0] *= ratio_w
                target["boxes"][:, 1] *= ratio_h
                target["boxes"][:, 2] *= ratio_w
                target["boxes"][:, 3] *= ratio_h
            if "orig_size" in target:
                target["orig_size"] = torch.tensor([orig_size[1], orig_size[0]])
        return img, target


class RandomCrop:
    def __init__(self, min_size=384, max_size=600):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img, target):
        w, h = img.size
        crop_w = random.randint(self.min_size, min(w, self.max_size))
        crop_h = random.randint(self.min_size, min(h, self.max_size))
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        img = T.functional.crop(img, top, left, crop_h, crop_w)
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            boxes = target["boxes"].clone()
            boxes[:, 0] = boxes[:, 0] - left
            boxes[:, 1] = boxes[:, 1] - top
            boxes[:, 2] = boxes[:, 2] - left
            boxes[:, 3] = boxes[:, 3] - top
            # Filter boxes that are outside the crop
            keep = (boxes[:, 2] > 0) & (boxes[:, 3] > 0) & (boxes[:, 0] < crop_w) & (boxes[:, 1] < crop_h)
            target["boxes"] = boxes[keep]
            if "labels" in target:
                target["labels"] = target["labels"][keep]
        return img, target


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            img = T.functional.hflip(img)
            if target is not None and "boxes" in target and len(target["boxes"]) > 0:
                w = img.size[0]
                boxes = target["boxes"].clone()
                boxes[:, 0] = w - boxes[:, 0]
                boxes[:, 2] = w - boxes[:, 2]
                # Swap x1, x2 since they may have flipped order
                x1 = boxes[:, 0].min(boxes[:, 2])
                x2 = boxes[:, 0].max(boxes[:, 2])
                boxes[:, 0] = x1
                boxes[:, 2] = x2
                target["boxes"] = boxes
        return img, target


class ToTensor:
    def __call__(self, img, target):
        img = T.functional.to_tensor(img)
        return img, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        img = T.functional.normalize(img, mean=self.mean, std=self.std)
        h, w = img.shape[-2:]
        if target is not None and "boxes" in target and len(target["boxes"]) > 0:
            # Convert boxes from absolute xyxy to normalized cxcywh
            boxes = target["boxes"].clone()
            boxes[:, 0] /= w
            boxes[:, 1] /= h
            boxes[:, 2] /= w
            boxes[:, 3] /= h
            target["boxes"] = box_xyxy_to_cxcywh(boxes)
        if target is not None:
            target["size"] = torch.tensor([h, w])
        return img, target


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


def make_detection_train_transforms(
    img_size=640,
    random_size_range=(480, 800),
    random_size_max=1333,
    flip_prob=0.5,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    return Compose(
        [
            RandomResize(random_size_range, max_size=random_size_max),
            RandomHorizontalFlip(p=flip_prob),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


def make_detection_eval_transforms(
    img_size=800,
    random_size_max=1333,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    return Compose(
        [
            RandomResize([img_size], max_size=random_size_max),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ]
    )


class CocoDetection(torch.utils.data.Dataset):
    """COCO-format detection dataset."""

    def __init__(self, img_dir, ann_file, transforms=None, return_masks=False):
        from pycocotools.coco import COCO

        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.transforms = transforms
        self.return_masks = return_masks
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        img_info = self.coco.imgs[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = torchvision.io.read_image(img_path)  # [C, H, W]
        # Convert to PIL for transforms
        img = T.functional.to_pil_image(img)

        h, w = img_info["height"], img_info["width"]

        boxes, labels, masks = [], [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
            if self.return_masks and "segmentation" in ann:
                masks.append(self.coco.annToMask(ann))

        target = {
            "image_id": img_id,
            "orig_size": torch.as_tensor([h, w]),
        }
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        if self.return_masks and len(masks) > 0:
            target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CustomDetectionDataset(torch.utils.data.Dataset):
    """Generic detection dataset from a JSON annotation file.

    Expected JSON format:
    {
        "images": [{"id": 1, "file_name": "img.jpg", "height": 480, "width": 640}],
        "annotations": [{"image_id": 1, "bbox": [x, y, w, h], "category_id": 0}],
        "categories": [{"id": 0, "name": "cat"}]
    }
    """

    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms
        with open(ann_file) as f:
            data = json.load(f)
        self.images = {img["id"]: img for img in data["images"]}
        self.categories = {cat["id"]: cat for cat in data.get("categories", [])}
        # Map category ids to 0-indexed
        cat_ids = sorted(self.categories.keys())
        self.cat_id_to_idx = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self.ids = list(sorted(self.images.keys()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = torchvision.io.read_image(img_path)
        img = T.functional.to_pil_image(img)

        h, w = img_info["height"], img_info["width"]
        anns = self.img_to_anns.get(img_id, [])

        boxes, labels = [], []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.cat_id_to_idx[ann["category_id"]])

        target = {
            "image_id": img_id,
            "orig_size": torch.as_tensor([h, w]),
        }
        if len(boxes) > 0:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def build_dataloader(dataset, batch_size, num_workers, distributed=True, drop_last=True):
    """Build dataloader for detection training/evaluation."""
    if distributed and torch.distributed.is_initialized():
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size, drop_last=drop_last)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader


def collate_fn(batch):
    """Collate function for detection dataloader. Returns (NestedTensor, list[dict])."""
    from .util.misc import nested_tensor_from_tensor_list

    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)
