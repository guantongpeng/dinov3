# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Detection training script for DINOv3.

Usage:
    python -m dinov3.eval.detection.train \
        --output_dir /path/to/output \
        model=dinov3_vit7b16 \
        datasets.root=/path/to/data \
        datasets.train_img_dir=/path/to/train2017 \
        datasets.train_ann_file=/path/to/annotations/instances_train2017.json \
        epochs=60 batch_size=2
"""
import logging
import math
import os
import sys
from typing import Any

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import dinov3.distributed as distributed
from dinov3.eval.detection.config import DetectionHeadConfig, DetectionTrainConfig
from dinov3.eval.detection.criterion import SetCriterion, build_criterion
from dinov3.eval.detection.dataset import (
    CocoDetection,
    CustomDetectionDataset,
    build_dataloader,
    collate_fn,
    make_detection_eval_transforms,
    make_detection_train_transforms,
)
from dinov3.eval.detection.matcher import HungarianMatcher
from dinov3.eval.detection.models.detr import PostProcess, build_model
from dinov3.eval.detection.models.position_encoding import PositionEncoding
from dinov3.eval.detection.util.box_ops import box_cxcywh_to_xyxy
from dinov3.eval.detection.util.misc import NestedTensor, reduce_dict
from dinov3.logging import MetricLogger, SmoothedValue

logger = logging.getLogger("dinov3")


def is_main_process():
    if torch.distributed.is_initialized():
        return distributed.get_rank() == 0
    return True


def build_detection_model(config: DetectionTrainConfig):
    """Build the detection model from config. Supports DETR and Faster R-CNN."""
    from dinov3.hub.backbones import dinov3_vit7b16, dinov3_vitl16plus

    backbone_registry = {
        "dinov3_vit7b16": (dinov3_vit7b16, 3, 768),
        "dinov3_vitl16plus": (dinov3_vitl16plus, 2, 1024),
    }

    model_name = config.model
    if model_name not in backbone_registry:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(backbone_registry.keys())}")

    backbone_fn, n_windows_sqrt, embed_dim = backbone_registry[model_name]

    # Build backbone
    backbone = backbone_fn(pretrained=True)

    detector_type = config.detector_type

    if detector_type == "faster_rcnn":
        # Build Faster R-CNN
        from dinov3.eval.detection.models.faster_rcnn import build_faster_rcnn

        config.head.layers_to_use = (
            config.head.layers_to_use
            if config.head.layers_to_use is not None
            else [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]
        )
        detector = build_faster_rcnn(backbone, config)

        if not config.train_backbone:
            for name, param in detector.backbone.named_parameters():
                param.requires_grad = False

        return detector, config.head

    if detector_type == "oriented_rcnn":
        # Build Oriented R-CNN
        from dinov3.eval.detection.models.oriented_rcnn import build_oriented_rcnn

        config.head.layers_to_use = (
            config.head.layers_to_use
            if config.head.layers_to_use is not None
            else [m * backbone.n_blocks // 4 - 1 for m in range(1, 5)]
        )
        detector = build_oriented_rcnn(backbone, config)

        if not config.train_backbone:
            for name, param in detector.backbone.named_parameters():
                param.requires_grad = False

        return detector, config.head

    # Default: DETR
    # Configure head
    head_config = config.head
    head_config.num_classes = config.num_classes
    head_config.n_windows_sqrt = n_windows_sqrt
    head_config.hidden_dim = embed_dim
    head_config.proposal_in_stride = backbone.patch_size
    head_config.proposal_tgt_strides = [int(m * backbone.patch_size) for m in (0.5, 1, 2, 4)]

    if head_config.layers_to_use is None:
        n_blocks = backbone.n_blocks
        head_config.layers_to_use = [m * n_blocks // 4 - 1 for m in range(1, 5)]

    # Build DETR model
    detector = build_model(backbone, head_config)

    # Freeze backbone if not training
    if not config.train_backbone:
        for name, param in detector.backbone.named_parameters():
            param.requires_grad = False

    # Optionally freeze encoder
    if not config.train_encoder:
        if detector.transformer.encoder is not None:
            for param in detector.transformer.encoder.parameters():
                param.requires_grad = False

    # Set up for inference
    detector.num_queries = detector.num_queries_one2one
    detector.transformer.two_stage_num_proposals = detector.num_queries

    return detector, head_config


def build_optimizer(detector, config: DetectionTrainConfig):
    """Build optimizer with separate learning rates for backbone, encoder, and decoder."""
    lr = config.optimizer.lr
    lr_backbone = config.optimizer.lr_backbone
    weight_decay = config.optimizer.weight_decay

    param_dicts = [
        {
            "params": [
                p
                for n, p in detector.named_parameters()
                if "backbone" not in n
                and "transformer.encoder" not in n
                and p.requires_grad
            ],
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in detector.named_parameters()
                if "transformer.encoder" in n and p.requires_grad
            ],
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in detector.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=lr,
        weight_decay=weight_decay,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
    )
    return optimizer


def build_lr_scheduler(optimizer, config: DetectionTrainConfig, epoch_length: int):
    """Build learning rate scheduler."""
    scheduler_cfg = config.scheduler

    if scheduler_cfg.type == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=scheduler_cfg.milestones,
            gamma=0.1,
        )
    elif scheduler_cfg.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs * epoch_length,
        )
    elif scheduler_cfg.type == "warmup_multistep":
        # Linear warmup + multistep
        warmup_iters = scheduler_cfg.warmup_epochs * epoch_length

        def lr_lambda(current_step):
            if current_step < warmup_iters:
                return scheduler_cfg.warmup_factor + (1.0 - scheduler_cfg.warmup_factor) * current_step / max(
                    1, warmup_iters
                )
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        multistep_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[m * epoch_length for m in scheduler_cfg.milestones],
            gamma=0.1,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, multistep_scheduler],
            milestones=[warmup_iters],
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_cfg.type}")

    return scheduler


def train_one_epoch(
    detector,
    criterion,
    dataloader,
    optimizer,
    device,
    epoch,
    max_norm=0.1,
    scaler=None,
    model_dtype=None,
    detector_type="detr",
):
    """Train for one epoch."""
    detector.train()
    # Keep backbone in eval mode if frozen
    if not any(p.requires_grad for p in detector.backbone.parameters()):
        detector.backbone.eval()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("loss", SmoothedValue(window_size=10, fmt="{value:.3f}"))
    header = f"Epoch: [{epoch}]"

    for samples, targets in metric_logger.log_every(dataloader, 50, header):
        samples: NestedTensor = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            "cuda", dtype=model_dtype, enabled=(model_dtype is not None)
        ):
            if detector_type in ("faster_rcnn", "oriented_rcnn"):
                loss_dict = detector(samples, targets)
                loss = sum(loss_dict.values())
            else:
                outputs = detector(samples)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(detector.parameters(), max_norm)
            optimizer.step()

        # Reduce loss dict across processes
        loss_dict_reduced = reduce_dict(loss_dict)
        loss_reduced = sum(loss_dict_reduced.values())

        metric_logger.update(loss=loss_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(detector, criterion, dataloader, postprocessor, device, detector_type="detr"):
    """Evaluate model on validation set."""
    detector.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Eval:"

    coco_evaluator = None
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_evaluator = True  # Will set up properly below
    except ImportError:
        logger.warning("pycocotools not installed, skipping COCO evaluation")

    all_results = []
    loss_stats = []

    for samples, targets in metric_logger.log_every(dataloader, 50, header):
        samples: NestedTensor = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if detector_type in ("faster_rcnn", "oriented_rcnn"):
            outputs = detector(samples, targets)
            loss_dict = outputs["losses"]
            loss = sum(loss_dict.values())
        else:
            outputs = detector(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_stats.append(loss.item())

        # Post-process predictions
        if detector_type in ("faster_rcnn", "oriented_rcnn"):
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            is_oriented = (detector_type == "oriented_rcnn")
            for i, target in enumerate(targets):
                image_id = target.get("image_id", 0)
                if isinstance(image_id, torch.Tensor):
                    image_id = image_id.item()
                scores = outputs["pred_logits"][i]
                boxes = outputs["pred_boxes"][i]  # cxcywh normalized (4-param) or cxcywht normalized (5-param)

                if scores.numel() == 0:
                    continue

                max_scores, labels = scores.max(dim=1)
                keep = max_scores > 0
                if not keep.any():
                    continue

                boxes_kept = boxes[keep]
                orig_h, orig_w = orig_target_sizes[i].tolist()

                if is_oriented:
                    # Convert oriented (cx, cy, w, h, theta) to axis-aligned (x1, y1, x2, y2)
                    # by using the circumscribed axis-aligned box of the rotated box
                    from dinov3.eval.detection.models.oriented_rcnn import obb_xywht_to_xyxy
                    obb_pixel = boxes_kept.clone()
                    obb_pixel[:, 0] *= orig_w
                    obb_pixel[:, 1] *= orig_h
                    obb_pixel[:, 2] *= orig_w
                    obb_pixel[:, 3] *= orig_h
                    corners = obb_xywht_to_xyxy(obb_pixel)  # [N, 8]
                    x1 = corners[:, 0::2].min(dim=1).values
                    y1 = corners[:, 1::2].min(dim=1).values
                    x2 = corners[:, 0::2].max(dim=1).values
                    y2 = corners[:, 1::2].max(dim=1).values
                    x1.clamp_(min=0, max=orig_w)
                    y1.clamp_(min=0, max=orig_h)
                    x2.clamp_(min=0, max=orig_w)
                    y2.clamp_(min=0, max=orig_h)
                else:
                    boxes_kept = box_cxcywh_to_xyxy(boxes_kept)
                    boxes_kept[:, 0] *= orig_w
                    boxes_kept[:, 1] *= orig_h
                    boxes_kept[:, 2] *= orig_w
                    boxes_kept[:, 3] *= orig_h
                    boxes_kept[:, 0].clamp_(min=0, max=orig_w)
                    boxes_kept[:, 1].clamp_(min=0, max=orig_h)
                    boxes_kept[:, 2].clamp_(min=0, max=orig_w)
                    boxes_kept[:, 3].clamp_(min=0, max=orig_h)
                    x1, y1, x2, y2 = boxes_kept[:, 0], boxes_kept[:, 1], boxes_kept[:, 2], boxes_kept[:, 3]

                for score, label, x1v, y1v, x2v, y2v in zip(max_scores[keep], labels[keep], x1, y1, x2, y2):
                    all_results.append(
                        {
                            "image_id": image_id,
                            "category_id": label.item(),
                            "bbox": [x1v.item(), y1v.item(), (x2v - x1v).item(), (y2v - y1v).item()],
                            "score": score.item(),
                        }
                    )
        else:
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessor(outputs, target_sizes=target_sizes, original_target_sizes=orig_target_sizes)

            for target, result in zip(targets, results):
                image_id = target.get("image_id", 0)
                if isinstance(image_id, torch.Tensor):
                    image_id = image_id.item()
                for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
                    x1, y1, x2, y2 = box.tolist()
                    all_results.append(
                        {
                            "image_id": image_id,
                            "category_id": label.item(),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score.item(),
                        }
                    )

    avg_loss = sum(loss_stats) / max(len(loss_stats), 1)

    # COCO evaluation if we have ground truth annotations
    eval_stats = {"val_loss": avg_loss}
    if coco_evaluator and len(all_results) > 0:
        try:
            import json
            import tempfile

            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            logger.info(f"Collected {len(all_results)} detections for evaluation")
            logger.info("Use pycocotools COCOeval for full COCO metrics")
        except Exception as e:
            logger.warning(f"COCO evaluation failed: {e}")

    metric_logger.synchronize_between_processes()
    return eval_stats


def train_detection(config: DetectionTrainConfig):
    """Main training function."""
    detector_type = config.detector_type
    logger.info(f"Detector type: {detector_type}")

    # 1. Build model
    logger.info("Building detection model...")
    detector, head_config = build_detection_model(config)

    device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
    detector = detector.to(device)

    n_params = sum(p.numel() for p in detector.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {n_params / 1e6:.1f}M")

    # 2. DDP
    if torch.distributed.is_initialized():
        detector = nn.parallel.DistributedDataParallel(detector, device_ids=[device])
        detector_without_ddp = detector.module
    else:
        detector_without_ddp = detector

    # 3. Build criterion (only for DETR; Faster R-CNN and Oriented R-CNN have built-in loss)
    criterion = None
    if detector_type == "detr":
        criterion, weight_dict = build_criterion(config)
        criterion = criterion.to(device)

    # 4. Build dataloaders
    logger.info("Building dataloaders...")
    train_transforms = make_detection_train_transforms(
        random_size_range=config.transforms.train_short_side_range,
        random_size_max=config.transforms.train_max_size,
        flip_prob=config.transforms.train_flip_prob,
    )
    eval_transforms = make_detection_eval_transforms(
        img_size=config.transforms.eval_short_side,
        random_size_max=config.transforms.eval_max_size,
    )

    # Detect dataset type and build accordingly
    if config.datasets.train_ann_file:
        train_dataset = CocoDetection(
            img_dir=config.datasets.train_img_dir,
            ann_file=config.datasets.train_ann_file,
            transforms=train_transforms,
        )
    else:
        train_dataset = CustomDetectionDataset(
            img_dir=config.datasets.train_img_dir,
            ann_file=config.datasets.train_ann_file,
            transforms=train_transforms,
        )

    if config.datasets.val_ann_file:
        val_dataset = CocoDetection(
            img_dir=config.datasets.val_img_dir,
            ann_file=config.datasets.val_ann_file,
            transforms=eval_transforms,
        )
    else:
        val_dataset = None

    train_dataloader = build_dataloader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        distributed=torch.distributed.is_initialized(),
        drop_last=True,
    )

    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = build_dataloader(
            val_dataset,
            batch_size=1,
            num_workers=config.num_workers,
            distributed=torch.distributed.is_initialized(),
            drop_last=False,
        )

    # 5. Build optimizer and scheduler
    optimizer = build_optimizer(detector_without_ddp, config)
    scheduler = build_lr_scheduler(optimizer, config, epoch_length=len(train_dataloader))

    # Load checkpoint if resuming
    start_epoch = 0
    best_val_loss = float("inf")
    if config.load_from and os.path.exists(config.load_from):
        logger.info(f"Resuming from checkpoint: {config.load_from}")
        checkpoint = torch.load(config.load_from, map_location="cpu")
        detector_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_val_loss" in checkpoint:
            best_val_loss = checkpoint["best_val_loss"]

    # 6. Post-processor for evaluation (DETR only; Faster R-CNN / Oriented R-CNN have built-in postprocessing)
    postprocessor = PostProcess(topk=head_config.topk, reparam=head_config.reparam) if detector_type == "detr" else None

    # 7. Training loop
    logger.info(f"Starting training from epoch {start_epoch} to {config.epochs}")
    for epoch in range(start_epoch, config.epochs):
        if torch.distributed.is_initialized():
            train_dataloader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            detector,
            criterion,
            train_dataloader,
            optimizer,
            device,
            epoch,
            max_norm=config.optimizer.gradient_clip,
            detector_type=detector_type,
        )

        # Step epoch-based scheduler
        if isinstance(scheduler, (torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.StepLR)):
            scheduler.step()

        logger.info(
            f"Epoch {epoch}: loss={train_stats.get('loss', 0):.4f}, lr={train_stats.get('lr', 0):.6f}"
        )

        # Evaluation
        if val_dataloader is not None and (epoch + 1) % config.eval_interval == 0:
            eval_stats = evaluate(
                detector_without_ddp, criterion, val_dataloader, postprocessor, device,
                detector_type=detector_type,
            )
            val_loss = eval_stats.get("val_loss", float("inf"))
            logger.info(f"Epoch {epoch}: val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if is_main_process():
                    save_checkpoint(
                        detector_without_ddp, optimizer, epoch, best_val_loss, config.output_dir, is_best=True
                    )

        # Save periodic checkpoint
        if is_main_process() and (epoch + 1) % 10 == 0:
            save_checkpoint(
                detector_without_ddp, optimizer, epoch, best_val_loss, config.output_dir, is_best=False
            )

    # Final save
    if is_main_process():
        save_checkpoint(
            detector_without_ddp, optimizer, config.epochs - 1, best_val_loss, config.output_dir, is_best=False
        )
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    return {"best_val_loss": best_val_loss}


def save_checkpoint(detector, optimizer, epoch, best_val_loss, output_dir, is_best=False):
    """Save model checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "model": detector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    if is_best:
        path = os.path.join(output_dir, "checkpoint_best.pth")
    else:
        path = os.path.join(output_dir, f"checkpoint_{epoch:04d}.pth")
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")
