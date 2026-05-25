# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
MMDetection training entry point for DINOv3 backbone.

Usage:
    # Single GPU training
    python -m dinov3.eval.detection.mm_train \\
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_1x_coco.py \\
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16

    # Multi-GPU training (via torchrun)
    torchrun --nproc_per_node=4 -m dinov3.eval.detection.mm_train \\
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitl16plus_3x_coco.py \\
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitl16plus \\
        --batch-size 8 --amp

    # Resume from checkpoint
    python -m dinov3.eval.detection.mm_train \\
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_1x_coco.py \\
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16 \\
        --resume ./work_dirs/faster_rcnn_dinov3_vitb16/epoch_8.pth

    # Unfreeze backbone for end-to-end fine-tuning
    python -m dinov3.eval.detection.mm_train \\
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_1x_coco.py \\
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_e2e \\
        --train-backbone
"""

import argparse
import logging
import os
import sys

logger = logging.getLogger("dinov3")


def _check_mmdetection():
    """Verify MMDetection is installed. Print install instructions if not."""
    missing = []
    try:
        import mmdet  # noqa: F401
    except ImportError:
        missing.append("mmdet")
    try:
        import mmengine  # noqa: F401
    except ImportError:
        missing.append("mmengine")
    try:
        import mmcv  # noqa: F401
    except ImportError:
        missing.append("mmcv")

    if missing:
        print(
            "MMDetection ecosystem packages not found: " + ", ".join(missing) + "\n\n"
            "Install with:\n"
            "  pip install openmim\n"
            "  mim install mmdet\n\n"
            "Or for specific versions:\n"
            "  pip install mmengine mmcv mmdet\n"
        )
        sys.exit(1)

    return mmdet, mmengine, mmcv


def register_custom_backbone():
    """Register DinoVisionTransformerBackbone with MMEngine's MODELS registry.

    This must be called before the config is parsed, so that the config's
    ``type='DinoVisionTransformerBackbone'`` can be resolved.
    """
    from mmengine.registry import MODELS
    from dinov3.eval.detection.mm_backbone import DinoVisionTransformerBackbone

    MODELS.register_module(
        name="DinoVisionTransformerBackbone",
        module=DinoVisionTransformerBackbone,
        force=True,
    )
    logger.info("Registered DinoVisionTransformerBackbone with MMEngine MODELS registry.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINOv3 + MMDetection detection model."
    )
    parser.add_argument(
        "config",
        help="Path to MMDetection config file (.py)",
    )
    parser.add_argument(
        "--work-dir",
        required=True,
        help="Output directory for logs and checkpoints.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint (.pth file path).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size in the config.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision (AMP) training.",
    )
    parser.add_argument(
        "--train-backbone",
        action="store_true",
        help="Unfreeze backbone for end-to-end fine-tuning.",
    )
    parser.add_argument(
        "--backbone-weights",
        default=None,
        help="Path to backbone weights (overrides pretrained loading).",
    )
    parser.add_argument(
        "--layers-to-use",
        type=int,
        nargs="+",
        default=None,
        help="ViT block indices for feature extraction (e.g. --layers-to-use 5 11 17 23).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--launcher",
        default="none",
        choices=["none", "pytorch", "slurm"],
        help="Launcher type. Use 'pytorch' with torchrun.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="Local rank for distributed training (set by torchrun).",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    """Apply command-line arguments as config overrides."""
    if args.batch_size is not None:
        cfg.train_dataloader.batch_size = args.batch_size
        logger.info(f"Batch size overridden to: {args.batch_size}")

    if args.train_backbone:
        cfg.model.backbone.frozen_stages = 0
        cfg.optim_wrapper.paramwise_cfg.custom_keys["backbone"] = dict(
            lr_mult=1.0
        )
        logger.info("Backbone unfrozen for end-to-end fine-tuning.")

    if args.backbone_weights is not None:
        cfg.model.backbone.pretrained = False
        cfg.model.backbone.init_cfg = dict(
            type="Pretrained", checkpoint=args.backbone_weights
        )
        logger.info(f"Using custom backbone weights: {args.backbone_weights}")

    if args.layers_to_use is not None:
        cfg.model.backbone.layers_to_use = args.layers_to_use
        logger.info(f"layers_to_use overridden to: {args.layers_to_use}")

    if args.lr is not None:
        cfg.optim_wrapper.optimizer.lr = args.lr
        logger.info(f"Learning rate overridden to: {args.lr}")

    if args.amp:
        cfg.optim_wrapper.type = "AmpOptimWrapper"
        cfg.optim_wrapper.loss_scale = "dynamic"
        logger.info("AMP training enabled.")

    if args.resume:
        cfg.resume = True
        cfg.load_from = args.resume
        logger.info(f"Resuming from: {args.resume}")

    cfg.work_dir = args.work_dir
    cfg.randomness = dict(seed=args.seed)

    return cfg


def main():
    args = parse_args()

    # 1. Verify MMDetection is available
    mmdet, mmengine, mmcv = _check_mmdetection()

    # 2. Register custom DINOv3 backbone module
    register_custom_backbone()

    # 3. Load config
    from mmengine.config import Config

    cfg = Config.fromfile(args.config)

    # Merge with CLI overrides
    cfg = apply_cli_overrides(cfg, args)

    # 4. Create output directory and save effective config
    os.makedirs(cfg.work_dir, exist_ok=True)
    cfg.dump(os.path.join(cfg.work_dir, "effective_config.py"))

    logger.info(f"Config saved to {cfg.work_dir}/effective_config.py")
    logger.info(f"Model type: {cfg.model.type}")
    logger.info(f"Backbone: {cfg.model.backbone.model_name}")
    logger.info(f"Layers to use: {cfg.model.backbone.layers_to_use}")
    logger.info(f"Frozen stages: {cfg.model.backbone.frozen_stages}")
    logger.info(f"Work directory: {cfg.work_dir}")

    # 5. Build and launch Runner
    from mmengine.runner import Runner

    runner = Runner.from_cfg(cfg)

    # 6. Train
    runner.train()

    return 0


if __name__ == "__main__":
    sys.exit(main())
