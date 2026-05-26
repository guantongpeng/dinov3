"""
Faster R-CNN with DINOv3 ViT-B/16 backbone, COCO128 dataset fine-tuning.

This config is for quick fine-tuning experiments on the COCO128 subset.
Uses 30 epochs with backbone frozen by default.

Usage:
    python -m dinov3.eval.detection.mm_train \
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_coco128.py \
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_coco128

    # With backbone unfrozen:
    python -m dinov3.eval.detection.mm_train \
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_coco128.py \
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_coco128 \
        --train-backbone
"""

_base_ = [
    "../_base_/dinov3_faster_rcnn_base.py",
    "../_base_/default_runtime.py",
]

# Dataset settings for COCO128
dataset_type = "CocoDataset"
data_root = "data/coco128/"

num_classes = 80

# Train pipeline
train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="RandomResize",
        scale=[(480, 1333), (800, 1333)],
        keep_ratio=True,
    ),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

# Test pipeline
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=None),
    dict(type="Resize", scale=(800, 1333), keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# DataLoaders
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_train2017.json",
        data_prefix=dict(img="train2017/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val2017.json",
    metric="bbox",
    format_only=False,
)

test_evaluator = val_evaluator

# Training settings: 30 epochs with backbone frozen
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=30, val_interval=5)

# Optimizer wrapper
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "neck": dict(lr_mult=1.0),
            "rpn_head": dict(lr_mult=1.0),
            "roi_head": dict(lr_mult=1.0),
        }
    ),
)

# LR schedule: warmup + step decay
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=100),
    dict(type="MultiStepLR", by_epoch=True, milestones=[20, 27], gamma=0.1),
]

# Default hooks
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=5, max_keep_ckpts=3, save_best="coco/bbox_mAP"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

# ViT-B/16: 12 blocks, 768-dim, patch_size=16
model = dict(
    backbone=dict(
        model_name="dinov3_vitb16",
        pretrained=False,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="data/pretrained_weights/dinov3_vitb16_timm.pth",
        ),
        layers_to_use=[2, 5, 8, 11],
    ),
    neck=dict(
        in_channels=[768, 768, 768, 768],
    ),
)

