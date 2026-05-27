"""
Faster R-CNN with DINOv3 ViT-B/16 backbone, DIOR dataset fine-tuning.

DIOR has 20 classes, 5862 train / 5863 val images.
Backbone is frozen (DINOv3 pretrained), only FPN + RPN + RoI Head are trained.

Usage:
    python -m dinov3.eval.detection.mm_train \
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_dior.py \
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_dior

    # With backbone unfrozen:
    python -m dinov3.eval.detection.mm_train \
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_dior.py \
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_dior \
        --train-backbone
"""

_base_ = [
    "../_base_/dinov3_faster_rcnn_base.py",
    "../_base_/default_runtime.py",
]

# Dataset settings for DIOR
dataset_type = "CocoDataset"
data_root = "data/DIOR/"
num_classes = 20

# DIOR class names (must match the order in annotations' categories)
dior_classes = (
    "airplane", "airport", "baseballfield", "basketballcourt", "bridge",
    "chimney", "dam", "Expressway-Service-area", "Expressway-toll-station",
    "golffield", "groundtrackfield", "harbor", "overpass", "ship",
    "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill",
)

# Override default COCO metainfo with DIOR classes
metainfo = dict(classes=dior_classes)

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
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_train2019.json",
        data_prefix=dict(img="train2019/"),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val2019.json",
        data_prefix=dict(img="val2019/"),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val2019.json",
    metric="bbox",
    format_only=False,
)

test_evaluator = val_evaluator

# Training: 12 epochs with backbone frozen, eval every 3 epochs
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=3)

# Optimizer: AdamW with layer-wise LR decay
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.0),
            "neck": dict(lr_mult=1.0),
            "rpn_head": dict(lr_mult=1.0),
            "roi_head": dict(lr_mult=1.0),
        }
    ),
)

# LR schedule: linear warmup + step decay at epochs 8 and 11
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", by_epoch=True, milestones=[8, 11], gamma=0.1),
]

# Default hooks
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="coco/bbox_mAP"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

# ViT-B/16 backbone (frozen) with converted pretrained weights
model = dict(
    backbone=dict(
        model_name="dinov3_vitb16",
        pretrained=False,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="data/pretrained_weights/dinov3_vitb16_timm.pth",
        ),
        layers_to_use=[2, 5, 8, 11],
        frozen_stages=-1,
    ),
    neck=dict(
        in_channels=[768, 768, 768, 768],
    ),
    roi_head=dict(
        bbox_head=dict(
            num_classes=num_classes,
        ), 
    ),
)
