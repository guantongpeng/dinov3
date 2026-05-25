"""
Mask R-CNN with DINOv3 ViT-L/16 backbone, 1x schedule, COCO dataset.

Usage:
    python -m dinov3.eval.detection.mm_train \
        dinov3/eval/detection/configs/mask_rcnn/mask_rcnn_dinov3_vitl16_1x_coco.py \
        --work-dir ./work_dirs/mask_rcnn_dinov3_vitl16
"""

_base_ = [
    "../_base_/dinov3_mask_rcnn_base.py",
    "../_base_/coco_detection.py",
    "../_base_/schedule_1x.py",
    "../_base_/default_runtime.py",
]

# ViT-L/16: 24 blocks, 1024-dim, patch_size=16
model = dict(
    backbone=dict(
        model_name="dinov3_vitl16",
        layers_to_use=[5, 11, 17, 23],
    ),
    neck=dict(
        in_channels=[1024, 1024, 1024, 1024],
    ),
)

train_dataloader = dict(batch_size=2)
