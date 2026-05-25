"""
Faster R-CNN with DINOv3 ViT-B/16 backbone, 1x schedule, COCO dataset.

Usage:
    python -m dinov3.eval.detection.mm_train \
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_1x_coco.py \
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16
"""

_base_ = [
    "../_base_/dinov3_faster_rcnn_base.py",
    "../_base_/coco_detection.py",
    "../_base_/schedule_1x.py",
    "../_base_/default_runtime.py",
]

# ViT-B/16: 12 blocks, 768-dim, patch_size=16
model = dict(
    backbone=dict(
        model_name="dinov3_vitb16",
        layers_to_use=[2, 5, 8, 11],
    ),
    neck=dict(
        in_channels=[768, 768, 768, 768],
    ),
)

# Batch size for ViT-B (can be larger than ViT-L)
train_dataloader = dict(batch_size=2)
