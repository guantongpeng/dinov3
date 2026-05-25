"""
Faster R-CNN with DINOv3 ViT-L/16+ backbone, 3x schedule, COCO dataset.

ViT-L/16+ uses SwiGLU FFN (ffn_ratio=6) for higher capacity.

Usage:
    python -m dinov3.eval.detection.mm_train \
        dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitl16plus_3x_coco.py \
        --work-dir ./work_dirs/faster_rcnn_dinov3_vitl16plus
"""

_base_ = [
    "../_base_/dinov3_faster_rcnn_base.py",
    "../_base_/coco_detection.py",
    "../_base_/schedule_3x.py",
    "../_base_/default_runtime.py",
]

# ViT-L/16+: 24 blocks, 1024-dim, SwiGLU FFN
model = dict(
    backbone=dict(
        model_name="dinov3_vitl16plus",
        layers_to_use=[5, 11, 17, 23],
    ),
    neck=dict(
        in_channels=[1024, 1024, 1024, 1024],
    ),
)

train_dataloader = dict(batch_size=2)
