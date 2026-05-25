"""
1x training schedule (12 epochs) for DINOv3 + MMDetection.

To use, include in your _base_ config:
    _base_ = ['../_base_/schedule_1x.py']
"""

# Optimizer wrapper
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),     # lower LR for pretrained ViT
            "neck": dict(lr_mult=1.0),
            "rpn_head": dict(lr_mult=1.0),
            "roi_head": dict(lr_mult=1.0),
        }
    ),
)

# Learning rate schedule: linear warmup + step decay
# Warmup for the first 500 iterations, then divide LR by 10 at epochs 8 and 11
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="MultiStepLR",
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

# Epoch-based training
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
