"""
3x training schedule (36 epochs) for DINOv3 + MMDetection.

To use, include in your _base_ config:
    _base_ = ['../_base_/schedule_3x.py']
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
# Warmup for the first 1000 iterations, then divide LR by 10 at epochs 27 and 33
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000,
    ),
    dict(
        type="MultiStepLR",
        by_epoch=True,
        milestones=[27, 33],
        gamma=0.1,
    ),
]

# Epoch-based training
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=36, val_interval=1)
