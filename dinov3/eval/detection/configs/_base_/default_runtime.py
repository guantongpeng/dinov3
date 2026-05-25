"""
Default runtime settings for DINOv3 + MMDetection training.

To use, include in your _base_ config:
    _base_ = ['../_base_/default_runtime.py']
"""

default_scope = "mmdet"

# Default hooks
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=3, save_best="coco/bbox_mAP"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

# Environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

# Visualization
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
)

# Logging
log_level = "INFO"
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)

# Resume / load
load_from = None
resume = False

# Training steps: train on training set, validate on validation set
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
