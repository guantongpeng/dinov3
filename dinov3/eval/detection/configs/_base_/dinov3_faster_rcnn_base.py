"""
Faster R-CNN base config with DINOv3 ViT backbone.

Key design:
- Backbone outputs 4 single-scale feature maps from evenly-spaced ViT blocks
- FPN neck builds the multi-scale feature pyramid
- RPN anchor strides must match the combined backbone+FPN output strides

Override the backbone section in concrete configs to set model-specific
parameters (model_name, layers_to_use, in_channels).
"""

# --------------------------------------------------------------------------
# Helper: compute model-specific values.
# These are set in the concrete config file that _base_extends this one.
# Provide sensible defaults for ViT-B/16.
# --------------------------------------------------------------------------
_model_name = "dinov3_vitb16"
_embed_dim = 768
_n_blocks = 12
_patch_size = 16
_layers_to_use = [2, 5, 8, 11]         # last block of each quarter for 12-block ViT
_num_feature_levels = len(_layers_to_use)  # 4
_out_indices = (0, 1, 2, 3)
_frozen_stages = -1                       # freeze backbone by default

# FPN will produce stride = [patch_size, patch_size*2, patch_size*4, patch_size*8, patch_size*16]
# = [16, 32, 64, 128, 256] for patch_size=16
_fpn_num_outs = 5
_fpn_in_channels = [_embed_dim] * _num_feature_levels

model = dict(
    type="FasterRCNN",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=_patch_size,   # critical: ViT requires input dims divisible by patch_size
    ),
    backbone=dict(
        type="DinoVisionTransformerBackbone",
        model_name=_model_name,
        pretrained=True,
        layers_to_use=_layers_to_use,
        out_indices=_out_indices,
        use_layernorm=True,
        out_channels=None,
        frozen_stages=_frozen_stages,
    ),
    neck=dict(
        type="FPN",
        in_channels=_fpn_in_channels,
        out_channels=256,
        num_outs=_fpn_num_outs,
        start_level=0,
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[_patch_size * (2 ** i) for i in range(_fpn_num_outs)],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=1.0),
    ),
    roi_head=dict(
        type="StandardRoIHead",
        bbox_roi_extractor=dict(
            type="SingleRoIExtractor",
            roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[_patch_size * (2 ** i) for i in range(_fpn_num_outs - 1)],
        ),
        bbox_head=dict(
            type="Shared2FCBBoxHead",
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,  # COCO
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                loss_weight=1.0,
            ),
            loss_bbox=dict(type="L1Loss", loss_weight=1.0),
        ),
    ),
    # Training settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            assigner=dict(
                type="MaxIoUAssigner",
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1,
            ),
            sampler=dict(
                type="RandomSampler",
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True,
            ),
            pos_weight=-1,
            debug=False,
        ),
    ),
    # Testing settings
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type="nms", iou_threshold=0.7),
            min_bbox_size=0,
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type="nms", iou_threshold=0.5),
            max_per_img=100,
        ),
    ),
)
