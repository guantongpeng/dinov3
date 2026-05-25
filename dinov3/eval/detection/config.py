# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from dinov3.eval.detection.models.position_encoding import PositionEncoding


@dataclass(kw_only=True)
class DetectionHeadConfig:
    num_classes: int = 91  # 91 classes in COCO
    # Deformable DETR tricks
    with_box_refine: bool = True
    two_stage: bool = True
    # DINO DETR tricks
    mixed_selection: bool = True
    look_forward_twice: bool = True  # was default False
    # Hybrid Matching tricks
    k_one2many: int = 6  # was 5
    lambda_one2many: float = 1.0
    num_queries_one2one: int = 300  # number of query slots for one_to_one matching
    num_queries_one2many: int = 1500  # was 0, number of query slots for one_to_many matching
    """
    Absolute coordinates & box regression reparameterization.
    If true, we use absolute coordindates & reparameterization for bounding boxes.
    """
    reparam: bool = True
    topk: int = 100

    # * Backbone
    # type of positional embedding to use on top of the image features
    position_embedding: PositionEncoding = PositionEncoding.SINE
    num_feature_levels: int = 1  # number of feature levels

    # * Transformer
    dec_layers: int = 6  # number of decoding layers in the transformer
    dim_feedforward: int = 2048  # intermediate size of the feedforward layers in the transformer blocks
    hidden_dim: int = 256  # size of the embeddings (dimension of the transformer)
    dropout: float = 0.0  # dropout applied in the transformer, was 0.1
    nheads: int = 8  # number of attention heads inside the transformer's attentions
    norm_type: str = "pre_norm"

    # Loss
    aux_loss: bool = True  # auxiliary decoding losses (loss at each layer)

    # * dev: proposals
    proposal_feature_levels: int = 4  # was 1
    proposal_min_size: int = 50
    # * dev decoder: global decoder
    decoder_type: str = "global_rpe_decomp"  # was deform
    decoder_use_checkpoint: bool = False
    decoder_rpe_hidden_dim: int = 512
    decoder_rpe_type: str = "linear"

    # Custom
    add_transformer_encoder: bool = True
    num_encoder_layers: int = 6
    layers_to_use: list[int] | None = None
    blocks_to_train: list[int] | None = None
    n_windows_sqrt: int = 0
    proposal_in_stride: int | None = None
    proposal_tgt_strides: list[int] | None = None
    backbone_use_layernorm: bool = False  # whether to use layernorm on each layer of the backbone's features


@dataclass(kw_only=True)
class MatcherConfig:
    matcher_cost_class: float = 2.0
    matcher_cost_bbox: float = 5.0
    matcher_cost_giou: float = 2.0


@dataclass(kw_only=True)
class LossConfig:
    cls_loss_coef: float = 2.0
    bbox_loss_coef: float = 5.0
    giou_loss_coef: float = 2.0


@dataclass(kw_only=True)
class OptimizerConfig:
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    lr_linear_proj_mult: float = 0.1
    weight_decay: float = 1e-4
    gradient_clip: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.999


@dataclass(kw_only=True)
class SchedulerConfig:
    type: str = "multistep"
    milestones: list[int] = field(default_factory=lambda: [40, 55])
    warmup_epochs: int = 1
    warmup_factor: float = 1e-3


@dataclass(kw_only=True)
class TransformConfig:
    train_short_side_range: tuple[int, int] = (480, 800)
    train_max_size: int = 1333
    train_flip_prob: float = 0.5
    eval_short_side: int = 800
    eval_max_size: int = 1333


@dataclass(kw_only=True)
class DatasetsConfig:
    root: str = MISSING
    train: str = ""  # dataset name or path
    val: str = ""
    train_img_dir: str = ""
    train_ann_file: str = ""
    val_img_dir: str = ""
    val_ann_file: str = ""


@dataclass(kw_only=True)
class RPNConfig:
    """Faster R-CNN Region Proposal Network config."""

    # Anchor parameters
    anchor_sizes: tuple[int, ...] = (32, 64, 128, 256, 512)
    anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    anchor_strides: tuple[int, ...] = (4, 8, 16, 32, 64)

    # RPN training
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 1000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_test: int = 500
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5

    # RPN loss weights
    rpn_cls_loss_coef: float = 1.0
    rpn_reg_loss_coef: float = 1.0


@dataclass(kw_only=True)
class RoIHeadConfig:
    """Faster R-CNN RoI (box) head config."""

    # RoI sampling
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_nms_thresh: float = 0.5

    # RoI pooling
    box_roi_output_size: int = 7
    box_roi_sampling_ratio: int = 2

    # Detection head
    box_head_fc_dim: int = 1024
    box_head_num_fc: int = 2

    # Loss weights
    box_cls_loss_coef: float = 1.0
    box_reg_loss_coef: float = 1.0


@dataclass(kw_only=True)
class FasterRCNNConfig:
    """Faster R-CNN specific config."""

    rpn: RPNConfig = field(default_factory=RPNConfig)
    roi_head: RoIHeadConfig = field(default_factory=RoIHeadConfig)

    # Feature pyramid
    num_feature_levels: int = 5  # P2-P6
    feature_hidden_dim: int = 256
    use_fpn: bool = True


@dataclass(kw_only=True)
class OrientedRCNNConfig:
    """Oriented R-CNN specific config for rotated object detection."""

    rpn: RPNConfig = field(default_factory=RPNConfig)
    roi_head: RoIHeadConfig = field(default_factory=RoIHeadConfig)

    # Feature pyramid
    num_feature_levels: int = 5  # P2-P6
    feature_hidden_dim: int = 256
    use_fpn: bool = True

    # Rotation-specific
    num_angles: int = 180  # number of angle bins (for angle prediction granularity)
    angle_version: str = "le90"  # angle representation: le90 ([-90, 90)), le135 ([-135, 45)), oc ([0, 180))


@dataclass(kw_only=True)
class DetectionTrainConfig:
    """Full config for detection training, combining head config + training params."""

    # Model
    model: str | None = None  # backbone model name (e.g. "dinov3_vit7b16")
    pretrained_weights: str | None = None  # path to backbone weights
    load_from: str | None = None  # path to detection checkpoint to resume from

    # Detector type: "detr", "faster_rcnn", or "oriented_rcnn"
    detector_type: str = "detr"

    head: DetectionHeadConfig = field(default_factory=DetectionHeadConfig)
    faster_rcnn: FasterRCNNConfig = field(default_factory=FasterRCNNConfig)
    oriented_rcnn: OrientedRCNNConfig = field(default_factory=OrientedRCNNConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    loss: LossConfig = field(default_factory=LossConfig)

    # Data
    datasets: DatasetsConfig = field(default_factory=DatasetsConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    num_classes: int = 91  # will be forwarded to head config

    # Training
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    epochs: int = 60
    batch_size: int = 2
    num_workers: int = 4
    seed: int = 42
    eval_interval: int = 5  # evaluate every N epochs
    output_dir: str = ""

    # Backbone
    train_backbone: bool = False  # whether to train backbone
    train_encoder: bool = True  # whether to train transformer encoder
    backbone_lr_mult: float = 0.1  # backbone lr multiplier
