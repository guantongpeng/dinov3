# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Faster R-CNN model for object detection with DINOv3 backbone.

Architecture:
    Backbone (ViT) → Feature Pyramid → RPN → proposals
                                   → RoI Align + Box Head → detections
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from ..util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from ..util.misc import NestedTensor
from .utils import LayerNorm2D


def _get_clones(module, N):
    return nn.ModuleList([module for _ in range(N)])


class FeaturePyramid(nn.Module):
    """Build a simple feature pyramid from ViT backbone features.

    Since ViT outputs features at a single resolution (patch_size stride),
    we create multi-scale features using strided convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, num_levels: int = 5):
        super().__init__()
        self.num_levels = num_levels
        self.out_channels = out_channels

        # Lateral connection to project backbone features
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.lateral_norm = LayerNorm2D(out_channels)

        # Top-down pathway: smooth convs for each level
        self.smooth_convs = nn.ModuleList()
        for _ in range(num_levels):
            self.smooth_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    LayerNorm2D(out_channels),
                )
            )

        # Extra downsampling layers for deeper levels
        self.downsample_convs = nn.ModuleList()
        for _ in range(num_levels - 1):
            self.downsample_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build feature pyramid from backbone features.

        Args:
            features: List of [B, C, H, W] feature maps from backbone layers.
                      All at same spatial resolution.

        Returns:
            List of [B, out_channels, H_i, W_i] feature maps at different scales.
        """
        # Take the last layer feature and project
        x = features[-1]
        x = self.lateral_conv(x)
        x = self.lateral_norm(x)
        x = F.relu(x)

        pyramid = []
        for i in range(self.num_levels):
            if i == 0:
                feat = self.smooth_convs[i](x)
            else:
                x = self.downsample_convs[i - 1](x)
                x = F.relu(x)
                feat = self.smooth_convs[i](x)
            pyramid.append(feat)

        return pyramid


class MultiLevelBackbone(nn.Module):
    """DINOv3 backbone wrapper that outputs multi-level features for Faster R-CNN.

    Extracts intermediate features from the ViT backbone and builds a feature pyramid.
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        layers_to_use: List[int],
        embed_dim: int,
        patch_size: int,
        out_channels: int = 256,
        num_levels: int = 5,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.backbone = backbone_model
        self.layers_to_use = layers_to_use
        self.patch_size = patch_size
        self.strides = [patch_size * (2**i) for i in range(num_levels)]

        # Infer embed dims for each selected layer
        n_blocks = self.backbone.n_blocks
        embed_dims = getattr(self.backbone, "embed_dims", [embed_dim] * n_blocks)
        selected_dims = [embed_dims[i] for i in range(n_blocks) if i in layers_to_use]

        # Layer norms for each selected layer
        self.layer_norms = nn.ModuleList()
        if use_layernorm:
            for dim in selected_dims:
                self.layer_norms.append(LayerNorm2D(dim))
        else:
            self.layer_norms = nn.ModuleList([nn.Identity() for _ in selected_dims])

        # Feature pyramid from last selected layer
        total_dim = sum(selected_dims)
        self.fpn = FeaturePyramid(total_dim, out_channels, num_levels)
        self.num_channels = [out_channels] * num_levels

    def forward(self, tensor_list: NestedTensor) -> Tuple[List[NestedTensor], None]:
        """Forward pass.

        Returns:
            features: List of NestedTensor at different scales.
            None: No positional encoding needed for Faster R-CNN.
        """
        xs = self.backbone.get_intermediate_layers(
            tensor_list.tensors, n=self.layers_to_use, reshape=True
        )
        # Apply layer norms
        xs = [ln(x).contiguous() for ln, x in zip(self.layer_norms, xs)]

        # Build feature pyramid from concatenated features
        concat_feat = torch.cat(xs, dim=1)

        # Create batched feature pyramid
        pyramid_feats = self.fpn([concat_feat])

        # Wrap each level as NestedTensor
        out: List[NestedTensor] = []
        for feat in pyramid_feats:
            m = tensor_list.mask
            if m is not None:
                mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            else:
                mask = torch.zeros(feat.shape[0], feat.shape[-2], feat.shape[-1], dtype=torch.bool, device=feat.device)
            out.append(NestedTensor(feat, mask))

        return out, None


class RPNHead(nn.Module):
    """Region Proposal Network head.

    Produces objectness scores and bounding box deltas for each anchor.
    Shared across all feature levels.
    """

    def __init__(self, in_channels: int, num_anchors: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        for layer in [self.conv, self.cls_logits, self.bbox_pred]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute RPN outputs for each feature level.

        Args:
            features: List of [B, C, H, W] feature maps.

        Returns:
            cls_logits: List of [B, num_anchors, H, W] objectness logits per level.
            bbox_preds: List of [B, num_anchors*4, H, W] box delta predictions per level.
        """
        cls_logits = []
        bbox_preds = []
        for feat in features:
            t = F.relu(self.conv(feat))
            cls_logits.append(self.cls_logits(t))
            bbox_preds.append(self.bbox_pred(t))
        return cls_logits, bbox_preds


class AnchorGenerator:
    """Generate anchors for multi-level feature maps.

    Anchors are in (cx, cy, w, h) format relative to the feature map grid,
    normalized to [0, 1] image coordinates.
    """

    def __init__(
        self,
        sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
        ratios: Tuple[float, ...] = (0.5, 1.0, 2.0),
        strides: Tuple[int, ...] = (4, 8, 16, 32, 64),
    ):
        self.sizes = sizes
        self.ratios = ratios
        self.strides = strides

    def num_anchors_per_level(self) -> int:
        return len(self.sizes) * len(self.ratios)

    def generate(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """Generate anchors for each feature level.

        Args:
            feature_maps: List of [B, C, H, W] tensors.

        Returns:
            List of [H*W*num_anchors, 4] anchor boxes in (cx, cy, w, h) format,
            normalized to [0, 1].
        """
        anchors_per_level = []
        for level, feat in enumerate(feature_maps):
            _, _, h, w = feat.shape
            stride = self.strides[level % len(self.strides)]

            # Grid centers
            shifts_y = torch.arange(0, h, dtype=torch.float32, device=feat.device) * stride
            shifts_x = torch.arange(0, w, dtype=torch.float32, device=feat.device) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_y = shift_y.reshape(-1)
            shift_x = shift_x.reshape(-1)

            # Generate base anchors for this level
            level_anchors = []
            for size in self.sizes:
                for ratio in self.ratios:
                    w_anchor = size * math.sqrt(ratio)
                    h_anchor = size / math.sqrt(ratio)
                    level_anchors.append([0.0, 0.0, w_anchor, h_anchor])
            base_anchors = torch.tensor(level_anchors, dtype=torch.float32, device=feat.device)

            # Place anchors at each grid location
            num_anchors = base_anchors.shape[0]
            num_positions = shift_x.shape[0]
            anchors = base_anchors.view(1, num_anchors, 4) + torch.stack(
                [shift_x, shift_y, shift_x, shift_y], dim=1
            ).view(num_positions, 1, 4)
            anchors = anchors.reshape(-1, 4)

            # Don't normalize here — keep in pixel coordinates for now
            anchors_per_level.append(anchors)

        return anchors_per_level


class RPN(nn.Module):
    """Region Proposal Network with anchor generation."""

    def __init__(
        self,
        head: RPNHead,
        anchor_generator: AnchorGenerator,
        pre_nms_top_n_train: int = 2000,
        post_nms_top_n_train: int = 1000,
        pre_nms_top_n_test: int = 1000,
        post_nms_top_n_test: int = 500,
        nms_thresh: float = 0.7,
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
    ):
        super().__init__()
        self.head = head
        self.anchor_generator = anchor_generator
        self.pre_nms_top_n_train = pre_nms_top_n_train
        self.post_nms_top_n_train = post_nms_top_n_train
        self.pre_nms_top_n_test = pre_nms_top_n_test
        self.post_nms_top_n_test = post_nms_top_n_test
        self.nms_thresh = nms_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def forward(self, features: List[torch.Tensor], image_sizes: List[Tuple[int, int]]):
        """Forward pass during training.

        Returns:
            proposals: List of [num_proposals, 4] boxes per image (xyxy, pixel coords).
            rpn_losses: Dict of RPN losses (empty during inference).
        """
        cls_logits, bbox_preds = self.head(features)
        anchors = self.anchor_generator.generate(features)

        return cls_logits, bbox_preds, anchors

    def compute_loss(
        self,
        cls_logits: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        anchors: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        image_sizes: List[Tuple[int, int]],
        cls_loss_coef: float = 1.0,
        reg_loss_coef: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute RPN training losses."""
        num_images = len(targets)
        device = cls_logits[0].device

        # Collect all anchors and predictions
        flat_anchors = []
        flat_cls = []
        flat_reg = []
        for level in range(len(anchors)):
            a = anchors[level]  # [N, 4]
            c = cls_logits[level].permute(0, 2, 3, 1).reshape(num_images, -1)  # [B, N]
            r = bbox_preds[level].permute(0, 2, 3, 1).reshape(num_images, -1, 4)  # [B, N, 4]
            flat_anchors.append(a)
            flat_cls.append(c)
            flat_reg.append(r)

        all_anchors = torch.cat(flat_anchors, dim=0)  # [total_anchors, 4]
        all_cls = torch.cat(flat_cls, dim=1)  # [B, total_anchors]
        all_reg = torch.cat(flat_reg, dim=1)  # [B, total_anchors, 4]

        num_anchors_per_image = all_anchors.shape[0]

        # For each image, sample anchors and compute loss
        total_cls_loss = torch.tensor(0.0, device=device)
        total_reg_loss = torch.tensor(0.0, device=device)
        num_pos = 0

        for i in range(num_images):
            if len(targets[i]["boxes"]) == 0:
                continue

            gt_boxes = box_cxcywh_to_xyxy(targets[i]["boxes"])  # [G, 4] xyxy
            img_h, img_w = image_sizes[i]
            # Rescale GT boxes to pixel coordinates
            gt_boxes[:, 0] *= img_w
            gt_boxes[:, 1] *= img_h
            gt_boxes[:, 2] *= img_w
            gt_boxes[:, 3] *= img_h

            # Match anchors to GT boxes
            matched_labels, matched_reg_targets = self._match_anchors(
                all_anchors, gt_boxes, img_w, img_h
            )

            # Sample positives and negatives
            pos_idx = torch.where(matched_labels == 1)[0]
            neg_idx = torch.where(matched_labels == 0)[0]

            num_pos_expected = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(len(pos_idx), num_pos_expected)
            num_neg = self.batch_size_per_image - num_pos

            if len(pos_idx) > num_pos:
                pos_idx = pos_idx[torch.randperm(len(pos_idx), device=device)[:num_pos]]
            if len(neg_idx) > num_neg:
                neg_idx = neg_idx[torch.randperm(len(neg_idx), device=device)[:num_neg]]

            sampled_idx = torch.cat([pos_idx, neg_idx])

            # Classification loss (binary cross entropy)
            cls_targets = matched_labels[sampled_idx].float()
            cls_preds = all_cls[i, sampled_idx]
            cls_loss = F.binary_cross_entropy_with_logits(cls_preds, cls_targets)

            total_cls_loss += cls_loss

            # Regression loss (smooth L1) for positives only
            if len(pos_idx) > 0:
                reg_preds = all_reg[i, pos_idx]
                reg_targets = matched_reg_targets[pos_idx]
                reg_loss = F.smooth_l1_loss(reg_preds, reg_targets, beta=1.0 / 9)
                total_reg_loss += reg_loss

        num_images_with_boxes = sum(1 for t in targets if len(t["boxes"]) > 0) or 1
        losses = {
            "loss_rpn_cls": total_cls_loss * cls_loss_coef / num_images_with_boxes,
            "loss_rpn_reg": total_reg_loss * reg_loss_coef / num_images_with_boxes,
        }
        return losses

    def _match_anchors(
        self, anchors: torch.Tensor, gt_boxes: torch.Tensor, img_w: int, img_h: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Match anchors to ground truth boxes.

        Args:
            anchors: [N, 4] in pixel coordinates (cxcywh).
            gt_boxes: [G, 4] in pixel coordinates (xyxy).

        Returns:
            matched_labels: [N] with values 1 (positive), 0 (negative), -1 (ignore).
            matched_reg_targets: [N, 4] box regression targets.
        """
        num_anchors = anchors.shape[0]
        device = anchors.device

        # Clamp anchors to image
        anchors_xyxy = box_cxcywh_to_xyxy(anchors)
        anchors_xyxy[:, 0].clamp_(min=0, max=img_w)
        anchors_xyxy[:, 1].clamp_(min=0, max=img_h)
        anchors_xyxy[:, 2].clamp_(min=0, max=img_w)
        anchors_xyxy[:, 3].clamp_(min=0, max=img_h)

        # Compute IoU between anchors and GT boxes
        ious = box_iou_matrix(anchors_xyxy, gt_boxes)  # [N, G]

        max_iou_per_anchor, matched_gt_idx = ious.max(dim=1)

        # Label assignment
        labels = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        labels[max_iou_per_anchor < self.bg_iou_thresh] = 0
        labels[max_iou_per_anchor >= self.fg_iou_thresh] = 1

        # Ensure each GT box gets at least one positive anchor
        if len(gt_boxes) > 0:
            max_iou_per_gt, _ = ious.max(dim=0)
            for gt_idx in range(len(gt_boxes)):
                best_anchor_idx = (ious[:, gt_idx] == max_iou_per_gt[gt_idx]).nonzero(as_tuple=True)[0]
                labels[best_anchor_idx] = 1

        # Regression targets
        matched_gt = gt_boxes[matched_gt_idx.clamp(min=0)]  # [N, 4]
        reg_targets = encode_box_deltas(anchors, matched_gt)

        return labels, reg_targets

    @torch.no_grad()
    def generate_proposals(
        self,
        cls_logits: List[torch.Tensor],
        bbox_preds: List[torch.Tensor],
        anchors: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        is_training: bool = True,
    ) -> List[torch.Tensor]:
        """Generate region proposals via NMS.

        Returns:
            List of [num_proposals, 4] boxes per image (xyxy, pixel coords).
        """
        pre_nms_top_n = self.pre_nms_top_n_train if is_training else self.pre_nms_top_n_test
        post_nms_top_n = self.post_nms_top_n_train if is_training else self.post_nms_top_n_test
        num_images = len(image_sizes)

        # Concatenate across all levels
        all_anchors_list = []
        all_cls_list = []
        all_reg_list = []
        for level in range(len(anchors)):
            c = cls_logits[level]  # [B, A, H, W]
            r = bbox_preds[level]  # [B, A*4, H, W]
            a = anchors[level]  # [H*W*A, 4]
            B, _, H, W = c.shape
            num_anchors_lvl = c.shape[1]
            c = c.permute(0, 2, 3, 1).reshape(B, -1)  # [B, H*W*A]
            r = r.permute(0, 2, 3, 1).reshape(B, -1, 4)  # [B, H*W*A, 4]
            all_cls_list.append(c)
            all_reg_list.append(r)
            all_anchors_list.append(a)

        all_cls = torch.cat(all_cls_list, dim=1)  # [B, total_anchors]
        all_reg = torch.cat(all_reg_list, dim=1)  # [B, total_anchors, 4]
        all_anchors = torch.cat(all_anchors_list, dim=0)  # [total_anchors, 4]

        proposals = []
        for i in range(num_images):
            scores = all_cls[i].sigmoid()
            deltas = all_reg[i]
            anchors_i = all_anchors

            # Apply deltas to anchors
            boxes = apply_box_deltas(anchors_i, deltas)  # xyxy

            # Clamp to image
            img_h, img_w = image_sizes[i]
            boxes[:, 0].clamp_(min=0, max=img_w)
            boxes[:, 1].clamp_(min=0, max=img_h)
            boxes[:, 2].clamp_(min=0, max=img_w)
            boxes[:, 3].clamp_(min=0, max=img_h)

            # Remove small boxes
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            keep = (w > 0) & (h > 0)
            boxes = boxes[keep]
            scores = scores[keep]

            # Pre-NMS top-K
            if pre_nms_top_n > 0 and len(scores) > pre_nms_top_n:
                scores, idx = scores.topk(pre_nms_top_n)
                boxes = boxes[idx]

            # NMS
            keep = nms(boxes, scores, self.nms_thresh)
            boxes = boxes[keep]
            scores = scores[keep]

            # Post-NMS top-K
            if post_nms_top_n > 0 and len(scores) > post_nms_top_n:
                scores, idx = scores.topk(post_nms_top_n)
                boxes = boxes[idx]

            proposals.append(boxes)

        return proposals


class RoIHeads(nn.Module):
    """RoI-based detection head for Faster R-CNN."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        fc_dim: int = 1024,
        num_fc: int = 2,
        roi_output_size: int = 7,
        roi_sampling_ratio: int = 2,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.5,
        batch_size_per_image: int = 512,
        positive_fraction: float = 0.25,
        nms_thresh: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.roi_output_size = roi_output_size
        self.roi_sampling_ratio = roi_sampling_ratio
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.nms_thresh = nms_thresh

        # Box head: FC layers
        fc_layers = []
        input_dim = in_channels * roi_output_size * roi_output_size
        for _ in range(num_fc):
            fc_layers.append(nn.Linear(input_dim, fc_dim))
            fc_layers.append(nn.ReLU(inplace=True))
            input_dim = fc_dim
        self.fc = nn.Sequential(*fc_layers)

        # Classifier and regressor
        self.cls_score = nn.Linear(fc_dim, num_classes)
        self.bbox_pred = nn.Linear(fc_dim, num_classes * 4)

        # Initialize
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.01)
                nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, features: List[torch.Tensor], proposals: List[torch.Tensor]):
        """Forward pass during inference.

        Args:
            features: List of [B, C, H_i, W_i] feature maps.
            proposals: List of [num_proposals, 4] boxes per image (xyxy, pixel coords).

        Returns:
            cls_logits: [total_proposals, num_classes]
            bbox_preds: [total_proposals, num_classes*4]
        """
        # Use the highest-resolution feature map for RoI Align
        feat = features[0]  # [B, C, H, W]

        # Assign each proposal to an image index
        batch_indices = []
        roi_boxes = []
        for i, props in enumerate(proposals):
            if len(props) == 0:
                continue
            batch_indices.append(torch.full((len(props),), i, dtype=torch.float32, device=props.device))
            roi_boxes.append(props)

        if len(roi_boxes) == 0:
            return (
                torch.zeros(0, self.num_classes, device=feat.device),
                torch.zeros(0, self.num_classes * 4, device=feat.device),
            )

        roi_batch_idx = torch.cat(batch_indices)
        roi_boxes_cat = torch.cat(roi_boxes)
        rois = torch.cat([roi_batch_idx[:, None], roi_boxes_cat], dim=1)

        # RoI Align
        roi_features = roi_align(
            feat,
            rois,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0,  # Features are already at correct scale
            sampling_ratio=self.roi_sampling_ratio,
            aligned=True,
        )

        # Box head
        x = roi_features.flatten(1)
        x = self.fc(x)
        cls_logits = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)

        return cls_logits, bbox_preds

    def compute_loss(
        self,
        features: List[torch.Tensor],
        proposals: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        image_sizes: List[Tuple[int, int]],
        cls_loss_coef: float = 1.0,
        reg_loss_coef: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute RoI head losses during training."""
        device = features[0].device

        # Sample positive/negative proposals
        sampled_proposals, gt_labels, gt_reg_targets = self._sample_proposals(
            proposals, targets, image_sizes
        )

        if sum(len(p) for p in sampled_proposals) == 0:
            return {
                "loss_box_cls": torch.tensor(0.0, device=device),
                "loss_box_reg": torch.tensor(0.0, device=device),
            }

        # Forward through box head
        cls_logits, bbox_preds = self.forward(features, sampled_proposals)

        # Classification loss (cross-entropy)
        gt_labels_cat = torch.cat(gt_labels)
        cls_loss = F.cross_entropy(cls_logits, gt_labels_cat)

        # Regression loss (smooth L1, only for positives)
        gt_reg_targets_cat = torch.cat(gt_reg_targets)
        pos_mask = gt_labels_cat < self.num_classes  # Exclude background class
        if pos_mask.any():
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            bbox_preds_pos = bbox_preds[pos_indices]
            gt_reg_pos = gt_reg_targets_cat[pos_indices]

            # Select predictions for the correct class
            # bbox_preds: [N, num_classes*4]
            # Reshape to [N, num_classes, 4]
            bbox_preds_pos = bbox_preds_pos.view(-1, self.num_classes, 4)
            gt_classes = gt_labels_cat[pos_indices]
            bbox_preds_pos = bbox_preds_pos[torch.arange(len(pos_indices), device=device), gt_classes]

            reg_loss = F.smooth_l1_loss(bbox_preds_pos, gt_reg_pos, beta=1.0)
        else:
            reg_loss = torch.tensor(0.0, device=device)

        losses = {
            "loss_box_cls": cls_loss * cls_loss_coef,
            "loss_box_reg": reg_loss * reg_loss_coef,
        }
        return losses

    def _sample_proposals(
        self,
        proposals: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        image_sizes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Sample positive/negative proposals for training.

        Returns:
            sampled_proposals: List of [num_sampled, 4] boxes per image.
            gt_labels: List of [num_sampled] class labels.
            gt_reg_targets: List of [num_sampled, 4] box regression targets.
        """
        device = proposals[0].device
        sampled_props = []
        sampled_labels = []
        sampled_reg_targets = []

        for i, (props, target) in enumerate(zip(proposals, targets)):
            if len(props) == 0:
                sampled_props.append(props)
                sampled_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                sampled_reg_targets.append(torch.zeros(0, 4, device=device))
                continue

            if len(target["boxes"]) == 0:
                # No GT boxes: all proposals are background
                num_sample = min(self.batch_size_per_image, len(props))
                idx = torch.randperm(len(props), device=device)[:num_sample]
                sampled_props.append(props[idx])
                sampled_labels.append(torch.full((num_sample,), self.num_classes, dtype=torch.long, device=device))
                sampled_reg_targets.append(torch.zeros(num_sample, 4, device=device))
                continue

            img_h, img_w = image_sizes[i]
            gt_boxes = box_cxcywh_to_xyxy(target["boxes"])
            gt_boxes[:, 0] *= img_w
            gt_boxes[:, 1] *= img_h
            gt_boxes[:, 2] *= img_w
            gt_boxes[:, 3] *= img_h

            ious = box_iou_matrix(props, gt_boxes)  # [P, G]
            max_iou_per_prop, matched_gt_idx = ious.max(dim=1)

            # Positive: IoU >= fg_thresh
            pos_idx = torch.where(max_iou_per_prop >= self.fg_iou_thresh)[0]
            # Negative: IoU < bg_thresh
            neg_idx = torch.where(max_iou_per_prop < self.bg_iou_thresh)[0]

            num_pos_expected = int(self.batch_size_per_image * self.positive_fraction)
            num_pos = min(len(pos_idx), num_pos_expected)
            num_neg = self.batch_size_per_image - num_pos

            if len(pos_idx) > num_pos:
                pos_idx = pos_idx[torch.randperm(len(pos_idx), device=device)[:num_pos]]
            if len(neg_idx) > num_neg:
                neg_idx = neg_idx[torch.randperm(len(neg_idx), device=device)[:num_neg]]

            sampled_idx = torch.cat([pos_idx, neg_idx])
            sampled_props.append(props[sampled_idx])

            # Build labels
            labels = torch.full((len(sampled_idx),), self.num_classes, dtype=torch.long, device=device)
            labels[:len(pos_idx)] = target["labels"][matched_gt_idx[pos_idx]]
            sampled_labels.append(labels)

            # Build regression targets for positives
            reg_targets = torch.zeros(len(sampled_idx), 4, device=device)
            if len(pos_idx) > 0:
                matched_gt = gt_boxes[matched_gt_idx[pos_idx]]
                reg_targets[:len(pos_idx)] = encode_box_deltas(
                    box_xyxy_to_cxcywh(props[pos_idx]),
                    box_xyxy_to_cxcywh(matched_gt),
                )
            sampled_reg_targets.append(reg_targets)

        return sampled_props, sampled_labels, sampled_reg_targets

    @torch.no_grad()
    def postprocess(
        self,
        cls_logits: torch.Tensor,
        bbox_preds: torch.Tensor,
        proposals: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        top_k: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process predictions for evaluation.

        Returns:
            List of dicts with 'scores', 'labels', 'boxes' (xyxy, pixel coords).
        """
        num_images = len(proposals)
        num_classes = self.num_classes

        # Split results per image
        results = []
        start = 0
        for i in range(num_images):
            num_props = len(proposals[i])
            if num_props == 0:
                results.append({
                    "scores": torch.zeros(0, device=cls_logits.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=cls_logits.device),
                    "boxes": torch.zeros(0, 4, device=cls_logits.device),
                })
                continue

            end = start + num_props
            cls_i = cls_logits[start:end]
            reg_i = bbox_preds[start:end]
            props_i = proposals[i]
            start = end

            img_h, img_w = image_sizes[i]

            # Apply box deltas
            reg_i = reg_i.view(-1, num_classes, 4)
            scores = cls_i.softmax(dim=1)  # [P, num_classes]

            # Per-class NMS
            all_scores = []
            all_labels = []
            all_boxes = []
            for cls_idx in range(num_classes):
                cls_scores = scores[:, cls_idx]
                cls_reg = reg_i[:, cls_idx]

                # Apply delta to proposals
                boxes = apply_box_deltas(box_xyxy_to_cxcywh(props_i), cls_reg)
                boxes = box_cxcywh_to_xyxy(boxes)

                # Clamp
                boxes[:, 0].clamp_(min=0, max=img_w)
                boxes[:, 1].clamp_(min=0, max=img_h)
                boxes[:, 2].clamp_(min=0, max=img_w)
                boxes[:, 3].clamp_(min=0, max=img_h)

                # Filter by score
                keep = cls_scores > score_thresh
                if not keep.any():
                    continue

                boxes_cls = boxes[keep]
                scores_cls = cls_scores[keep]

                # NMS
                keep_nms = nms(boxes_cls, scores_cls, nms_thresh)
                boxes_cls = boxes_cls[keep_nms]
                scores_cls = scores_cls[keep_nms]

                all_scores.append(scores_cls)
                all_labels.append(torch.full_like(scores_cls.long(), cls_idx))
                all_boxes.append(boxes_cls)

            if len(all_scores) == 0:
                results.append({
                    "scores": torch.zeros(0, device=cls_logits.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=cls_logits.device),
                    "boxes": torch.zeros(0, 4, device=cls_logits.device),
                })
                continue

            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)
            all_boxes = torch.cat(all_boxes)

            # Top-K
            if len(all_scores) > top_k:
                all_scores, idx = all_scores.topk(top_k)
                all_labels = all_labels[idx]
                all_boxes = all_boxes[idx]

            results.append({
                "scores": all_scores,
                "labels": all_labels,
                "boxes": all_boxes,
            })

        return results


class FasterRCNN(nn.Module):
    """Faster R-CNN model with DINOv3 backbone."""

    def __init__(
        self,
        backbone: MultiLevelBackbone,
        rpn: RPN,
        roi_heads: RoIHeads,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.num_classes = num_classes

    def forward(self, samples: NestedTensor, targets=None):
        """Forward pass.

        During training (targets is not None):
            Returns a dict of losses.

        During inference (targets is None):
            Returns a dict with 'pred_logits', 'pred_boxes'.

        During eval (not self.training, targets is not None):
            Returns a dict with both losses and predictions.
        """
        if not isinstance(samples, NestedTensor):
            from ..util.misc import nested_tensor_from_tensor_list

            samples = nested_tensor_from_tensor_list(samples)

        # Backbone
        features, _ = self.backbone(samples)

        # Get raw feature tensors for RPN
        feat_tensors = [f.tensors for f in features]

        # Image sizes
        image_sizes = []
        for i in range(samples.tensors.shape[0]):
            h = samples.tensors.shape[2]
            w = samples.tensors.shape[3]
            image_sizes.append((h, w))

        # RPN
        cls_logits, bbox_preds, anchors = self.rpn(feat_tensors, image_sizes)

        # Generate proposals
        proposals = self.rpn.generate_proposals(
            cls_logits, bbox_preds, anchors, image_sizes, is_training=self.training
        )

        if self.training and targets is not None:
            # Compute losses only during training
            losses = {}
            losses.update(
                self.rpn.compute_loss(
                    cls_logits, bbox_preds, anchors, targets, image_sizes,
                    cls_loss_coef=1.0, reg_loss_coef=1.0,
                )
            )
            losses.update(
                self.roi_heads.compute_loss(
                    feat_tensors, proposals, targets, image_sizes,
                    cls_loss_coef=1.0, reg_loss_coef=1.0,
                )
            )
            return losses

        # Inference: get predictions from RoI heads
        cls_logits, bbox_preds = self.roi_heads(feat_tensors, proposals)
        results = self.roi_heads.postprocess(
            cls_logits, bbox_preds, proposals, image_sizes,
            score_thresh=0.05, nms_thresh=self.roi_heads.nms_thresh,
            top_k=100,
        )

        # Convert to DETR-compatible format
        max_dets = max(len(r["scores"]) for r in results) if results else 0
        if max_dets == 0:
            pred_logits = torch.zeros(len(results), 0, self.num_classes, device=samples.tensors.device)
            pred_boxes = torch.zeros(len(results), 0, 4, device=samples.tensors.device)
        else:
            pred_logits = torch.zeros(len(results), max_dets, self.num_classes, device=samples.tensors.device)
            pred_boxes = torch.zeros(len(results), max_dets, 4, device=samples.tensors.device)
            for i, r in enumerate(results):
                n = len(r["scores"])
                pred_logits[i, :n, r["labels"]] = r["scores"][:, None]
                img_h, img_w = image_sizes[i]
                pred_boxes[i, :n, 0] = r["boxes"][:, 0] / img_w
                pred_boxes[i, :n, 1] = r["boxes"][:, 1] / img_h
                pred_boxes[i, :n, 2] = r["boxes"][:, 2] / img_w
                pred_boxes[i, :n, 3] = r["boxes"][:, 3] / img_h

        out = {"pred_logits": pred_logits, "pred_boxes": box_xyxy_to_cxcywh(pred_boxes)}

        # Also compute losses if targets given (eval mode with annotations)
        if targets is not None:
            losses = {}
            losses.update(
                self.rpn.compute_loss(
                    cls_logits, bbox_preds, anchors, targets, image_sizes,
                    cls_loss_coef=1.0, reg_loss_coef=1.0,
                )
            )
            losses.update(
                self.roi_heads.compute_loss(
                    feat_tensors, proposals, targets, image_sizes,
                    cls_loss_coef=1.0, reg_loss_coef=1.0,
                )
            )
            out["losses"] = losses

        return out


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def box_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes.

    Args:
        boxes1: [N, 4] xyxy format.
        boxes2: [M, 4] xyxy format.

    Returns:
        iou: [N, M] pairwise IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def encode_box_deltas(anchors_cxcywh: torch.Tensor, gt_boxes_cxcywh: torch.Tensor) -> torch.Tensor:
    """Encode ground truth boxes as deltas relative to anchors.

    Args:
        anchors_cxcywh: [N, 4] anchor boxes.
        gt_boxes_cxcywh: [N, 4] ground truth boxes.

    Returns:
        deltas: [N, 4] encoded box regression targets.
    """
    ax, ay, aw, ah = anchors_cxcywh.unbind(-1)
    gx, gy, gw, gh = gt_boxes_cxcywh.unbind(-1)

    dx = (gx - ax) / (aw + 1e-6)
    dy = (gy - ay) / (ah + 1e-6)
    dw = torch.log(gw / (aw + 1e-6))
    dh = torch.log(gh / (ah + 1e-6))

    return torch.stack([dx, dy, dw, dh], dim=-1)


def apply_box_deltas(anchors_cxcywh: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    """Apply box deltas to anchors to get predicted boxes.

    Args:
        anchors_cxcywh: [N, 4] anchor boxes.
        deltas: [N, 4] predicted deltas.

    Returns:
        boxes: [N, 4] predicted boxes in xyxy format.
    """
    ax, ay, aw, ah = anchors_cxcywh.unbind(-1)
    dx, dy, dw, dh = deltas.unbind(-1)

    gx = ax + dx * aw
    gy = ay + dy * ah
    gw = aw * dw.exp()
    gh = ah * dh.exp()

    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    return torch.stack([x1, y1, x2, y2], dim=-1)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """Non-maximum suppression.

    Args:
        boxes: [N, 4] in xyxy format.
        scores: [N] confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        keep: indices of kept boxes.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    # Sort by score descending
    _, order = scores.sort(descending=True)
    boxes = boxes[order]

    keep = []
    remaining = torch.arange(len(boxes), device=boxes.device)
    while len(remaining) > 0:
        # Keep the current highest-scoring box
        current = remaining[0]
        keep.append(current)

        if len(remaining) == 1:
            break

        # Compute IoU between current box and remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[remaining[1:]]

        x1 = torch.max(current_box[0], remaining_boxes[:, 0])
        y1 = torch.max(current_box[1], remaining_boxes[:, 1])
        x2 = torch.min(current_box[2], remaining_boxes[:, 2])
        y2 = torch.min(current_box[3], remaining_boxes[:, 3])

        inter_w = (x2 - x1).clamp(min=0)
        inter_h = (y2 - y1).clamp(min=0)
        inter = inter_w * inter_h

        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        union = area_current + area_remaining - inter
        iou = inter / union.clamp(min=1e-6)

        # Keep boxes with IoU <= threshold
        mask = iou <= iou_threshold
        remaining = remaining[1:][mask]

    keep = torch.tensor(keep, device=boxes.device)
    return order[keep]


def build_faster_rcnn(backbone_model, config) -> FasterRCNN:
    """Build Faster R-CNN model from config.

    Args:
        backbone_model: ViT backbone model (loaded with pretrained weights).
        config: DetectionTrainConfig or FasterRCNNConfig.
    """
    num_classes = config.num_classes

    # Determine backbone params
    embed_dim = backbone_model.embed_dim
    patch_size = backbone_model.patch_size
    n_blocks = backbone_model.n_blocks

    if hasattr(config, "head"):
        layers_to_use = config.head.layers_to_use
        use_layernorm = config.head.backbone_use_layernorm
        faster_rcnn_cfg = config.faster_rcnn
    else:
        layers_to_use = config.layers_to_use
        use_layernorm = getattr(config, "backbone_use_layernorm", True)
        faster_rcnn_cfg = config

    if layers_to_use is None:
        layers_to_use = [m * n_blocks // 4 - 1 for m in range(1, 5)]

    # Multi-level backbone with feature pyramid
    backbone = MultiLevelBackbone(
        backbone_model=backbone_model,
        layers_to_use=layers_to_use,
        embed_dim=embed_dim,
        patch_size=patch_size,
        out_channels=faster_rcnn_cfg.feature_hidden_dim,
        num_levels=faster_rcnn_cfg.num_feature_levels,
        use_layernorm=use_layernorm,
    )

    # RPN
    rpn_cfg = faster_rcnn_cfg.rpn
    num_anchors = len(rpn_cfg.anchor_sizes) * len(rpn_cfg.anchor_ratios)
    rpn_head = RPNHead(faster_rcnn_cfg.feature_hidden_dim, num_anchors)
    anchor_generator = AnchorGenerator(
        sizes=rpn_cfg.anchor_sizes,
        ratios=rpn_cfg.anchor_ratios,
        strides=rpn_cfg.anchor_strides,
    )
    rpn = RPN(
        head=rpn_head,
        anchor_generator=anchor_generator,
        pre_nms_top_n_train=rpn_cfg.rpn_pre_nms_top_n_train,
        post_nms_top_n_train=rpn_cfg.rpn_post_nms_top_n_train,
        pre_nms_top_n_test=rpn_cfg.rpn_pre_nms_top_n_test,
        post_nms_top_n_test=rpn_cfg.rpn_post_nms_top_n_test,
        nms_thresh=rpn_cfg.rpn_nms_thresh,
        fg_iou_thresh=rpn_cfg.rpn_fg_iou_thresh,
        bg_iou_thresh=rpn_cfg.rpn_bg_iou_thresh,
        batch_size_per_image=rpn_cfg.rpn_batch_size_per_image,
        positive_fraction=rpn_cfg.rpn_positive_fraction,
    )

    # RoI heads
    roi_cfg = faster_rcnn_cfg.roi_head
    roi_heads = RoIHeads(
        in_channels=faster_rcnn_cfg.feature_hidden_dim,
        num_classes=num_classes,
        fc_dim=roi_cfg.box_head_fc_dim,
        num_fc=roi_cfg.box_head_num_fc,
        roi_output_size=roi_cfg.box_roi_output_size,
        roi_sampling_ratio=roi_cfg.box_roi_sampling_ratio,
        fg_iou_thresh=roi_cfg.box_fg_iou_thresh,
        bg_iou_thresh=roi_cfg.box_bg_iou_thresh,
        batch_size_per_image=roi_cfg.box_batch_size_per_image,
        positive_fraction=roi_cfg.box_positive_fraction,
        nms_thresh=roi_cfg.box_nms_thresh,
    )

    return FasterRCNN(backbone, rpn, roi_heads, num_classes)
