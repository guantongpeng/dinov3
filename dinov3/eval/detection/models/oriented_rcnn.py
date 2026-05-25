# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Oriented R-CNN for rotated object detection with DINOv3 backbone.

Architecture:
    Backbone (ViT) -> Feature Pyramid -> RPN -> proposals (axis-aligned)
                                        -> Rotated RoI Align + Box Head -> rotated detections (cx, cy, w, h, theta)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_align

from ..util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from ..util.misc import NestedTensor
from .faster_rcnn import (
    AnchorGenerator,
    FeaturePyramid,
    MultiLevelBackbone,
    RPN,
    RPNHead,
    box_iou_matrix,
)
from .utils import LayerNorm2D


# ---------------------------------------------------------------------------
# Rotated box utilities
# ---------------------------------------------------------------------------


def obb_xywht_to_xyxy(obb: torch.Tensor) -> torch.Tensor:
    """Convert oriented bounding box (cx, cy, w, h, theta) to 4-corner xyxyxyxy format.

    Args:
        obb: [..., 5] (cx, cy, w, h, theta) in radians.

    Returns:
        corners: [..., 8] (x0, y0, x1, y1, x2, y2, x3, y3).
    """
    cx, cy, w, h, theta = obb.unbind(-1)
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    # Half-dimensions
    hw = w * 0.5
    hh = h * 0.5

    # Corner offsets in CCW order: top-left, top-right, bottom-right, bottom-left
    dx = torch.stack([-hw, hw, hw, -hw], dim=-1)
    dy = torch.stack([-hh, -hh, hh, hh], dim=-1)

    # Rotate
    rx = dx * cos_t.unsqueeze(-1) - dy * sin_t.unsqueeze(-1)
    ry = dx * sin_t.unsqueeze(-1) + dy * cos_t.unsqueeze(-1)

    # Translate
    corners_x = rx + cx.unsqueeze(-1)
    corners_y = ry + cy.unsqueeze(-1)

    return torch.stack([corners_x, corners_y], dim=-1).reshape(*obb.shape[:-1], 8)


def obb_xyxy_to_xywht(corners: torch.Tensor) -> torch.Tensor:
    """Convert 4-corner xyxyxyxy to (cx, cy, w, h, theta).

    Uses the long-edge definition (le90): theta is the angle of the longer edge.
    theta in [-pi/2, pi/2).

    Args:
        corners: [..., 8] (x0, y0, x1, y1, x2, y2, x3, y3).

    Returns:
        obb: [..., 5] (cx, cy, w, h, theta) in radians.
    """
    shape = corners.shape[:-1]
    corners = corners.reshape(-1, 4, 2)

    # Center
    cx = corners[:, :, 0].mean(dim=1)
    cy = corners[:, :, 1].mean(dim=1)

    # Edges: vector from point 0->1 and 1->2
    v1 = corners[:, 1] - corners[:, 0]
    v2 = corners[:, 2] - corners[:, 1]

    len1 = torch.norm(v1, dim=1)
    len2 = torch.norm(v2, dim=1)

    # Long edge determines theta
    long_is_01 = len1 >= len2
    theta = torch.where(
        long_is_01,
        torch.atan2(v1[:, 1], v1[:, 0]),
        torch.atan2(v2[:, 1], v2[:, 0]),
    )

    w = torch.where(long_is_01, len1, len2)
    h = torch.where(long_is_01, len2, len1)

    obb = torch.stack([cx, cy, w, h, theta], dim=-1)
    return obb.reshape(*shape, 5)


def encode_oriented_deltas(
    proposals_cxcywh: torch.Tensor, gt_obb: torch.Tensor
) -> torch.Tensor:
    """Encode oriented GT boxes as deltas relative to axis-aligned proposals.

    Args:
        proposals_cxcywh: [N, 4] axis-aligned proposal boxes (cx, cy, w, h).
        gt_obb: [N, 5] oriented GT boxes (cx, cy, w, h, theta) in radians.

    Returns:
        deltas: [N, 5] encoded regression targets.
    """
    px, py, pw, ph = proposals_cxcywh.unbind(-1)
    gx, gy, gw, gh, gt = gt_obb.unbind(-1)

    dx = (gx - px) / (pw + 1e-6)
    dy = (gy - py) / (ph + 1e-6)
    dw = torch.log(gw / (pw + 1e-6))
    dh = torch.log(gh / (ph + 1e-6))
    dt = gt  # theta is predicted directly (not relative), normalized to [-1, 1] via pi

    return torch.stack([dx, dy, dw, dh, dt], dim=-1)


def apply_oriented_deltas(
    proposals_cxcywh: torch.Tensor, deltas: torch.Tensor
) -> torch.Tensor:
    """Apply oriented deltas to axis-aligned proposals to get oriented boxes.

    Args:
        proposals_cxcywh: [N, 4] axis-aligned proposal boxes.
        deltas: [N, 5] predicted deltas.

    Returns:
        obb: [N, 5] oriented boxes (cx, cy, w, h, theta) in radians.
    """
    px, py, pw, ph = proposals_cxcywh.unbind(-1)
    dx, dy, dw, dh, dt = deltas.unbind(-1)

    gx = px + dx * pw
    gy = py + dy * ph
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gt = dt  # theta prediction

    return torch.stack([gx, gy, gw, gh, gt], dim=-1)


def rotated_iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of oriented bounding boxes.

    Args:
        boxes1: [N, 5] (cx, cy, w, h, theta) in radians.
        boxes2: [M, 5] (cx, cy, w, h, theta) in radians.

    Returns:
        iou: [N, M] pairwise IoU matrix.
    """
    N, M = boxes1.shape[0], boxes2.shape[0]
    device = boxes1.device

    if N == 0 or M == 0:
        return torch.zeros(N, M, device=device)

    corners1 = obb_xywht_to_xyxy(boxes1)  # [N, 8]
    corners2 = obb_xywht_to_xyxy(boxes2)  # [M, 8]

    area1 = boxes1[:, 2] * boxes1[:, 3]  # [N]
    area2 = boxes2[:, 2] * boxes2[:, 3]  # [M]

    # Vectorized rotated intersection over all N x M pairs
    inter = torch.zeros(N, M, device=device)
    for i in range(N):
        inter[i] = _batch_rotated_intersection(corners1[i], corners2)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou


def _batch_rotated_intersection(
    corners_a: torch.Tensor, corners_b: torch.Tensor
) -> torch.Tensor:
    """Compute intersection area between one box and M other boxes.

    Args:
        corners_a: [8] (x0,y0,x1,y1,x2,y2,x3,y3) for single box.
        corners_b: [M, 8] for M boxes.

    Returns:
        inter: [M] intersection areas.
    """
    M = corners_b.shape[0]
    device = corners_a.device
    inter = torch.zeros(M, device=device)

    poly_a = corners_a.view(4, 2)  # [4, 2]
    for j in range(M):
        poly_b = corners_b[j].view(4, 2)
        inter[j] = _convex_polygon_intersection_area(poly_a, poly_b)

    return inter


def _triangle_area(p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
    """Signed triangle area (cross product)."""
    return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))


def _convex_polygon_intersection_area(
    poly1: torch.Tensor, poly2: torch.Tensor
) -> torch.Tensor:
    """Compute intersection area of two convex polygons using Sutherland-Hodgman clipping.

    Args:
        poly1: [4, 2]
        poly2: [4, 2]

    Returns:
        area of intersection.
    """
    # Clip poly1 against each edge of poly2
    output = poly1.clone()

    for i in range(4):
        if output.shape[0] < 3:
            return torch.tensor(0.0, device=poly1.device)

        p1 = poly2[i]
        p2 = poly2[(i + 1) % 4]
        edge = p2 - p1

        input_list = output
        output = torch.empty(0, 2, device=poly1.device)

        n_input = input_list.shape[0]
        for j in range(n_input):
            current = input_list[j]
            prev = input_list[(j + n_input - 1) % n_input]

            # Check if points are inside (to the left of) the edge
            d_current = _triangle_area(p1, p2, current)
            d_prev = _triangle_area(p1, p2, prev)

            # Current is inside
            if d_current >= 0:
                if d_prev < 0:
                    # Edge entering: add intersection
                    intersect = _line_intersection(prev, current, p1, p2)
                    output = torch.cat([output, intersect.unsqueeze(0)], dim=0)
                output = torch.cat([output, current.unsqueeze(0)], dim=0)
            elif d_prev >= 0:
                # Edge leaving: add intersection
                intersect = _line_intersection(prev, current, p1, p2)
                output = torch.cat([output, intersect.unsqueeze(0)], dim=0)

    if output.shape[0] < 3:
        return torch.tensor(0.0, device=poly1.device)

    return _polygon_area(output)


def _line_intersection(
    p1: torch.Tensor, p2: torch.Tensor, p3: torch.Tensor, p4: torch.Tensor
) -> torch.Tensor:
    """Find intersection point of lines p1-p2 and p3-p4."""
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    x4, y4 = p4[0], p4[1]

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    denom = denom + 1e-10 if denom.abs() < 1e-10 else denom

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return torch.stack([x, y])


def _polygon_area(poly: torch.Tensor) -> torch.Tensor:
    """Compute area of polygon using shoelace formula.

    Args:
        poly: [K, 2] vertices.

    Returns:
        area.
    """
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * torch.abs((x[:-1] * y[1:] - x[1:] * y[:-1]).sum() +
                           (x[-1] * y[0] - x[0] * y[-1]))


def rotated_nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float
) -> torch.Tensor:
    """Non-maximum suppression for oriented bounding boxes.

    Args:
        boxes: [N, 5] (cx, cy, w, h, theta) in radians.
        scores: [N] confidence scores.
        iou_threshold: IoU threshold for suppression.

    Returns:
        keep: indices of kept boxes.
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    _, order = scores.sort(descending=True)
    boxes = boxes[order]

    # Corners for IoU computation
    corners = obb_xywht_to_xyxy(boxes)  # [N, 8]

    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)

    for idx in range(len(boxes)):
        if suppressed[idx]:
            continue

        current = idx
        keep.append(current)

        if idx == len(boxes) - 1:
            break

        # Compute IoU between current and remaining unsuppressed boxes
        remaining = torch.arange(idx + 1, len(boxes), device=boxes.device)
        remaining = remaining[~suppressed[idx + 1:]]

        if len(remaining) == 0:
            break

        # Compute pairwise rotated IoU
        for r_idx in remaining:
            if suppressed[r_idx]:
                continue
            inter = _convex_polygon_intersection_area(
                corners[current].view(4, 2),
                corners[r_idx].view(4, 2),
            )
            area_c = boxes[current, 2] * boxes[current, 3]
            area_r = boxes[r_idx, 2] * boxes[r_idx, 3]
            union = area_c + area_r - inter
            iou = inter / union.clamp(min=1e-6)
            if iou > iou_threshold:
                suppressed[r_idx] = True

    keep = torch.tensor(keep, device=boxes.device)
    return order[keep]


# ---------------------------------------------------------------------------
# Oriented RoI Heads
# ---------------------------------------------------------------------------


class OrientedRoIHeads(nn.Module):
    """RoI-based detection head for Oriented R-CNN.

    Predicts oriented bounding boxes (cx, cy, w, h, theta) instead of
    axis-aligned boxes. Uses the same RoI Align as Faster R-CNN but
    the regression head outputs 5 parameters per class.
    """

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
        nms_thresh: float = 0.1,
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

        # Classifier and regressor (5 params: cx, cy, w, h, theta)
        self.cls_score = nn.Linear(fc_dim, num_classes)
        self.bbox_pred = nn.Linear(fc_dim, num_classes * 5)

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
            bbox_preds: [total_proposals, num_classes * 5] (cx, cy, w, h, theta)
        """
        feat = features[0]  # [B, C, H, W]

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
                torch.zeros(0, self.num_classes * 5, device=feat.device),
            )

        roi_batch_idx = torch.cat(batch_indices)
        roi_boxes_cat = torch.cat(roi_boxes)
        rois = torch.cat([roi_batch_idx[:, None], roi_boxes_cat], dim=1)

        # RoI Align
        roi_features = roi_align(
            feat,
            rois,
            output_size=(self.roi_output_size, self.roi_output_size),
            spatial_scale=1.0,
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

        sampled_proposals, gt_labels, gt_reg_targets = self._sample_proposals(
            proposals, targets, image_sizes
        )

        if sum(len(p) for p in sampled_proposals) == 0:
            return {
                "loss_box_cls": torch.tensor(0.0, device=device),
                "loss_box_reg": torch.tensor(0.0, device=device),
            }

        cls_logits, bbox_preds = self.forward(features, sampled_proposals)

        gt_labels_cat = torch.cat(gt_labels)
        cls_loss = F.cross_entropy(cls_logits, gt_labels_cat)

        gt_reg_targets_cat = torch.cat(gt_reg_targets)
        pos_mask = gt_labels_cat < self.num_classes
        if pos_mask.any():
            pos_indices = pos_mask.nonzero(as_tuple=True)[0]
            bbox_preds_pos = bbox_preds[pos_indices]
            gt_reg_pos = gt_reg_targets_cat[pos_indices]

            bbox_preds_pos = bbox_preds_pos.view(-1, self.num_classes, 5)
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

        Uses axis-aligned IoU for matching proposals to GT horizontal boxes,
        then encodes oriented deltas for positive matches.
        """
        device = proposals[0].device
        sampled_props = []
        sampled_labels = []
        sampled_reg_targets = []

        for i, (props, target) in enumerate(zip(proposals, targets)):
            if len(props) == 0:
                sampled_props.append(props)
                sampled_labels.append(torch.zeros(0, dtype=torch.long, device=device))
                sampled_reg_targets.append(torch.zeros(0, 5, device=device))
                continue

            img_h, img_w = image_sizes[i]

            if len(target["boxes"]) == 0:
                num_sample = min(self.batch_size_per_image, len(props))
                idx = torch.randperm(len(props), device=device)[:num_sample]
                sampled_props.append(props[idx])
                sampled_labels.append(torch.full((num_sample,), self.num_classes, dtype=torch.long, device=device))
                sampled_reg_targets.append(torch.zeros(num_sample, 5, device=device))
                continue

            # Use horizontal GT boxes for proposal matching (axis-aligned IoU)
            if "boxes_obb" in target and target["boxes_obb"].numel() > 0:
                # If oriented boxes available, use axis-aligned circumscribed boxes for matching
                gt_obb = target["boxes_obb"]  # [G, 5] (cx, cy, w, h, theta) normalized
                gt_boxes_xyxy = obb_xywht_to_xyxy(gt_obb)  # [G, 8]
                # Use axis-aligned bounding rectangle for matching
                gt_x1 = gt_boxes_xyxy[:, 0::2].min(dim=1).values
                gt_y1 = gt_boxes_xyxy[:, 1::2].min(dim=1).values
                gt_x2 = gt_boxes_xyxy[:, 0::2].max(dim=1).values
                gt_y2 = gt_boxes_xyxy[:, 1::2].max(dim=1).values
                gt_boxes_match = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=1)
                gt_boxes_match[:, 0] *= img_w
                gt_boxes_match[:, 1] *= img_h
                gt_boxes_match[:, 2] *= img_w
                gt_boxes_match[:, 3] *= img_h
                gt_obb_pixel = gt_obb.clone()
                gt_obb_pixel[:, 0] *= img_w
                gt_obb_pixel[:, 1] *= img_h
                gt_obb_pixel[:, 2] *= img_w
                gt_obb_pixel[:, 3] *= img_h
                use_oriented = True
            else:
                gt_boxes_match = box_cxcywh_to_xyxy(target["boxes"])
                gt_boxes_match[:, 0] *= img_w
                gt_boxes_match[:, 1] *= img_h
                gt_boxes_match[:, 2] *= img_w
                gt_boxes_match[:, 3] *= img_h
                # Convert axis-aligned boxes to pseudo-oriented (theta=0)
                gt_obb_pixel = torch.cat([
                    box_xyxy_to_cxcywh(gt_boxes_match),
                    torch.zeros(len(gt_boxes_match), 1, device=device),
                ], dim=1)
                use_oriented = False

            ious = box_iou_matrix(props, gt_boxes_match)
            max_iou_per_prop, matched_gt_idx = ious.max(dim=1)

            pos_idx = torch.where(max_iou_per_prop >= self.fg_iou_thresh)[0]
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

            labels = torch.full((len(sampled_idx),), self.num_classes, dtype=torch.long, device=device)
            labels[:len(pos_idx)] = target["labels"][matched_gt_idx[pos_idx]]
            sampled_labels.append(labels)

            # Build oriented regression targets for positives
            reg_targets = torch.zeros(len(sampled_idx), 5, device=device)
            if len(pos_idx) > 0:
                matched_gt_obb = gt_obb_pixel[matched_gt_idx[pos_idx]]
                props_cxcywh = box_xyxy_to_cxcywh(props[pos_idx])
                reg_targets[:len(pos_idx)] = encode_oriented_deltas(props_cxcywh, matched_gt_obb)
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
        nms_thresh: float = 0.1,
        top_k: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process predictions for evaluation.

        Returns:
            List of dicts with 'scores', 'labels', 'boxes' (5-param oriented: cx, cy, w, h, theta).
        """
        num_images = len(proposals)
        num_classes = self.num_classes

        results = []
        start = 0
        for i in range(num_images):
            num_props = len(proposals[i])
            if num_props == 0:
                results.append({
                    "scores": torch.zeros(0, device=cls_logits.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=cls_logits.device),
                    "boxes": torch.zeros(0, 5, device=cls_logits.device),
                })
                continue

            end = start + num_props
            cls_i = cls_logits[start:end]
            reg_i = bbox_preds[start:end]
            props_i = proposals[i]
            start = end

            img_h, img_w = image_sizes[i]

            # Apply box deltas
            reg_i = reg_i.view(-1, num_classes, 5)
            scores = cls_i.softmax(dim=1)  # [P, num_classes]

            all_scores = []
            all_labels = []
            all_boxes = []
            for cls_idx in range(num_classes):
                cls_scores = scores[:, cls_idx]
                cls_reg = reg_i[:, cls_idx]

                # Apply oriented delta to proposals
                props_cxcywh = box_xyxy_to_cxcywh(props_i)
                obb = apply_oriented_deltas(props_cxcywh, cls_reg)  # [P, 5] in pixel coords

                # Clamp center and size
                obb[:, 0].clamp_(min=0, max=img_w)
                obb[:, 1].clamp_(min=0, max=img_h)
                obb[:, 2].clamp_(min=1, max=img_w * 2)
                obb[:, 3].clamp_(min=1, max=img_h * 2)
                # Clamp theta to [-pi/2, pi/2)
                obb[:, 4].clamp_(min=-math.pi / 2, max=math.pi / 2)

                # Filter by score
                keep = cls_scores > score_thresh
                if not keep.any():
                    continue

                boxes_cls = obb[keep]
                scores_cls = cls_scores[keep]

                # NMS
                keep_nms = rotated_nms(boxes_cls, scores_cls, nms_thresh)
                boxes_cls = boxes_cls[keep_nms]
                scores_cls = scores_cls[keep_nms]

                all_scores.append(scores_cls)
                all_labels.append(torch.full_like(scores_cls.long(), cls_idx))
                all_boxes.append(boxes_cls)

            if len(all_scores) == 0:
                results.append({
                    "scores": torch.zeros(0, device=cls_logits.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=cls_logits.device),
                    "boxes": torch.zeros(0, 5, device=cls_logits.device),
                })
                continue

            all_scores = torch.cat(all_scores)
            all_labels = torch.cat(all_labels)
            all_boxes = torch.cat(all_boxes)

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


# ---------------------------------------------------------------------------
# Oriented R-CNN Model
# ---------------------------------------------------------------------------


class OrientedRCNN(nn.Module):
    """Oriented R-CNN model with DINOv3 backbone.

    Uses the same RPN as Faster R-CNN (axis-aligned proposals) and
    oriented RoI heads for rotated bounding box prediction.
    """

    def __init__(
        self,
        backbone: MultiLevelBackbone,
        rpn: RPN,
        roi_heads: OrientedRoIHeads,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.num_classes = num_classes

    def forward(self, samples: NestedTensor, targets=None):
        """Forward pass.

        During training (targets is not None, self.training=True):
            Returns a dict of losses.

        During inference (targets is None):
            Returns a dict with 'pred_logits', 'pred_boxes'.
        """
        if not isinstance(samples, NestedTensor):
            from ..util.misc import nested_tensor_from_tensor_list

            samples = nested_tensor_from_tensor_list(samples)

        # Backbone
        features, _ = self.backbone(samples)
        feat_tensors = [f.tensors for f in features]

        # Image sizes
        image_sizes = []
        for i in range(samples.tensors.shape[0]):
            h = samples.tensors.shape[2]
            w = samples.tensors.shape[3]
            image_sizes.append((h, w))

        # RPN (axis-aligned proposals)
        cls_logits, bbox_preds, anchors = self.rpn(feat_tensors, image_sizes)
        proposals = self.rpn.generate_proposals(
            cls_logits, bbox_preds, anchors, image_sizes, is_training=self.training
        )

        if self.training and targets is not None:
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

        # Inference
        cls_logits, bbox_preds = self.roi_heads(feat_tensors, proposals)
        results = self.roi_heads.postprocess(
            cls_logits, bbox_preds, proposals, image_sizes,
            score_thresh=0.05, nms_thresh=self.roi_heads.nms_thresh,
            top_k=100,
        )

        # Convert to DETR-compatible format
        # pred_boxes stores 5 values (cx, cy, w, h, theta) per box
        max_dets = max(len(r["scores"]) for r in results) if results else 0
        if max_dets == 0:
            pred_logits = torch.zeros(len(results), 0, self.num_classes, device=samples.tensors.device)
            pred_boxes = torch.zeros(len(results), 0, 5, device=samples.tensors.device)
        else:
            pred_logits = torch.zeros(len(results), max_dets, self.num_classes, device=samples.tensors.device)
            pred_boxes = torch.zeros(len(results), max_dets, 5, device=samples.tensors.device)
            for i, r in enumerate(results):
                n = len(r["scores"])
                pred_logits[i, :n, r["labels"]] = r["scores"][:, None]
                img_h, img_w = image_sizes[i]
                pred_boxes[i, :n, 0] = r["boxes"][:, 0] / img_w  # cx
                pred_boxes[i, :n, 1] = r["boxes"][:, 1] / img_h  # cy
                pred_boxes[i, :n, 2] = r["boxes"][:, 2] / img_w  # w
                pred_boxes[i, :n, 3] = r["boxes"][:, 3] / img_h  # h
                pred_boxes[i, :n, 4] = r["boxes"][:, 4]           # theta (radians)

        out = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}

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
# Builder
# ---------------------------------------------------------------------------


def build_oriented_rcnn(backbone_model, config) -> OrientedRCNN:
    """Build Oriented R-CNN model from config.

    Args:
        backbone_model: ViT backbone model (loaded with pretrained weights).
        config: DetectionTrainConfig.
    """
    num_classes = config.num_classes

    embed_dim = backbone_model.embed_dim
    patch_size = backbone_model.patch_size
    n_blocks = backbone_model.n_blocks

    if hasattr(config, "head"):
        layers_to_use = config.head.layers_to_use
        use_layernorm = config.head.backbone_use_layernorm
        oriented_rcnn_cfg = config.oriented_rcnn
    else:
        layers_to_use = config.layers_to_use
        use_layernorm = getattr(config, "backbone_use_layernorm", True)
        oriented_rcnn_cfg = config

    if layers_to_use is None:
        layers_to_use = [m * n_blocks // 4 - 1 for m in range(1, 5)]

    # Multi-level backbone with feature pyramid
    backbone = MultiLevelBackbone(
        backbone_model=backbone_model,
        layers_to_use=layers_to_use,
        embed_dim=embed_dim,
        patch_size=patch_size,
        out_channels=oriented_rcnn_cfg.feature_hidden_dim,
        num_levels=oriented_rcnn_cfg.num_feature_levels,
        use_layernorm=use_layernorm,
    )

    # RPN (same as Faster R-CNN)
    rpn_cfg = oriented_rcnn_cfg.rpn
    num_anchors = len(rpn_cfg.anchor_sizes) * len(rpn_cfg.anchor_ratios)
    rpn_head = RPNHead(oriented_rcnn_cfg.feature_hidden_dim, num_anchors)
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

    # Oriented RoI heads
    roi_cfg = oriented_rcnn_cfg.roi_head
    roi_heads = OrientedRoIHeads(
        in_channels=oriented_rcnn_cfg.feature_hidden_dim,
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

    return OrientedRCNN(backbone, rpn, roi_heads, num_classes)
