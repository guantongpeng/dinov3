# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
DETR criterion (loss function) with support for:
- Focal loss (sigmoid-based, for PlainDETR)
- L1 + GIoU box regression loss
- One-to-one + one-to-many hybrid matching
- Auxiliary losses at each decoder layer
- Box refinement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .matcher import HungarianMatcher


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.0):
    """Loss used in PlainDETR for classification (sigmoid, not softmax)."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses,
        focal_alpha=0.25,
        k_one2many=0,
        lambda_one2many=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.k_one2many = k_one2many
        self.lambda_one2many = lambda_one2many

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (focal loss, sigmoid-based)."""
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        # Build one-hot targets: matched queries get 1 at their class, all others get all-0 (background)
        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], self.num_classes],
            dtype=src_logits.dtype,
            device=src_logits.device,
        )
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_onehot[idx] = target_classes_o.unsqueeze(-1).float()

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2.0
        ) * src_logits.shape[1]
        losses = {"loss_cls": loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes (L1 + GIoU)."""
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses = {"loss_bbox": loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes))
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """This performs the loss computation.

        Params:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size.
                     The expected keys in each dict are:
                         "labels": Tensor of dim [num_target_boxes] (class label)
                         "boxes": Tensor of dim [num_target_boxes, 4] (cxcywh format)
        """
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if torch.distributed.is_initialized() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / torch.distributed.get_world_size(), min=1)
        num_boxes = num_boxes.item()

        # --- One-to-one matching loss ---
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss in self.losses:
            losses.update(getattr(self, f"loss_{loss}")(outputs, targets, indices, num_boxes))

        # --- One-to-many matching loss ---
        if "pred_logits_one2many" in outputs and self.k_one2many > 0:
            outputs_one2many = {
                "pred_logits": outputs["pred_logits_one2many"],
                "pred_boxes": outputs["pred_boxes_one2many"],
            }
            targets_one2many = []
            for t in targets:
                t_repeat = {
                    k: v.repeat(self.k_one2many, *([1] * (v.dim() - 1))) for k, v in t.items()
                }
                targets_one2many.append(t_repeat)
            indices_one2many = self.matcher(outputs_one2many, targets_one2many)
            for loss in self.losses:
                l_dict = getattr(self, f"loss_{loss}")(
                    outputs_one2many, targets_one2many, indices_one2many, num_boxes * self.k_one2many
                )
                l_dict = {k + "_one2many": v * self.lambda_one2many for k, v in l_dict.items()}
                losses.update(l_dict)

        # --- Auxiliary losses (one-to-one) ---
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices_aux = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    l_dict = getattr(self, f"loss_{loss}")(aux_outputs, targets, indices_aux, num_boxes)
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # --- Auxiliary losses (one-to-many) ---
        if "aux_outputs_one2many" in outputs and self.k_one2many > 0:
            for i, aux_outputs in enumerate(outputs["aux_outputs_one2many"]):
                aux_o2m = {
                    "pred_logits": aux_outputs["pred_logits"],
                    "pred_boxes": aux_outputs["pred_boxes"],
                }
                indices_aux = self.matcher(aux_o2m, targets_one2many)
                for loss in self.losses:
                    l_dict = getattr(self, f"loss_{loss}")(
                        aux_o2m, targets_one2many, indices_aux, num_boxes * self.k_one2many
                    )
                    l_dict = {
                        k + f"_one2many_aux_{i}": v * self.lambda_one2many
                        for k, v in l_dict.items()
                    }
                    losses.update(l_dict)

        # --- Encoder outputs (two-stage) loss ---
        if "enc_outputs" in outputs:
            enc_outputs = outputs["enc_outputs"]
            # For two-stage, use a simple binary foreground classification + box loss
            bin_targets = []
            for t in targets:
                bin_targets.append({"labels": torch.zeros_like(t["labels"]), "boxes": t["boxes"]})
            enc_indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                l_dict = getattr(self, f"loss_{loss}")(enc_outputs, bin_targets, enc_indices, num_boxes)
                l_dict = {k + "_enc": v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses


def build_criterion(config):
    """Build criterion from DetectionTrainConfig.

    Args:
        config: DetectionTrainConfig with .head, .matcher, .loss sub-configs
    """
    matcher = HungarianMatcher(
        cost_class=config.matcher.matcher_cost_class,
        cost_bbox=config.matcher.matcher_cost_bbox,
        cost_giou=config.matcher.matcher_cost_giou,
    )

    # Base loss weights (for the final decoder layer)
    base_weight_dict = {
        "loss_cls": config.loss.cls_loss_coef,
        "loss_bbox": config.loss.bbox_loss_coef,
        "loss_giou": config.loss.giou_loss_coef,
    }

    # Build full weight dict
    weight_dict = dict(base_weight_dict)

    # Auxiliary losses (one-to-one): each intermediate decoder layer
    if config.head.aux_loss:
        for i in range(config.head.dec_layers - 1):
            weight_dict.update({k + f"_aux_{i}": v for k, v in base_weight_dict.items()})

    # One-to-many losses: final decoder layer
    if config.head.k_one2many > 0:
        for k, v in base_weight_dict.items():
            weight_dict[k + "_one2many"] = v * config.head.lambda_one2many
        # One-to-many auxiliary
        if config.head.aux_loss:
            for i in range(config.head.dec_layers - 1):
                for k, v in base_weight_dict.items():
                    weight_dict[k + f"_one2many_aux_{i}"] = v * config.head.lambda_one2many

    # Encoder (two-stage) losses
    if config.head.two_stage:
        for k, v in base_weight_dict.items():
            weight_dict[k + "_enc"] = v

    losses = ["labels", "boxes"]

    criterion = SetCriterion(
        num_classes=config.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        focal_alpha=0.25,
        k_one2many=config.head.k_one2many,
        lambda_one2many=config.head.lambda_one2many,
    )
    return criterion, weight_dict
