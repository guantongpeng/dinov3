# Faster R-CNN 目标检测下游任务支持

## 概述

在现有的 DETR-based 检测框架基础上，新增 Faster R-CNN 目标检测器支持。通过 `detector_type` 配置项切换检测器类型，保持向后兼容。

## 修改文件

### `dinov3/eval/detection/config.py`

新增 3 个配置类，并在 `DetectionTrainConfig` 中增加 `detector_type` 和 `faster_rcnn` 字段。

| 配置类 | 说明 |
|--------|------|
| `RPNConfig` | RPN anchor 参数、NMS 阈值、正负样本匹配 IoU 阈值、采样参数、损失权重 |
| `RoIHeadConfig` | RoI 采样参数、RoI Align 输出尺寸、全连接层维度、损失权重 |
| `FasterRCNNConfig` | 特征金字塔通道数 (`feature_hidden_dim=256`)、层数 (`num_feature_levels=5`)、是否使用 FPN |

`DetectionTrainConfig` 新增字段：
- `detector_type: str = "detr"` — 可选 `"detr"` 或 `"faster_rcnn"`
- `faster_rcnn: FasterRCNNConfig` — Faster R-CNN 专属配置

### `dinov3/eval/detection/models/faster_rcnn.py`（新文件）

Faster R-CNN 完整实现，核心组件：

```
Backbone (ViT) → FeaturePyramid → RPN → proposals
                               → RoI Align → FC Head → detections
```

| 类 | 职责 |
|----|------|
| `FeaturePyramid` | 将 ViT 单尺度特征通过 strided conv 构建多尺度金字塔 |
| `MultiLevelBackbone` | 包装 DINOv3 ViT backbone，提取中间层特征并构建 FPN，输出 `List[NestedTensor]` |
| `AnchorGenerator` | 为每层特征图生成 (cx, cy, w, h) 格式的 anchor boxes |
| `RPNHead` | 共享权重的 3×3 conv + 1×1 cls conv + 1×1 reg conv |
| `RPN` | 完整 RPN：anchor-target 匹配（IoU-based）、损失计算（BCE + smooth L1）、proposal 生成（NMS + top-K） |
| `RoIHeads` | RoI Align 提取 7×7 特征 → 2 层 FC (1024 dim) → 分类/回归分支，per-class NMS 后处理 |
| `FasterRCNN` | 顶层模型。训练时 `forward(samples, targets)` 返回 loss dict；推理时返回 `{"pred_logits", "pred_boxes"}` 兼容 DETR 格式 |

辅助函数：`box_iou_matrix`、`encode_box_deltas`、`apply_box_deltas`、`nms`、`build_faster_rcnn`

### `dinov3/eval/detection/train.py`

| 函数 | 改动 |
|------|------|
| `build_detection_model` | 根据 `config.detector_type` 分支构建 DETR 或 Faster R-CNN |
| `train_one_epoch` | 新增 `detector_type` 参数；Faster R-CNN 模式下直接调用 `model(samples, targets)` 获取 loss |
| `evaluate` | 新增 `detector_type` 参数；Faster R-CNN 模式下单次前向同时获取预测和损失；预测结果转换为 COCO 评估格式 |
| `train_detection` | 根据 `detector_type` 控制 criterion 构建、postprocessor 创建、训练/评估参数传递 |

## 架构对比

| | DETR | Faster R-CNN |
|------|------|------|
| 检测范式 | Transformer query-based | Anchor-based two-stage |
| Backbone 输出 | 多层特征拼接为单尺度 | 特征金字塔 5 个尺度 |
| Proposal 生成 | Learned query + decoder | RPN + NMS |
| 匹配策略 | Hungarian 匹配 | IoU-based anchor 匹配 |
| 分类损失 | Sigmoid Focal Loss | Cross-Entropy |
| 回归损失 | L1 + GIoU | Smooth L1 |
| 后处理 | top-K 直接输出 | Per-class NMS |

## 使用方式

```bash
# Faster R-CNN 训练
python -m dinov3.eval.detection.run \
    output_dir=./output/faster_rcnn \
    model=dinov3_vit7b16 \
    detector_type=faster_rcnn \
    datasets.train_img_dir=/path/to/train2017 \
    datasets.train_ann_file=/path/to/instances_train2017.json \
    datasets.val_img_dir=/path/to/val2017 \
    datasets.val_ann_file=/path/to/instances_val2017.json \
    epochs=60 batch_size=2

# 默认 DETR（向后兼容，无需改动）
python -m dinov3.eval.detection.run \
    output_dir=./output/detr \
    model=dinov3_vit7b16 \
    ...
```

## 关键配置项

```yaml
detector_type: faster_rcnn

faster_rcnn:
  num_feature_levels: 5
  feature_hidden_dim: 256
  rpn:
    anchor_sizes: [32, 64, 128, 256, 512]
    anchor_ratios: [0.5, 1.0, 2.0]
    rpn_nms_thresh: 0.7
    rpn_fg_iou_thresh: 0.7
    rpn_bg_iou_thresh: 0.3
  roi_head:
    box_roi_output_size: 7
    box_head_fc_dim: 1024
    box_fg_iou_thresh: 0.5
    box_nms_thresh: 0.5
```
