# Oriented R-CNN 旋转目标检测支持

## 概述

在现有的 DETR 和 Faster R-CNN 检测框架基础上，新增 Oriented R-CNN 旋转目标检测器支持。Oriented R-CNN 在 Faster R-CNN 的架构上，将轴对齐的边界框回归替换为旋转边界框 (cx, cy, w, h, theta) 回归，适用于遥感图像、场景文本等需要旋转框的任务。

通过 `detector_type="oriented_rcnn"` 启用该检测器。

## 修改文件

### `dinov3/eval/detection/config.py`

新增 `OrientedRCNNConfig` 配置类，并在 `DetectionTrainConfig` 中增加 `oriented_rcnn` 字段。

| 配置类 | 说明 |
|--------|------|
| `OrientedRCNNConfig` | 包含 `rpn`、`roi_head`、`num_feature_levels`、`feature_hidden_dim`、`use_fpn` 以及旋转特有参数 `num_angles`、`angle_version` |

`DetectionTrainConfig` 新增字段:
- `detector_type` 新增可选值 `"oriented_rcnn"`
- `oriented_rcnn: OrientedRCNNConfig` — Oriented R-CNN 专属配置

### `dinov3/eval/detection/models/oriented_rcnn.py`（新文件）

Oriented R-CNN 完整实现，核心组件:

```
Backbone (ViT) → FeaturePyramid → RPN → axis-aligned proposals
                               → RoI Align → FC Head → oriented detections (cx, cy, w, h, theta)
```

#### 旋转框工具函数

| 函数 | 功能 |
|------|------|
| `obb_xywht_to_xyxy` | 旋转框 (cx, cy, w, h, theta) → 4 角点 (x0,y0,x1,y1,x2,y2,x3,y3)，CCW 顺序 |
| `obb_xyxy_to_xywht` | 4 角点 → 旋转框 (long-edge le90 定义，theta ∈ [-pi/2, pi/2)) |
| `encode_oriented_deltas` | 将旋转 GT 框编码为相对于轴对齐 proposal 的 5 参数 delta |
| `apply_oriented_deltas` | 对轴对齐 proposal 应用 5 参数 delta，得到旋转框 |
| `rotated_iou_matrix` | 两组旋转框之间的成对 IoU 矩阵（Sutherland-Hodgman 裁剪法求交集） |
| `rotated_nms` | 旋转非极大值抑制 |

#### 模型类

| 类 | 职责 |
|----|------|
| `OrientedRoIHeads` | RoI Align 提取 7x7 特征 → 2 层 FC (1024 dim) → 分类 + 5 参数回归 (cx,cy,w,h,theta)；训练时采样正负 proposal、计算损失；推理时 per-class rotated NMS 后处理 |
| `OrientedRCNN` | 顶层模型。复用 Faster R-CNN 的 RPN（生成轴对齐 proposal），搭配 `OrientedRoIHeads` 输出旋转框 |
| `build_oriented_rcnn` | 从配置构建 Oriented R-CNN 的工厂函数 |

### `dinov3/eval/detection/train.py`

| 函数 | 改动 |
|------|------|
| `build_detection_model` | 新增 `detector_type == "oriented_rcnn"` 分支，调用 `build_oriented_rcnn` |
| `train_one_epoch` | `detector_type in ("faster_rcnn", "oriented_rcnn")` 使用内置损失 |
| `evaluate` | `oriented_rcnn` 模式下 5 参数输出转外接轴对齐框提交 COCO 评估 |
| `train_detection` | `oriented_rcnn` 跳过 DETR criterion 和后处理器 |

### `dinov3/eval/detection/dataset.py`

| 新增 | 说明 |
|------|------|
| `DOTADetection` | DOTA 格式旋转检测数据集：8 点多边形 → (cx, cy, w, h, theta) |
| `_polygon_to_obb` | 4 点四边形到旋转框的转换（le90 定义） |
| `RandomHorizontalFlip` | 新增 `boxes_obb` 水平翻转处理（cx 取反、theta 取负） |
| `Normalize` | 新增 `boxes_obb` 归一化（cx、w 按宽度除，cy、h 按高度除，theta 不变） |

## 架构对比

| | Faster R-CNN | Oriented R-CNN |
|------|------|------|
| Proposal | Axis-aligned RPN | Axis-aligned RPN |
| Box 参数 | 4 (cx, cy, w, h) | 5 (cx, cy, w, h, theta) |
| 回归目标 | `encode_box_deltas` (4) | `encode_oriented_deltas` (5) |
| IoU 计算 | Axis-aligned IoU | Rotated IoU (Sutherland-Hodgman) |
| NMS | Axis-aligned NMS | Rotated NMS |
| 角度表示 | — | le90: theta ∈ [-pi/2, pi/2) |

## 使用方式

```bash
# Oriented R-CNN 训练（DOTA 数据集）
python -m dinov3.eval.detection.run \
    output_dir=./output/oriented_rcnn \
    model=dinov3_vit7b16 \
    detector_type=oriented_rcnn \
    datasets.train_img_dir=/path/to/DOTA/train/images \
    datasets.train_ann_file=/path/to/DOTA/train/annotations.json \
    datasets.val_img_dir=/path/to/DOTA/val/images \
    datasets.val_ann_file=/path/to/DOTA/val/annotations.json \
    epochs=60 batch_size=2
```

## 关键配置项

```yaml
detector_type: oriented_rcnn

oriented_rcnn:
  num_feature_levels: 5
  feature_hidden_dim: 256
  num_angles: 180         # 角度预测粒度（预留）
  angle_version: "le90"   # 角度表示：le90 ([-90,90)), le135 ([-135,45)), oc ([0,180))
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
    box_nms_thresh: 0.1    # 旋转 NMS 阈值通常更低
```

## 数据集格式

### DOTA JSON 格式

```json
{
  "images": [
    {"id": 1, "file_name": "P0001.png", "height": 1024, "width": 1024}
  ],
  "annotations": [
    {
      "image_id": 1,
      "category_id": 0,
      "segmentation": [[x1, y1, x2, y2, x3, y3, x4, y4]],
      "bbox": [x, y, w, h]
    }
  ],
  "categories": [
    {"id": 0, "name": "plane"}
  ]
}
```

`DOTADetection` 优先使用 `bbox_obb` 字段（预计算的 5 参数框），其次使用 `segmentation` 的多边形转换为旋转框，最后回退到轴对齐 `bbox`（theta=0）。
