# DINOv3 目标检测训练模块实现文档

## 概述

DINOv3 原始仓库仅提供了检测任务的模型架构和推理代码（模型 + COCO 预训练权重），缺少训练所需的完整流水线。本文档记录了缺失模块的补充实现，包括损失函数、匹配器、数据集加载和训练脚本。

## 文件清单

### 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `dinov3/eval/detection/matcher.py` | 106 | HungarianMatcher——预测框与真实框之间的二部图匹配 |
| `dinov3/eval/detection/criterion.py` | 260 | SetCriterion——Focal Loss + L1 + GIoU 损失函数，支持混合匹配 |
| `dinov3/eval/detection/dataset.py` | 325 | COCO 数据集、自定义 JSON 数据集、数据增强、DataLoader |
| `dinov3/eval/detection/train.py` | 568 | 训练脚本——模型构建、优化器、学习率调度、训练/验证循环 |
| `dinov3/eval/detection/run.py` | 90 | CLI 入口，支持 YAML 配置文件和命令行参数 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `dinov3/eval/detection/config.py` | 新增 `DetectionTrainConfig`、`MatcherConfig`、`LossConfig`、`OptimizerConfig`、`SchedulerConfig`、`TransformConfig`、`DatasetsConfig`、`RPNConfig`、`RoIHeadConfig`、`FasterRCNNConfig` |
| `dinov3/eval/detection/util/box_ops.py` | 新增 `box_area`、`box_iou`、`generalized_box_iou` 工具函数 |

### 未修改的已有文件

| 文件 | 说明 |
|------|------|
| `dinov3/eval/detection/models/detr.py` | PlainDETR / PlainDETRReParam 模型 |
| `dinov3/eval/detection/models/transformer.py` | Transformer / TransformerReParam |
| `dinov3/eval/detection/models/backbone.py` | DINOBackbone、BackboneWithPositionEncoding |
| `dinov3/eval/detection/models/global_rpe_decomp_decoder.py` | GlobalDecoder（全局交叉注意力解码器） |
| `dinov3/eval/detection/models/global_ape_decoder.py` | GlobalAPEDecoder（绝对位置编码解码器） |
| `dinov3/eval/detection/models/windows.py` | WindowsWrapper（大图窗口切分） |
| `dinov3/eval/detection/models/position_encoding.py` | 位置编码实现 |
| `dinov3/eval/detection/util/misc.py` | NestedTensor、collate、辅助函数 |
| `dinov3/hub/detectors.py` | DetectorWithProcessor、COCO 预训练权重加载 |

## 架构设计

### 训练流水线

```
数据集 (COCO / 自定义 JSON)
   │
   ▼
数据增强 (随机缩放、随机翻转、归一化)
   │
   ▼
DINOv3 Backbone (默认冻结)
   │
   ├── DETR 路径 ──────────────────────────────┐
   │  Transformer Encoder                       │
   │  Transformer Decoder (global_rpe_decomp)   │
   │  分类头 (nn.Linear)                         │
   │  回归头 (3层 MLP)                           │
   │      │                                      │
   │      ▼                                      │
   │  HungarianMatcher ──► SetCriterion         │
   │      │                    │                 │
   │      ▼                    ▼                 │
   │  loss_cls (Focal) + loss_bbox (L1)         │
   │  + loss_giou + 辅助损失                     │
   │                                             │
   └── Faster R-CNN 路径 ────────────────────────┘
       RPN + RoIHead
       (内置损失，无需外部分配器)
```

### 损失函数设计

#### PlainDETR 分类损失：Sigmoid + Focal Loss

与标准 DETR 使用 softmax + cross-entropy 不同，DINOv3 使用的 PlainDETR 采用 sigmoid + focal loss：

- 每个 query 独立输出每个类别的置信度（sigmoid 激活）
- 无背景类——未匹配的 query 所有类别目标设为 0
- Focal loss 参数：`alpha=0.25, gamma=2.0`

```
sigmoid_focal_loss(inputs, targets) =
    alpha_t * (1 - p_t)^gamma * BCEWithLogits(inputs, targets)
```

#### 混合匹配（Hybrid Matching）

同时使用 one-to-one 和 one-to-many 两种匹配策略：

| 组件 | Query 数量 | 匹配方式 | 权重 |
|------|-----------|----------|------|
| One-to-one 主损失 | 300 | 匈牙利算法，每个 GT 匹配 1 个 query | 1.0× |
| One-to-many 主损失 | 1500 | 匈牙利算法，每个 GT 重复 k=6 次 | `lambda_one2many` × |
| One-to-one 辅助损失 | 300 × 5 层 | 同上 | 1.0× |
| One-to-many 辅助损失 | 1500 × 5 层 | 同上 | `lambda_one2many` × |
| Encoder 两阶段损失 | Proposal 输出 | 匈牙利算法，二值前景分类 | 1.0× |

权重字典共 39 个键（3 种损失 × (1 最终 + 5 辅助 + 1 Encoder) × (1 o2o + 1 o2m) - 3 个 Encoder 无 o2m）。

## 关键设计决策

1. **Sigmoid Focal Loss**：PlainDETR 使用逐类 sigmoid + focal loss，区别于标准 DETR 的 softmax。每个 query 独立预测每类的 logit，没有显式的背景类。

2. **ReParam 模式**：默认使用 `PlainDETRReParam`，采用绝对 XYXY 坐标 + `delta2bbox` 变换。损失始终在归一化的 cxcywh 空间计算。

3. **Self-attention 遮罩**：one-to-one 和 one-to-many 的 query 通过布尔 attention mask 互相隔离，防止信息泄漏。

4. **两阶段生成**：Encoder 输出经过 4 级多尺度投影生成初始 proposal，按前景分数 Top-K 筛选后送入 Decoder。

## 使用方式

### 基础训练（COCO 数据集）

```bash
python -m dinov3.eval.detection.run \
    output_dir=./output/detection_coco \
    model=dinov3_vit7b16 \
    num_classes=91 \
    datasets.train_img_dir=/path/to/train2017 \
    datasets.train_ann_file=/path/to/instances_train2017.json \
    datasets.val_img_dir=/path/to/val2017 \
    datasets.val_ann_file=/path/to/instances_val2017.json \
    epochs=60 batch_size=2
```

### 使用 YAML 配置文件

```bash
python -m dinov3.eval.detection.run \
    config=configs/detection_train.yaml \
    output_dir=./output/detection
```

### 自定义数据集微调

```bash
python -m dinov3.eval.detection.run \
    output_dir=./output/detection_custom \
    model=dinov3_vit7b16 \
    num_classes=10 \
    datasets.train_img_dir=/path/to/train_images \
    datasets.train_ann_file=/path/to/train.json \
    head.k_one2many=0 \
    head.num_queries_one2one=300 \
    epochs=30 batch_size=4
```

### 从 Checkpoint 恢复训练

```bash
python -m dinov3.eval.detection.run \
    load_from=./output/detection/checkpoint_best.pth \
    output_dir=./output/detection_resume \
    model=dinov3_vit7b16
```

## 配置参数参考

### DetectionTrainConfig（顶层训练配置）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `str \| None` | `None` | Backbone 名称（`dinov3_vit7b16`、`dinov3_vitl16plus`） |
| `detector_type` | `str` | `"detr"` | 检测器类型，`"detr"` 或 `"faster_rcnn"` |
| `num_classes` | `int` | `91` | 目标类别数 |
| `epochs` | `int` | `60` | 训练轮数 |
| `batch_size` | `int` | `2` | 每 GPU 批大小 |
| `num_workers` | `int` | `4` | DataLoader 工作进程数 |
| `seed` | `int` | `42` | 随机种子 |
| `eval_interval` | `int` | `5` | 每 N 个 epoch 验证一次 |
| `train_backbone` | `bool` | `False` | 是否训练 backbone |
| `train_encoder` | `bool` | `True` | 是否训练 Transformer Encoder |
| `output_dir` | `str` | `""` | 输出目录 |

### MatcherConfig（匹配器配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `matcher_cost_class` | `2.0` | 匈牙利匹配中分类代价系数 |
| `matcher_cost_bbox` | `5.0` | L1 框回归代价系数 |
| `matcher_cost_giou` | `2.0` | GIoU 代价系数 |

### LossConfig（损失权重配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `cls_loss_coef` | `2.0` | Focal Loss 权重 |
| `bbox_loss_coef` | `5.0` | L1 框回归损失权重 |
| `giou_loss_coef` | `2.0` | GIoU 损失权重 |

### OptimizerConfig（优化器配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `lr` | `1e-4` | 检测头 + Encoder 学习率 |
| `lr_backbone` | `1e-5` | Backbone 学习率（基础 LR 的 0.1×） |
| `weight_decay` | `1e-4` | AdamW 权重衰减 |
| `gradient_clip` | `0.1` | 梯度裁剪最大范数 |
| `beta1` | `0.9` | AdamW beta1 |
| `beta2` | `0.999` | AdamW beta2 |

### SchedulerConfig（学习率调度配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `type` | `"multistep"` | 类型：`"multistep"`、`"cosine"`、`"warmup_multistep"` |
| `milestones` | `[40, 55]` | 学习率衰减的 epoch 节点 |
| `warmup_epochs` | `1` | 预热 epoch 数（warmup_multistep） |
| `warmup_factor` | `1e-3` | 预热初始学习率倍率 |

### TransformConfig（数据增强配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `train_short_side_range` | `(480, 800)` | 训练时短边随机缩放范围 |
| `train_max_size` | `1333` | 训练时长边最大值 |
| `train_flip_prob` | `0.5` | 随机水平翻转概率 |
| `eval_short_side` | `800` | 验证时短边尺寸 |
| `eval_max_size` | `1333` | 验证时长边最大值 |

### DatasetsConfig（数据集配置）

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `train_img_dir` | `""` | 训练集图片目录 |
| `train_ann_file` | `""` | 训练集 COCO 格式 JSON 标注 |
| `val_img_dir` | `""` | 验证集图片目录 |
| `val_ann_file` | `""` | 验证集 COCO 格式 JSON 标注 |

## 数据集格式

### COCO 格式（CocoDetection）

标准 COCO JSON 标注格式，通过 `pycocotools` 加载。支持分割掩码（可选）。

### 自定义 JSON 格式（CustomDetectionDataset）

不依赖 `pycocotools` 的最小化 JSON 格式：

```json
{
    "images": [
        {"id": 1, "file_name": "img001.jpg", "height": 480, "width": 640}
    ],
    "annotations": [
        {"image_id": 1, "bbox": [100, 50, 200, 150], "category_id": 0}
    ],
    "categories": [
        {"id": 0, "name": "person"},
        {"id": 1, "name": "car"}
    ]
}
```

### 目标数据格式约定

每个 target 字典必须包含：
- `boxes`：`torch.Tensor`，形状 `[N, 4]`，XYXY 格式，绝对像素坐标
- `labels`：`torch.Tensor`，形状 `[N]`，0 起始的类别索引
- `image_id`：`int`（可选，用于 COCO 评估）
- `orig_size`：`torch.Tensor`，形状 `[2]`，`[H, W]` 格式（数据增强时自动添加）

## 调参建议

| 场景 | 建议 |
|------|------|
| **冻结 Backbone** | 默认冻结，仅训练检测头 + Encoder；通过 `blocks_to_train` 解冻部分 ViT block 进行微调 |
| **小数据集（< 5K 图片）** | 关闭 one2many（`k_one2many=0`），减少 query 数（`num_queries_one2one=300`） |
| **少类别（< 10 类）** | 仅修改 `num_classes`，其他结构不变 |
| **大分辨率图片** | 开启 `n_windows_sqrt=3`，但显存消耗翻倍（窗口 + 全局特征 concat） |
| **迁移学习** | `train_backbone=False`，仅训练检测头；如需微调 backbone，设 `lr_backbone=1e-5` |
| **ReParam 模式** | 推荐开启（默认），绝对坐标回归更稳定 |

## 验证结果

所有模块通过导入测试和功能测试：

```
box_ops：area、iou、generalized_box_iou —— 通过
matcher：HungarianMatcher —— 通过（处理空目标、可变框数量）
criterion：SetCriterion —— 通过（Focal Loss、o2o + o2m + aux + encoder 损失共 39 个权重）
dataset：CocoDetection、CustomDetectionDataset、数据增强 —— 通过
config：DetectionTrainConfig 及所有子配置 —— 通过
train：build_detection_model、train_one_epoch —— 通过
run：CLI 入口 —— 通过
```
