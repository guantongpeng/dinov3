# DIOR 数据集 Faster R-CNN 微调训练 (DINOv3)

## 概述

本文档描述使用 DINOv3 ViT-B/16 作为骨干网络，在 DIOR 遥感目标检测数据集上进行 Faster R-CNN 下游任务微调训练。骨干网络冻结，仅训练 FPN + RPN + RoI Head。

## DIOR 数据集

**DIOR** (Dataset for object deTection in Optical Remote sensing images) 是一个大规模遥感图像目标检测数据集。

| 属性 | 值 |
|---|---|
| 类别数 | 20 |
| 训练集 | 5,862 张图像, 32,591 个标注 |
| 验证集 | 5,863 张图像, 35,434 个标注 |
| 测试集 | 11,738 张图像, 124,440 个标注 |
| 图像尺寸 | 800×800 像素 |
| 格式 | PASCAL VOC XML → COCO JSON |

### DIOR 20 类

```
airplane, airport, baseballfield, basketballcourt, bridge,
chimney, dam, Expressway-Service-area, Expressway-toll-station,
golffield, groundtrackfield, harbor, overpass, ship,
stadium, storagetank, tenniscourt, trainstation, vehicle, windmill
```

### 数据集划分

数据集使用 `ImageSets/Main` 下的官方划分文件：

| 划分 | 文件 | 图像数 | 标注数 |
|---|---|---|---|
| train | `train.txt` | 5,862 | 32,591 |
| val | `val.txt` | 5,863 | 35,434 |
| test | `test.txt` | 11,738 | 124,440 |
| **总计** | | **23,463** | **192,465** |

### 数据集准备

```bash
# 1. 解压数据
cd data/DIOR
unzip Annotations.zip -d DIOR/Annotations/
unzip JPEGImages-trainval.zip -d DIOR/
mv DIOR/JPEGImages-trainval/*.jpg DIOR/JPEGImages/
unzip JPEGImages-test.zip -d DIOR/
mv DIOR/JPEGImages-test/*.jpg DIOR/JPEGImages/
unzip ImageSets.zip -d DIOR/

# 2. VOC XML → COCO JSON 转换（使用 ImageSets/Main 中的官方 train/val/test 划分）
cd ../..  # 回到项目根目录
python scripts/convert_dior_to_coco.py

# 3. 创建数据符号链接
cd data/DIOR
ln -sf DIOR/JPEGImages train2019
ln -sf DIOR/JPEGImages val2019
```

## 训练配置

### 模型

- **骨干网络**: DINOv3 ViT-B/16 (冻结, frozen_stages=-1)
- **Neck**: FPN (256 channels, 5 层输出)
- **检测头**: Faster R-CNN (RPN + StandardRoIHead with Shared2FCBBoxHead)
- **预训练权重**: timm vit_base_patch16_dinov3.lvd1689m (转换为项目自定义格式)

### 超参数

| 参数 | 值 |
|---|---|
| 优化器 | AdamW (lr=1e-4, weight_decay=1e-4) |
| 骨干网络 LR 倍率 | 0.0 (冻结) |
| LR 调度 | Linear warmup (500 iter) + MultiStepLR (milestones=[8,11], gamma=0.1) |
| 训练轮数 | 12 (1x schedule) |
| Batch size | 2 (AspectRatioBatchSampler) |
| 混合精度 | AMP |
| 梯度裁剪 | max_norm=1.0 |
| 评估间隔 | 每 3 个 epoch |

### 关键配置说明

DIOR 数据集仅有 20 个类别（不同于 COCO 的 80 类），配置通过 `_base_` 继承 `dinov3_faster_rcnn_base.py` 的模型结构（Faster R-CNN + FPN + RPN + RoI Head），并做以下覆写：

- **类别数**: `num_classes = 20`，通过 `roi_head.bbox_head.num_classes` 覆写
- **类别名**: 通过 `metainfo` 显式设置 20 个 DIOR 类别
- **骨干网络**: DINOv3 ViT-B/16，`frozen_stages=-1` 冻结，使用自定义预训练权重
- **数据划分**: 严格使用 `ImageSets/Main` 中的 `train.txt`/`val.txt` 划分

## 训练命令

```bash
# 冻结骨干网络训练（仅微调 FPN + RPN + RoI Head）
python -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_dior.py \
    --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_dior \
    --amp

# 端到端微调（同时训练骨干网络）
python -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_dior.py \
    --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_dior \
    --train-backbone --amp
```

## 训练结果

### Epoch 3 (初步结果)

| 指标 | 值 |
|---|---|
| bbox_mAP | 0.095 |
| bbox_mAP_50 | 0.212 |
| bbox_mAP_75 | 0.081 |
| bbox_mAP_s | 0.000 |
| bbox_mAP_m | 0.121 |
| bbox_mAP_l | 0.123 |

> 训练仍在进行中（共 12 epochs），最终结果将更新。

## 文件说明

| 文件路径 | 说明 |
|---|---|
| `data/DIOR/` | DIOR 数据集根目录 |
| `data/DIOR/DIOR/Annotations/` | VOC XML 标注 (23,463 个文件) |
| `data/DIOR/DIOR/JPEGImages/` | 图像文件 (23,463 张) |
| `data/DIOR/DIOR/ImageSets/Main/` | train/val/test 划分 |
| `data/DIOR/annotations/` | COCO JSON 标注（由 `scripts/convert_dior_to_coco.py` 生成） |
| `scripts/convert_dior_to_coco.py` | VOC→COCO 格式转换脚本（按 ImageSets/Main 划分） |
| `dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_dior.py` | 训练配置 |
| `work_dirs/faster_rcnn_dinov3_vitb16_dior/` | 训练输出（checkpoint、日志） |
