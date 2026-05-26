# COCO128 Faster R-CNN Fine-tuning with DINOv3

## 概述

本文档描述了如何使用 DINOv3 作为骨干网络 (backbone)，在 COCO128 数据集上进行 Faster R-CNN 目标检测下游任务的微调训练。

## 数据集

**COCO128** 是 COCO 数据集的一个小型子集，包含：
- 128 张图像（从 COCO train2017 中采样）
- 80 个类别
- 训练集：108 张图像，823 个标注
- 验证集：20 张图像，106 个标注

数据集位置：`data/coco128/`

### 数据集结构

```
data/coco128/
  annotations/
    instances_train2017.json  # COCO format train annotations
    instances_val2017.json    # COCO format val annotations
  train2017/ -> images/train2017/  # 128 images (symlink)
  val2017/ -> images/train2017/    # same images, filtered by annotation
  images/train2017/           # actual image files
```

### 数据集准备

```bash
# 1. 下载 COCO128
mkdir -p data/coco128 && cd data/coco128
curl -L -o coco128.zip "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
unzip coco128.zip

# 2. 转换 YOLO 格式 → COCO JSON 格式
python data/coco128/convert_yolo_to_coco.py

# 3. 创建 MMDetection 期望的目录结构
cd data/coco128
mkdir -p annotations
cd annotations
ln -sf ../train.json instances_train2017.json
ln -sf ../val.json instances_val2017.json
cd ..
ln -sf images/train2017 train2017
ln -sf images/train2017 val2017
```

## 预训练权重

由于 Meta 官方 CDN (`dl.fbaipublicfiles.com`) 已下线，本项目使用 **timm** 提供的 DINOv3 权重作为替代。

### 权重准备

```bash
# 1. 下载 timm 格式的 DINOv3 ViT-B/16 权重
curl -L -o /tmp/dinov3_vitb16_timm.safetensors \
  "https://huggingface.co/timm/vit_base_patch16_dinov3.lvd1689m/resolve/main/model.safetensors"

# 2. 转换为项目自定义格式
python scripts/convert_timm_to_dinov3.py \
  /tmp/dinov3_vitb16_timm.safetensors \
  -o data/pretrained_weights/dinov3_vitb16_timm.pth
```

### 权重转换详情

timm 的 state dict 与项目自定义 `DinoVisionTransformer` 的映射关系：

| timm key | custom key | 说明 |
|---|---|---|
| `cls_token` | `cls_token` | 分类 token |
| `reg_token` | `storage_tokens` | 寄存器 token |
| `patch_embed.proj.*` | `patch_embed.proj.*` | Patch 嵌入层 |
| `blocks.{i}.norm1.*` | `blocks.{i}.norm1.*` | 注意力前 LayerNorm |
| `blocks.{i}.attn.qkv.weight` | `blocks.{i}.attn.qkv.weight` | QKV 投影权重 |
| `blocks.{i}.attn.proj.*` | `blocks.{i}.attn.proj.*` | 注意力输出投影 |
| `blocks.{i}.gamma_1` | `blocks.{i}.ls1.gamma` | Layer Scale 1 |
| `blocks.{i}.norm2.*` | `blocks.{i}.norm2.*` | MLP 前 LayerNorm |
| `blocks.{i}.mlp.fc1.*` | `blocks.{i}.mlp.fc1.*` | MLP 第一层 |
| `blocks.{i}.mlp.fc2.*` | `blocks.{i}.mlp.fc2.*` | MLP 第二层 |
| `blocks.{i}.gamma_2` | `blocks.{i}.ls2.gamma` | Layer Scale 2 |
| `norm.*` | `norm.*` | 最终 LayerNorm |

以下参数保留项目初始化值（timm 格式中不存在）：
- `mask_token`：遮罩 token
- `rope_embed.periods`：RoPE 位置编码周期
- `blocks.{i}.attn.qkv.bias` / `bias_mask`：注意力偏置和掩码

## 训练配置

### 训练命令

```bash
# 基础训练（冻结骨干网络，只训练 FPN + RPN + RoI Head）
python -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_coco128.py \
    --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_coco128 \
    --amp

# 端到端微调（同时训练骨干网络）
python -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_coco128.py \
    --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_coco128 \
    --train-backbone \
    --amp

# 多 GPU 训练
torchrun --nproc_per_node=4 -m dinov3.eval.detection.mm_train \
    dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_coco128.py \
    --work-dir ./work_dirs/faster_rcnn_dinov3_vitb16_coco128 \
    --batch-size 8 --amp
```

### 训练参数

| 参数 | 值 |
|---|---|
| 模型 | Faster R-CNN + DINOv3 ViT-B/16 |
| 数据预处理 | mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375] |
| 骨干网络输出层 | [2, 5, 8, 11] (4 层均匀分布) |
| FPN 输出通道 | 256 |
| FPN 输出层数 | 5 (stride: 16, 32, 64, 128, 256) |
| 优化器 | AdamW (lr=1e-4, weight_decay=1e-4) |
| 骨干网络学习率倍率 | 0.1 (冻结时) / 1.0 (解冻时) |
| LR 调度 | Linear warmup 100 iter + MultiStepLR (milestones=[20, 27], gamma=0.1) |
| 训练轮数 | 30 |
| Batch size | 2 |
| 混合精度 | AMP (--amp flag) |
| 梯度裁剪 | max_norm=1.0 |

## 训练结果

在 COCO128 数据集上训练 30 个 epoch 的结果：

| 指标 | 值 |
|---|---|
| bbox_mAP | 0.030 |
| bbox_mAP_50 | 0.069 |
| bbox_mAP_75 | 0.037 |

> **注意**：COCO128 仅有 108 张训练图像，且骨干网络被冻结（未进行端到端微调），因此 mAP 较低属于正常现象。在完整的 COCO 数据集上使用更大的 ViT-L/16+ 骨干网络可获得更好的结果。

## 文件说明

| 文件 | 说明 |
|---|---|
| `data/coco128/` | COCO128 数据集目录 |
| `data/coco128/convert_yolo_to_coco.py` | YOLO→COCO 格式转换脚本 |
| `data/pretrained_weights/dinov3_vitb16_timm.pth` | 转换后的预训练权重 |
| `scripts/convert_timm_to_dinov3.py` | timm→自定义格式权重转换脚本 |
| `dinov3/eval/detection/configs/faster_rcnn/faster_rcnn_dinov3_vitb16_coco128.py` | 训练配置文件 |
| `work_dirs/faster_rcnn_dinov3_vitb16_coco128/` | 训练输出目录（checkpoint、日志） |
