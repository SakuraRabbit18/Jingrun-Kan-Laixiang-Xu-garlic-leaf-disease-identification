# 大蒜叶部病害识别系统

基于改进 ResNet18 的大蒜叶部病害识别系统，通过深度学习技术实现对大蒜叶部病害的自动分类与识别。

## 项目概述

本项目旨在利用深度学习技术，构建一个高效的大蒜叶部病害识别模型。通过改进的 ResNet18 网络架构，结合注意力机制（CBAM 和 Triplet Attention），实现对大蒜叶部常见病害的自动识别，为农业生产提供病害早期检测与诊断支持。

## 项目结构

```plaintext
Jingrun-Kan-Laixiang-Xu-garlic-leaf-disease-identification/
├── train_MyResnet18.py        # 模型训练主脚本
├── MyResBet18.py              # 改进的ResNet18模型定义
├── Triplet.py                 # Triplet Attention模块实现
├── CBAM.py                    # CBAM注意力机制实现
├── Pconv.py                   # 部分卷积实现
├── LeafDataset.py             # 数据集加载类
├── inference.py               # 模型推理脚本
└── split_dataset/             # 数据集目录
    ├── train/                 # 训练集
    ├── val/                   # 验证集
    └── test/                  # 测试集
```

## 核心技术

1. **改进的 ResNet18**：在原始 ResNet18 基础上进行改进，引入注意力机制

2. 注意力机制

   ：

   - CBAM (Convolutional Block Attention Module)
   - Triplet Attention

3. **部分卷积**：减少计算量，提高模型效率

4. **迁移学习**：使用预训练的 ResNet18 权重作为初始值

## 推荐环境

- Python 3.9
- PyTorch ==2.3.1
- torchvision==0.18.1

## 数据集准备

数据集应按照以下结构组织：

```plaintext
split_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

每个类别文件夹中包含对应类别的大蒜叶片图像。

## 模型训练

使用train_MyResnet18.py脚本进行模型训练：

```bash
python train_MyResnet18.py
```

训练参数说明：

- 批处理大小：4
- 训练轮次：100
- 图像尺寸：224x224
- 优化器：Adam
- 学习率：采用可变学习率策略
- 数据增强：包括 Resize、ToTensor 和 Normalize

训练过程中会自动保存模型权重，并在训练结束后在测试集上评估模型性能。

如果需要调整超参数，请自行到`train_MyResnet18.py`中调整相关参数。

## 模型推理

使用提供的推理脚本对新图像进行预测：

### 单张图像预测

```bash
python inference.py --model_path myResNet18_bs32_ep100_224.pth --image_path .\split_dataset\test\Blight\20.jpg 
```

### 批量图像预测

```bash
python inference.py --model_path myResNet18_bs32_ep100_224.pth --image_dir test_images/
```

推理脚本会输出预测的类别及对应的置信度。

## 性能评估

训练完成后，模型会在测试集上进行评估，输出准确率等关键指标。用户也可以根据需要在`train_function.py`中扩展评估指标。