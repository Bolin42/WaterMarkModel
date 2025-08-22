# WaterMarkModel: 基于扩散模型的水印嵌入与提取系统

## 项目概述
WaterMarkModel是一个结合扩散扩散模型（Stable Diffusion）实现数字水印嵌入与提取的系统。该系统通过训练编码器将水印信息嵌入生成图像中，并通过解码器从图像中提取水印，实现图像所有权的验证与保护。

## 核心功能
- 基于Stable Diffusion的图像生成与水印嵌入
- 水印解码器实现从图像中提取水印信息
- 完整的训练、验证与测试流程
- 自动生成训练报告与评估指标

## 环境配置

### 依赖项
项目依赖以下关键库（完整列表见`requirements.txt`）：
- Python 3.x
- PyTorch 2.1.0
- torchvision 0.16.0
- diffusers 0.21.4
- transformers 4.34.1
- 其他：Pillow, tqdm, scipy, numpy等

### 安装方法
```bash
pip install -r requirements.txt
```

## 数据准备

### 提示词数据
- 训练提示词：`data/prompts.txt`（10条提示词）
- 验证提示词：`data/validation_prompts.txt`（5条提示词）

提示词示例：
- "a majestic eagle soaring through a mountain range"
- "a high-quality photograph of a majestic lion in the savannah"

## 模型架构

### 关键组件
1. **WatermarkEncoder**：线性层结构
   - 输入：水印信息（长度128）
   - 输出：潜在空间特征（4096维度）
   - 可训练参数：2,265,856

2. **WatermarkDecoder**：负责从图像中提取水印
   - 可训练参数：708,704

3. **基础模型**：Stable Diffusion v1.5 (`AI-ModelScope/stable-diffusion-v1-5`)

## 训练配置

### 核心参数
- 图像分辨率：256x256
- 批处理大小：1
- 学习率：0.0001
- 优化器：AdamW（β1=0.9, β2=0.999, 权重衰减=0.01）
- 训练轮次：10
- 梯度累积步数：8
- 损失函数：MSE（权重1.0）

### 训练过程
```bash
# 训练命令示例
python train.py
```

训练过程中会自动生成：
- 训练报告：`logs/detailed_training_report.txt`
- 损失曲线：`training_loss.png` 和 `validation_loss.png`
- 学习率曲线：`learning_rate.png`

## 评估指标

### 关键指标
1. **不可见性评估**
   - PSNR（峰值信噪比）：值越高表示水印对图像质量影响越小
   - SSIM（结构相似性）：值越接近1表示水印对图像结构影响越小

2. **鲁棒性评估**
   - BER（比特错误率）：值越低表示水印提取准确性越高
   - Accuracy（准确率）：值越高表示水印提取效果越好

### 实验结果
- 正常对比实验平均结果：
  - PSNR: 16.3450 dB
  - SSIM: 0.5522
  - BER: 0.4990
  - Accuracy: 0.5010

## 目录结构
```
WaterMarkModel/
├── data/                  # 提示词数据
│   ├── prompts.txt        # 训练提示词
│   └── validation_prompts.txt # 验证提示词
├── logs/                  # 训练日志
│   └── detailed_training_report.txt # 训练报告
├── output/                # 实验结果
│   └── 正常对比实验/      # 对比实验结果
├── results/               # 测试结果
│   └── testing/           # 测试运行结果
├── requirements.txt       # 依赖列表
└── train.py               # 训练脚本
```

## 注意事项
- 训练需要GPU支持（推荐NVIDIA A10及以上）
- 目前未实现FID、IS、LPIPS等图像质量评估指标
- 未实现水印在各种攻击下的鲁棒性测试

## 未来改进方向
1. 增加水印鲁棒性测试（抗压缩、裁剪、滤波等攻击）
2. 实现更多图像质量评估指标
3. 优化编码器/解码器结构以提升水印不可见性与鲁棒性
4. 增加学习率衰减策略和预热步骤
5. 支持更大分辨率图像的水印处理
