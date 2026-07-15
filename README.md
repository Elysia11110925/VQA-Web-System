# VQA · 视觉问答 Web 系统

> 北京科技大学 · 计算机与人工智能实践 · 第二周项目
>
> Visual Question Answering — 基于 SAN 堆叠注意力网络的多模态问答系统

---

## 📋 项目概述

本项目实现了一个完整的**视觉问答（VQA）**Web 系统。用户上传一张图片并输入英文问题，AI 根据图片内容给出答案。

基于论文：[Stacked Attention Networks (CVPR 2016)](https://arxiv.org/abs/1511.02274)，使用 VGG19 图像编码器 + LSTM 问题编码器 + 2层堆叠注意力机制，在 VQA v2 数据集上训练 5 个 epoch。

### 核心功能

- 🖼 **图片拖拽上传**：点击或拖拽图片到上传区，实时预览
- ❓ **自然语言提问**：输入英文问题，AI 看图回答
- 📊 **Top-5 答案展示**：返回 5 个最可能的答案 + 置信度概率条
- 🧠 **SAN 堆叠注意力模型**：VGG19 + LSTM + 2层 Stacked Attention
- ⚡ **GPU 推理加速**
- ⌨️ **键盘快捷键**：`Ctrl+Enter` 提交

---

## 🖥 在线演示

启动后在浏览器打开：**http://127.0.0.1:5001**

### 推理示例

| 图片 | 问题 | 预期 |
|------|------|------|
| Stop Sign | What does the sign say? | stop / red / white |
| Stop Sign | What color is the sign? | red / white |
| COCO 街景 | What is in this picture? | — |

---

## ⚙️ 环境配置

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.13.0 |
| torchvision | 0.28.0 |
| CUDA | 12.x |
| Flask | 3.x |
| Pillow | 12.3.0 |
| opencv-python | 5.0.0 |

### 安装步骤

```bash
# 1. 创建 conda 环境
conda create -n vqa python=3.10
conda activate vqa

# 2. 安装依赖
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu118
pip install flask flask-cors opencv-python pillow
```

---

## 🚀 快速启动

```bash
git clone https://github.com/Elysia11110925/VQA-Web-System.git
cd VQA-Web-System
conda activate vqa
python app.py
```

浏览器打开 **http://127.0.0.1:5001** 即可使用。

---

## 🧠 模型方案

### SAN (Stacked Attention Networks)

| 参数 | 值 |
|------|-----|
| 图像编码器 | VGG19 (ImageNet pretrained, features only) |
| 图像特征 | 196 区域 × 1024 维 |
| 问题编码器 | Embedding + 2层 LSTM (hidden=64) |
| 注意力层数 | 2 层 Stacked Attention |
| 答案分类 | 1000 类 (VQA v2 高频答案) |
| 训练数据 | VQA v2 (train2014 + val2014) |
| 训练轮数 | 5 epochs |
| 训练准确率 | Train: 29.8%, Valid: 33.2% |

### 架构流程

```
图片 → VGG19 CNN → 196个图像区域特征 (每区域1024维)
                                    ↓
问题 → Embedding → 2层LSTM → 问题特征 (1024维)
                                    ↓
        ┌─ Attention Layer 1 ─┐
        │  问题特征 扫描 196区域  │ → 大致相关区域 → u1
        └──────────────────────┘
                    ↓
        ┌─ Attention Layer 2 ─┐
        │  u1 再次扫描 196区域   │ → 精确定位答案区域 → u2
        └──────────────────────┘
                    ↓
              MLP → 1000类答案分布
```

---

## 📁 项目结构

```
VQA-SAN/
├── app.py                              # Flask 后端（SAN 模型推理）
├── models.py                           # SAN / Baseline 模型定义
├── data_loader.py                      # VQA 数据加载器
├── templates/
│   └── index.html                      # 前端页面（图片上传+问答）
├── infer_san.py                        # 命令行推理演示脚本
├── train_local.py                      # 本地训练脚本
├── infer_local.py                      # 命令行推理脚本
├── test_model_load.py                  # 模型加载测试脚本
├── dataset/
│   ├── vocab_questions.txt             # 问题词表（3,853词）
│   ├── vocab_answers.txt               # 答案词表（1,000词）
│   └── Resized_Images/
│       └── test_img.jpeg               # Stop Sign 测试图
├── models_saved/
│   └── best_model.pt                   # 训练好的 SAN 模型（5 epoch，~92MB）
├── utils/
│   └── text_helper.py                  # 文本处理工具（VocabDict, tokenize）
├── requirements.txt                    # 依赖列表
├── README.md                           # 本文件
└── 实验记录.md                          # 详细实验记录
```

---

## 📚 参考资料

- [Stacked Attention Networks (CVPR 2016)](https://arxiv.org/abs/1511.02274) — SAN 论文
- [VQA v2 Dataset](https://visualqa.org/) — 官方数据集
- [VQA-SAN 参考项目](https://github.com/williamcfrancis/Visual-Question-Answering-using-Stacked-Attention-Networks) — 原始实现

---

> **作者**: Elysia11110925
> **日期**: 2026年7月
> **课程**: 北京科技大学 · 计算机与人工智能实践 · 暑期小学期
