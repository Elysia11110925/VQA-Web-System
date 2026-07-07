# VQA · 视觉问答 Web 系统

> 北京科技大学 · 计算机与人工智能实践 · 第二周项目
>
> Visual Question Answering — 基于 ViLT 预训练模型的多模态问答系统

---

## 📋 项目概述

本项目实现了一个完整的**视觉问答（VQA）**Web 系统。用户上传一张图片并输入英文问题，AI 根据图片内容给出答案。

参考项目：[VQA-SAN](https://github.com/williamcfrancis/Visual-Question-Answering-using-Stacked-Attention-Networks)，论文：[Stacked Attention Networks (CVPR 2016)](https://arxiv.org/abs/1511.02274)。

### 核心功能

- 🖼 **图片拖拽上传**：点击或拖拽图片到上传区，实时预览
- ❓ **自然语言提问**：输入英文问题，AI 看图回答
- 📊 **Top-5 答案展示**：返回 5 个最可能的答案 + 置信度概率条
- 🤖 **ViLT 预训练模型**：已在大规模 VQA v2 数据集上训练，直接可用
- ⚡ **GPU 推理加速**：单次推理仅需 ~200ms
- ⌨️ **键盘快捷键**：`Ctrl+Enter` 提交

---

## 🖥 在线演示

启动后在浏览器打开：**http://127.0.0.1:5001**

### 推理示例

| 图片 | 问题 | 答案 | 置信度 |
|------|------|------|:--:|
| Stop Sign | What does the sign say? | **stop** | 99.75% |
| Stop Sign | What color is the sign? | **red** | — |
| COCO 街景 | What is in this picture? | **shoes / dog** | — |

---

## ⚙️ 环境配置

| 组件 | 版本 |
|------|------|
| Python | 3.14.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| transformers | 5.13.0 |
| Flask | 3.1.3 |
| opencv-python | 5.0.0 |

### 安装步骤

```bash
# 1. 安装 PyTorch (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. 安装依赖
pip install transformers flask flask-cors opencv-python Pillow

# 3. (国内用户) 设置 HuggingFace 镜像
set HF_ENDPOINT=https://hf-mirror.com
```

---

## 🚀 快速启动

```bash
git clone https://github.com/Elysia11110925/VQA-Web-System.git
cd VQA-Web-System
pip install transformers flask flask-cors opencv-python Pillow
set PYTHONIOENCODING=utf-8
set HF_ENDPOINT=https://hf-mirror.com
python app.py
```

浏览器打开 **http://127.0.0.1:5001** 即可使用。

首次启动会自动下载 ViLT 预训练模型（~350MB，仅需一次）。如遇 HuggingFace 连接问题，确保已设置 `HF_ENDPOINT=https://hf-mirror.com`。

---

## 🧠 模型方案

### 为什么不用从头训练的模型？

| 方案 | 数据量 | 准确率 | 说明 |
|------|:--:|:--:|------|
| 从头训练 Baseline | 3,779 条 | 27.82% | 只会答 yes/no，1000 分类数据严重不足 |
| **ViLT 预训练** | 44 万条 (VQA v2) | **~70%** | 直接可用，回答多样化 |

VQA 是 **1000 分类**问题（答案词表 1000 个词），需要大规模数据才能训练。正确做法是使用预训练模型，与第一周用 `bert-base-uncased` 同理。

### ViLT 模型详情

| 参数 | 值 |
|------|-----|
| 模型名称 | `dandelin/vilt-b32-finetuned-vqa` |
| 架构 | Vision-and-Language Transformer |
| 图像编码 | ViT (Vision Transformer) patch embedding |
| 文本编码 | BERT tokenizer |
| 融合方式 | Transformer 共注意力（co-attention） |
| 预训练数据 | VQA v2.0 (44 万+问答对) |
| 模型大小 | ~350MB |
| 推理速度 | GPU ~200ms / CPU ~1.5s |

### 项目中也包含（参考代码）

- `models.py`：Baseline + Stacked Attention Networks (SAN) 模型定义
- `train_local.py`：本地训练脚本（需 COCO 数据集）
- `infer_local.py`：命令行推理脚本

---

## 🏗 系统架构

```
┌──────────────────────┐    POST /api/predict     ┌──────────────────────┐
│   前端 (HTML/CSS/JS)   │ ────────────────────────►│   Flask 后端 (app.py)  │
│                       │    multipart/form-data    │                       │
│  · 图片拖拽上传+预览   │◄──────────────────────── │  · ViLT 模型推理       │
│  · 问题输入框          │    JSON (top-5 答案)      │  · GPU 加速           │
│  · 答案概率条展示      │                          │  · 图片+文本联合处理    │
└──────────────────────┘                          └──────────────────────┘
```

### API 接口

**`POST /api/predict`**

```json
// 请求: multipart/form-data
//   image: 图片文件 (PNG/JPG)
//   question: 文本问题 (英文)

// 响应:
{
  "question": "What does the sign say?",
  "top_answer": "stop",
  "top_probability": 0.9975,
  "predictions": [
    {"answer": "stop", "probability": 0.9975},
    {"answer": "hammer time", "probability": 0.0006},
    ...
  ]
}
```

---

## 📁 项目结构

```
VQA-SAN/
├── app.py                              # Flask 后端（使用 ViLT 模型）
├── templates/
│   └── index.html                      # 前端页面（图片上传+问答）
├── infer_local.py                      # 命令行推理脚本
├── train_local.py                      # 本地训练脚本
├── models.py                           # Baseline + SAN 模型定义
├── data_loader.py                      # VQA 数据加载器
├── dataset/
│   ├── vocab_questions.txt             # 问题词表（17,856词）
│   ├── vocab_answers.txt               # 答案词表（1,000词）
│   └── Resized_Images/
│       └── test_img.jpeg               # Stop Sign 测试图
├── utils/
│   └── text_helper.py                  # 文本处理工具
├── README.md                           # 本文件
└── 实验记录.md                          # 详细实验记录
```

---

## 🐛 已知问题与解决

| # | 问题 | 原因 | 解决方案 |
|---|------|------|----------|
| 1 | 从头训练只会答 yes/no | 训练数据太少（3,779条），1000分类 | 换用 ViLT 预训练模型 |
| 2 | 图片尺寸不一致导致 batch 报错 | COCO 图片尺寸各异 | 添加 `transforms.Resize((224,224))` |
| 3 | .npy 数据路径指向 Google Drive | 原项目为 Colab 设计 | 重新生成本地路径子集 |
| 4 | ViLT 下载走 huggingface.co 超时 | 国内网络限制 | 使用 hf-mirror.com 镜像 |
| 5 | 训练时 Python 输出被缓冲 | stdout 缓冲 | 使用 `py -u` 无缓冲模式 |
| 6 | 扩数据到 3,779 条仅提升 1% | 1000 分类数据需求远超预期 | 放弃从头训练，改用预训练模型 |

---

## 📊 实验记录

详细训练日志、模型对比和问题排查见 [实验记录.md](./实验记录.md)。

---

## 📚 参考资料

- [Stacked Attention Networks (CVPR 2016)](https://arxiv.org/abs/1511.02274) — SAN 论文
- [ViLT](https://arxiv.org/abs/2102.03334) — Vision-and-Language Transformer
- [VQA v2 Dataset](https://visualqa.org/) — 官方数据集
- [VQA-SAN 参考项目](https://github.com/williamcfrancis/Visual-Question-Answering-using-Stacked-Attention-Networks) — 原始实现

---

> **作者**: Elysia11110925  
> **日期**: 2026年7月  
> **课程**: 北京科技大学 · 计算机与人工智能实践 · 暑期小学期
