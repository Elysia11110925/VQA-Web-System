# -*- coding: utf-8 -*-
"""SAN 模型推理演示脚本
用于验证课前展示：SAN 堆叠注意力网络的完整推理流程。
模型未在大规模数据上训练，结果随机——这正是 Web 系统选用 ViLT 预训练模型的原因。
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# ── 路径 ──
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from models import VqaModel, SANModel
from utils.text_helper import VocabDict, load_str_list

DATASET = os.path.join(BASE, 'dataset')
MODEL_SAVE = os.path.join(BASE, 'models_saved', 'san_demo.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEP = '=' * 60

# ══════════════════════════════════════════════════════════════
# 1. 加载词表
# ══════════════════════════════════════════════════════════════
print(SEP)
print('SAN 堆叠注意力网络 · 推理演示')
print(SEP)

qst_vocab = VocabDict(os.path.join(DATASET, 'vocab_questions.txt'))
ans_vocab = load_str_list(os.path.join(DATASET, 'vocab_answers.txt'))
print(f'\n[词表] 问题词表: {qst_vocab.vocab_size} 词  |  答案词表: {len(ans_vocab)} 词')

# ══════════════════════════════════════════════════════════════
# 2. 构建 SAN 模型
# ══════════════════════════════════════════════════════════════
print(f'\n[模型] 构建 SAN (Stacked Attention Networks) ...')
print(f'  设备: {DEVICE}')

model = SANModel(
    embed_size=1024,
    qst_vocab_size=qst_vocab.vocab_size,
    ans_vocab_size=len(ans_vocab),
    word_embed_size=300,
    num_layers=2,
    hidden_size=512
).to(DEVICE)

# 统计参数量
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'  总参数: {total:,}  |  可训练: {trainable:,}')

# 打印模型结构
print(f'\n[结构] SAN 模型组件:')
print(f'  ├─ ImgAttentionEncoder (VGG19 CNN → 196区域 × {1024}维)')
print(f'  ├─ QstEncoder (Embedding → 2层LSTM → {1024}维)')
print(f'  ├─ Attention Layer 1 (512通道, 问题→图像粗定位)')
print(f'  └─ Attention Layer 2 (512通道, 精确定位答案区域)')

# 如果之前存过权重就加载，否则用随机初始化（演示用）
if os.path.exists(MODEL_SAVE):
    model.load_state_dict(torch.load(MODEL_SAVE, map_location=DEVICE))
    print(f'\n  已加载保存的权重: {MODEL_SAVE}')
else:
    print(f'\n  [!] 未找到训练权重，使用随机初始化（演示模型架构）')
    print(f'  --> 推理结果将是随机的，这就是为什么 Web 系统选用了 ViLT 预训练模型')

model.eval()

# ══════════════════════════════════════════════════════════════
# 3. 图片预处理
# ══════════════════════════════════════════════════════════════
print(f'\n[预处理] 图片变换: Resize(224,224) → ToTensor → Normalize(ImageNet)')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ══════════════════════════════════════════════════════════════
# 4. 推理函数
# ══════════════════════════════════════════════════════════════
def predict(image_path, question, top_k=5):
    """SAN 模型推理：图片 + 问题 → Top-K 答案"""
    # 处理图片
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 处理问题（分词 → 查词表 → padding）
    max_qst_len = 30
    tokens = question.lower().strip().split()
    qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_len)
    qst2idc[:len(tokens)] = [qst_vocab.word2idx(w) for w in tokens]
    qst_tensor = torch.from_numpy(qst2idc).long().unsqueeze(0).to(DEVICE)

    print(f'  图片: {os.path.basename(image_path)}')
    print(f'  问题: "{question}"')
    print(f'  问题分词: {tokens}')
    print(f'  编码ID: {qst2idc[:len(tokens)+2]}... (pad到{max_qst_len})')

    # 推理
    with torch.no_grad():
        output = model(image_tensor, qst_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    results = []
    for i in range(top_k):
        ans = ans_vocab[top_indices[0][i].item()]
        prob = top_probs[0][i].item()
        results.append((ans, prob))

    return results


# ══════════════════════════════════════════════════════════════
# 5. 执行演示推理
# ══════════════════════════════════════════════════════════════
test_img = os.path.join(DATASET, 'Resized_Images', 'test_img.jpeg')

if not os.path.exists(test_img):
    print(f'\n❌ 测试图片不存在: {test_img}')
    sys.exit(1)

questions = [
    'What does the sign say?',
    'What color is the sign?',
]

for q in questions:
    print(f'\n{"─"*50}')
    results = predict(test_img, q)
    print(f'\n  Top-5 预测结果:')
    for rank, (ans, prob) in enumerate(results, 1):
        bar = '█' * int(prob * 40)
        print(f'    {rank}. {ans:20s}  {prob:.4f}  {bar}')

# ══════════════════════════════════════════════════════════════
# 6. 结论
# ══════════════════════════════════════════════════════════════
print(f'\n{SEP}')
print('说明')
print(SEP)
print('''
以上是 SAN (Stacked Attention Networks) 模型的完整推理流程演示：

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

当前模型未在大规模 VQA v2 数据集上训练（需要44万条数据），
因此上述预测结果没有实际意义。

这正是本项目的核心经验：
  - SAN 架构精巧，但 VQA 是 1000 分类任务，需要海量数据
  - 用 3,779 条数据训练的结果只有 27% 准确率，只会答 yes/no
  - Web 系统因此选用了 ViLT 预训练模型（在完整 VQA v2 上训练）

论文: Stacked Attention Networks for Image Question Answering (CVPR 2016)
      https://arxiv.org/abs/1511.02274
项目: https://github.com/Elysia11110925/VQA-Web-System
''')
