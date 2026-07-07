# -*- coding: utf-8 -*-
"""VQA Baseline 模型本地训练脚本（适配子集数据）"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import VqaDataset
from models import VqaModel, SANModel
from utils.text_helper import VocabDict

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, 'dataset')
MODEL_DIR = os.path.join(BASE, 'models_saved')
LOG_DIR = os.path.join(BASE, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================
# 超参数
# ============================================================
MAX_QST_LENGTH = 30
MAX_NUM_ANS = 10
EMBED_SIZE = 1024
WORD_EMBED_SIZE = 300
NUM_LAYERS = 2
HIDDEN_SIZE = 512
LEARNING_RATE = 0.001
STEP_SIZE = 10
GAMMA = 0.1
NUM_EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 0  # Windows 下必须为 0
SAVE_STEP = 5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ============================================================
# 数据加载
# ============================================================
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

train_dataset = VqaDataset(
    input_dir=DATASET,
    input_vqa='train_subset.npy',
    max_qst_length=MAX_QST_LENGTH,
    max_num_ans=MAX_NUM_ANS,
    transform=transform
)
valid_dataset = VqaDataset(
    input_dir=DATASET,
    input_vqa='valid_subset.npy',
    max_qst_length=MAX_QST_LENGTH,
    max_num_ans=MAX_NUM_ANS,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

qst_vocab_size = train_dataset.qst_vocab.vocab_size
ans_vocab_size = train_dataset.ans_vocab.vocab_size
print(f'Qst vocab: {qst_vocab_size}, Ans vocab: {ans_vocab_size}')
print(f'Train: {len(train_dataset)} samples, Valid: {len(valid_dataset)} samples')

# ============================================================
# 模型初始化
# ============================================================
model = VqaModel(
    embed_size=EMBED_SIZE,
    qst_vocab_size=qst_vocab_size,
    ans_vocab_size=ans_vocab_size,
    word_embed_size=WORD_EMBED_SIZE,
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

# 只训练部分参数（冻结 VGG）
params = (
    list(model.img_encoder.fc.parameters()) +
    list(model.qst_encoder.parameters()) +
    list(model.fc1.parameters()) +
    list(model.fc2.parameters())
)
optimizer = optim.Adam(params, lr=LEARNING_RATE)
scheduler = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ============================================================
# 训练循环
# ============================================================
best_acc = 0.0
best_model_path = os.path.join(MODEL_DIR, 'best_model.pt')

for epoch in range(NUM_EPOCHS):
    for phase in ['train', 'valid']:
        if phase == 'train':
            scheduler.step()
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = valid_loader

        running_loss = 0.0
        running_corr = 0
        batch_count = len(loader)

        for batch_idx, batch in enumerate(loader):
            image = batch['image'].to(DEVICE)
            question = batch['question'].to(DEVICE)
            label = batch['answer_label'].to(DEVICE)
            multi_choice = batch['answer_multi_choice']

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                output = model(image, question)
                _, pred = torch.max(output, 1)
                loss = criterion(output, label)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item()
            running_corr += torch.stack(
                [(ans == pred.cpu()) for ans in multi_choice]
            ).any(dim=0).sum()

            if batch_idx % 100 == 0:
                print(f'| {phase.upper():5s} | Epoch [{epoch+1:02d}/{NUM_EPOCHS}] '
                      f'Step [{batch_idx:04d}/{batch_count}] Loss: {loss.item():.4f}')

        epoch_loss = running_loss / batch_count
        epoch_acc = running_corr.double() / len(loader.dataset)

        print(f'| {phase.upper():5s} | Epoch [{epoch+1:02d}/{NUM_EPOCHS}] '
              f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

        # 保存最佳模型
        if phase == 'valid' and epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'  -> Best model saved! Acc: {best_acc:.4f}')

print(f'\nTraining done. Best val acc: {best_acc:.4f}')
print(f'Model saved to: {best_model_path}')
