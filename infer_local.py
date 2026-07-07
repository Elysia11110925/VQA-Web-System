# -*- coding: utf-8 -*-
"""VQA 单图片+单问题推理脚本"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import VqaModel, SANModel
from utils.text_helper import VocabDict, load_str_list

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE, 'dataset')
MODEL_PATH = os.path.join(BASE, 'models_saved', 'best_model.pt')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载词表
qst_vocab = VocabDict(os.path.join(DATASET, 'vocab_questions.txt'))
ans_vocab = load_str_list(os.path.join(DATASET, 'vocab_answers.txt'))
print(f'Question vocab: {qst_vocab.vocab_size}, Answer vocab: {len(ans_vocab)}')

# 加载模型
model = VqaModel(
    embed_size=1024, qst_vocab_size=qst_vocab.vocab_size,
    ans_vocab_size=len(ans_vocab), word_embed_size=300,
    num_layers=2, hidden_size=512
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print('Model loaded OK')

# 图片预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def predict(image_path, question, top_k=5):
    """输入图片路径和问题，返回 top-k 答案"""
    # 处理图片
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, 224, 224]

    # 处理问题
    max_qst_length = 30
    qst_tokens = question.lower().strip().split()
    qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)
    qst2idc[:len(qst_tokens)] = [qst_vocab.word2idx(w) for w in qst_tokens]
    qst_tensor = torch.from_numpy(qst2idc).long().unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        output = model(image, qst_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    results = []
    for i in range(top_k):
        ans = ans_vocab[top_indices[0][i].item()]
        prob = top_probs[0][i].item()
        results.append((ans, prob))

    return results


if __name__ == '__main__':
    # 测试
    test_img = os.path.join(DATASET, 'Resized_Images', 'test_img.jpeg')
    if os.path.exists(test_img):
        print(f'\nTest image: {test_img}')
        print('Question: What does the sign say?')
        results = predict(test_img, 'What does the sign say?')
        print('\nTop-5 predictions:')
        for ans, prob in results:
            print(f"  '{ans}' - {prob:.4f}")
    else:
        print(f'Test image not found: {test_img}')
        # 用一张已下载的COCO图片测试
        val_dir = os.path.join(DATASET, 'Resized_Images', 'val2014')
        if os.path.exists(val_dir):
            imgs = os.listdir(val_dir)
            if imgs:
                test_img = os.path.join(val_dir, imgs[0])
                print(f'\nUsing: {test_img}')
                results = predict(test_img, 'What is in the picture?')
                print('\nTop-5 predictions:')
                for ans, prob in results:
                    print(f"  '{ans}' - {prob:.4f}")
