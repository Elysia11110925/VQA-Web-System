# -*- coding: utf-8 -*-
"""VQA Web 服务 — 第二周项目
Flask 后端：接收图片+问题，调用 SAN 堆叠注意力网络模型返回答案
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ============================================================
# 路径 & 设备
# ============================================================
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from models import SANModel
from utils.text_helper import VocabDict, load_str_list, tokenize

DATASET = os.path.join(BASE, 'dataset')
MODEL_PATH = os.path.join(BASE, 'models_saved', 'best_model.pt')
QST_VOCAB_PATH = os.path.join(DATASET, 'vocab_questions.txt')
ANS_VOCAB_PATH = os.path.join(DATASET, 'vocab_answers.txt')
MAX_QST_LENGTH = 30

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Flask 初始化
# ============================================================
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)

# ============================================================
# 图片预处理
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# ============================================================
# 全局模型 & 词表
# ============================================================
model = None
qst_vocab = None
ans_vocab = None


def load_model():
    """加载训练好的 SAN 模型和词表"""
    global model, qst_vocab, ans_vocab

    print('Loading vocabularies...')
    qst_vocab = VocabDict(QST_VOCAB_PATH)
    ans_vocab = load_str_list(ANS_VOCAB_PATH)
    print(f'  Question vocab: {qst_vocab.vocab_size} words')
    print(f'  Answer vocab:   {len(ans_vocab)} answers')

    print(f'Loading SAN model from {MODEL_PATH}...')
    model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f'  Model loaded. Device: {DEVICE}, Parameters: {total:,}')
    print(f'  Architecture: SAN (VGG19 + LSTM + 2-layer Stacked Attention)')


def predict(image_file, question, top_k=5):
    """SAN 模型推理：图片 + 问题 → Top-K 答案"""

    # 图片预处理
    image = Image.open(image_file).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 问题预处理（分词 → 查词表 → padding到30）
    tokens = tokenize(question)
    qst_ids = np.array([qst_vocab.word2idx('<pad>')] * MAX_QST_LENGTH)
    qst_ids[:len(tokens)] = [qst_vocab.word2idx(w) for w in tokens]
    qst_tensor = torch.from_numpy(qst_ids).long().unsqueeze(0).to(DEVICE)

    # 推理
    with torch.no_grad():
        output = model(image_tensor, qst_tensor)
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)

    results = []
    for i in range(top_k):
        idx = top_indices[0][i].item()
        ans = ans_vocab[idx]
        prob = round(top_probs[0][i].item(), 4)
        results.append({'answer': ans, 'probability': prob})

    return results


# ============================================================
# API 路由
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({'error': 'Please enter a question'}), 400

    try:
        results = predict(image_file, question)
        return jsonify({
            'question': question,
            'predictions': results,
            'top_answer': results[0]['answer'],
            'top_probability': results[0]['probability']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# 启动
# ============================================================
if __name__ == '__main__':
    print('=' * 60)
    print('VQA Web System - SAN (Stacked Attention Networks)')
    print('=' * 60)
    load_model()
    print(f'\nServer running at: http://127.0.0.1:5001')
    print('Press Ctrl+C to stop.\n')
    app.run(host='127.0.0.1', port=5001, debug=False)
