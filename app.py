# -*- coding: utf-8 -*-
"""VQA Web 服务 — 第二周项目
Flask 后端：接收图片+问题，调用 ViLT 预训练模型返回答案
"""
import os, sys, warnings
warnings.filterwarnings('ignore')

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import ViltProcessor, ViltForQuestionAnswering

# ============================================================
# Flask 初始化
# ============================================================
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
CORS(app)

BASE = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型（启动时加载）
model = None
processor = None


def load_model():
    global model, processor
    print('Loading ViLT VQA model from hf-mirror.com...')
    processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
    model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa')
    model = model.to(DEVICE)
    model.eval()
    print(f'Model loaded. Device: {DEVICE}')


def predict(image_file, question):
    image = Image.open(image_file).convert('RGB')
    encoding = processor(image, question, return_tensors='pt')
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5, dim=-1)

    results = []
    for i in range(5):
        idx = top_indices[0][i].item()
        ans = model.config.id2label[idx]
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
    print('Loading ViLT VQA model...')
    load_model()
    print(f'\nServer: http://127.0.0.1:5001')
    app.run(host='127.0.0.1', port=5001, debug=False)
