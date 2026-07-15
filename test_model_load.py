"""Test loading the trained SAN model"""
import torch
import sys
sys.path.insert(0, '.')
from models import SANModel

print('Testing model load...')
model = torch.load('models_saved/best_model.pt', map_location='cpu', weights_only=False)
model.eval()
total = sum(p.numel() for p in model.parameters())
print(f'Model loaded successfully. Params: {total:,}')
print(f'Type: {type(model).__name__}')
print(f'Embed size: {model.img_encoder.fc[0].out_features}')
print(f'Answer vocab size: {model.mlp[1].out_features}')

# Test inference with vocab
from utils.text_helper import VocabDict, load_str_list, tokenize
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

qst_vocab = VocabDict('dataset/vocab_questions.txt')
ans_vocab = load_str_list('dataset/vocab_answers.txt')
print(f'Qst vocab: {qst_vocab.vocab_size}, Ans vocab: {len(ans_vocab)}')

# Test on a real image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

image = Image.open('dataset/Resized_Images/test_img.jpeg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

tokens = tokenize('what does the sign say')
max_qst_len = 30
qst_ids = np.array([qst_vocab.word2idx('<pad>')] * max_qst_len)
qst_ids[:len(tokens)] = [qst_vocab.word2idx(w) for w in tokens]
qst_tensor = torch.from_numpy(qst_ids).long().unsqueeze(0)

with torch.no_grad():
    output = model(image_tensor, qst_tensor)
    probs = torch.softmax(output, dim=1)
    top_probs, top_indices = torch.topk(probs, k=5, dim=1)

print('\nTop-5 predictions for "what does the sign say?":')
for i in range(5):
    idx = top_indices[0][i].item()
    ans = ans_vocab[idx]
    prob = top_probs[0][i].item()
    print(f'  {i+1}. {ans:20s} {prob:.4f}')
