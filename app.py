from flask import Flask, redirect, url_for, request, render_template, app,jsonify
import re
import torch
import os
import argparse
from model import DeepPunctuation, DeepPunctuationCRF
from config import *
from werkzeug.utils import secure_filename


app = Flask(__name__)



@app.route('/',methods=['GET'])
def home():
	return jsonify({'message':'it works'})



@app.route('/puncuate',methods=['POST'])
def upload():
	if request.method == 'POST':
		pretrained_model = 'xlm-roberta-large'
		use_crf = False
		cuda=True
		lstm_dim = -1
		# tokenizer
		tokenizer = MODELS[pretrained_model][1].from_pretrained(pretrained_model)
		token_style = MODELS[pretrained_model][3]
		json_data=request.json
		text=request.json.get('text')
		# logs
		model_save_path = 'xlm-roberta-large-bn.pt'

		# Model
		device = torch.device('cuda' if (cuda and torch.cuda.is_available()) else 'cpu')
		if use_crf:
			deep_punctuation = DeepPunctuationCRF(pretrained_model, freeze_bert=False, lstm_dim=lstm_dim)
		else:
			deep_punctuation = DeepPunctuation(pretrained_model, freeze_bert=False, lstm_dim=lstm_dim)
		deep_punctuation.to(device)
		sequence_length=256
		language='bn'
		deep_punctuation.load_state_dict(torch.load(model_save_path, map_location='cpu'))
		deep_punctuation.eval()

		text = re.sub(r"[,:\-–.!;?]", '', text)
		words_original_case = text.split()
		words = text.lower().split()

		word_pos = 0
		sequence_len = sequence_length
		result = ""
		decode_idx = 0
		punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
		if language != 'en':
			punctuation_map[2] = '।'

		while word_pos < len(words):
			x = [TOKEN_IDX[token_style]['START_SEQ']]
			y_mask = [0]

			while len(x) < sequence_len and word_pos < len(words):
				tokens = tokenizer.tokenize(words[word_pos])
				if len(tokens) + len(x) >= sequence_len:
					break
				else:
					for i in range(len(tokens) - 1):
						x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
						y_mask.append(0)
					x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
					y_mask.append(1)
					word_pos += 1
			x.append(TOKEN_IDX[token_style]['END_SEQ'])
			y_mask.append(0)
			if len(x) < sequence_len:
				x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
				y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
			attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

			x = torch.tensor(x).reshape(1, -1)
			y_mask = torch.tensor(y_mask)
			attn_mask = torch.tensor(attn_mask).reshape(1, -1)
			x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

			with torch.no_grad():
				if use_crf:
					y = torch.zeros(x.shape[0])
					y_predict = deep_punctuation(x, attn_mask, y)
					y_predict = y_predict.view(-1)
				else:
					y_predict = deep_punctuation(x, attn_mask)
					y_predict = y_predict.view(-1, y_predict.shape[2])
					y_predict = torch.argmax(y_predict, dim=1).view(-1)
			for i in range(y_mask.shape[0]):
				if y_mask[i] == 1:
					result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
					decode_idx += 1


		return jsonify({'text':result})

if __name__ == '__main__':
	app.run(debug=True,port=8080)