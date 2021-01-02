import torch
from transformers import BertJapaneseTokenizer, BertForMaskedLM, BertConfig
import numpy as np
import pandas as pd
from copy import deepcopy

class Tokenizer():
	def __init__(self, model_file):
		self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_file)

	# text → ids
	def encode(self, text):
		return self.tokenizer.encode(text)

	# text → 分かち書き (tokens)
	def wakati(self, text):
		ids = self.encode(text)
		return self.tokenizer.convert_ids_to_tokens(ids)

	# tokens → ids
	def encode_tokens2ids(self, tokens):
		ids = self.tokenizer.convert_tokens_to_ids(tokens)
		return torch.tensor([ids])

class BERT():
	def __init__(self, model_file, tokenizer):
		self.model = BertForMaskedLM.from_pretrained(model_file)
		self.tokenizer = tokenizer

	def predict_mask(self, tokens_tensor, masked_index):
		self.model.eval()
		with torch.no_grad():
			outputs = self.model(tokens_tensor)
			predictions = outputs[0]
			_, predicted_indexes = torch.topk(predictions[0, masked_index], k=5)
			predicted_tokens = self.tokenizer.tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
			return predicted_tokens

class DataSet:
	def convert_for_bert(self, csv_file, tokenizer):
		data = pd.read_csv(csv_file, header=0)
		outputs = []
		for _, row in data.iterrows():
			wakati = tokenizer.wakati(row[0])
			masked = deepcopy(wakati)
			masked[row[1]] = '[MASK]'
			tokens_tensor = tokenizer.encode_tokens2ids(masked)
			outputs.append({'masked_index': row[1], 
							'original_wakati': wakati, 
							'masked_wakati': masked, 
							'input_tokens': tokens_tensor})
		return outputs


def test():
	model_file = 'cl-tohoku/bert-base-japanese-whole-word-masking'
	tokenizer = Tokenizer(model_file)
	bert = BERT(model_file, tokenizer)

	text = "正月にはこまを投げて遊びましょう"
	ids = tokenizer.encode(text)
	wakati = tokenizer.wakati(text)
	print(wakati)
	masked_index = 4
	wakati[masked_index] = '[MASK]'
	print(wakati)

	tokens_tensor = tokenizer.encode_tokens2ids(wakati)
	rank = bert.predict_mask(tokens_tensor, masked_index)
	print(rank)


def main():
	model_file = 'cl-tohoku/bert-base-japanese-whole-word-masking'
	data_file = 'input.csv'
	tokenizer = Tokenizer(model_file)
	bert = BERT(model_file, tokenizer)
	DS = DataSet()

	datas = DS.convert_for_bert(data_file, tokenizer)
	for data in datas:
		ranking = bert.predict_mask(data['input_tokens'], data['masked_index'])
		print('--------------')
		print(data['original_wakati'])
		print(data['masked_wakati'])
		print('Candidate: ', ranking)

if __name__ == '__main__':
	main()