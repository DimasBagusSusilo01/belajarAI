import torch
from torch.utils.data import Dataset
import pandas as pd
from tokenizer import Tokenizer
from config_rwakuv import *

class ChatDataset(Dataset):
	def __init__(self, csv_file, vocab_file):
		self.df = pd.read_csv(csv_file)
		self.tokenizer = Tokenizer()
		self.tokenizer.load(vocab_file)
		self.samples = []
		for _, row in self.df.iterrows():
			text = str(row["text"])
			respon = str(row["respon"])
			emosi = str(row["label"])
			dialog = (
				"<USER> "
				+ text
				+ " <BOT> "
				+ respon
				+ " <EMOTION> "
				+ emosi)
				
			encoded = self.tokenizer.encode(dialog)

			input_ids = encoded[:-1]
			target_ids = encoded[1:]

			bot_token = self.tokenizer.word2idx["<BOT>"]
			bot_index = encoded.index(bot_token)

			for i in range(bot_index):
				target_ids[i] = -100
			input_ids = input_ids[:MAX_LEN]
			target_ids = target_ids[:MAX_LEN]

			while len(input_ids) < MAX_LEN:
				input_ids.append(0)

			while len(target_ids) < MAX_LEN:
				target_ids.append(0)
			
			self.samples.append(
			(
				torch.tensor(input_ids, dtype=torch.long),
				torch.tensor(target_ids, dtype=torch.long)
			)
			)
			
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		return self.samples[idx]
