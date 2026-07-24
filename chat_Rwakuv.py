import torch

from tokenizer import Tokenizer
from model_RwaKuv import Rwkv
from config_rwakuv import *


tokenizer = Tokenizer()
tokenizer.load("vocab.json")

model = Rwkv()
model.load_state_dict(
	torch.load("rwkv_chatbot.pth")
)
model.eval()

while True:
	text = input("Kamu : ")
	if text.lower() == "exit":
		break
	 
	dialog = f"<USER> {text} <BOT>"
	encoded = tokenizer.encode(dialog, add_eos=False)
	generated = encoded.copy()
	for i in range(MAX_NEW_TOKENS):
		input_ids = torch.tensor(
			generated,
			dtype=torch.long
		).unsqueeze(0)
		with torch.no_grad():
			output = model(input_ids)
			logits = output[0, -1]
			# Jangan keluarkan token khusus
			special_tokens = [
				"<PAD>",
				"<SOS>",
				"<USER>",
				"<BOT>"]
			for token in special_tokens:
				logits[tokenizer.word2idx[token]] = -1e9
			for token in set(generated):
				logits[token] /= REPETITION_PENALTY
			
			logits = logits / TEMPERATURE
			topk_logits, top_indices = torch.topk(logits, TOP_K)
			probs = torch.softmax(topk_logits, dim=-1)
			sorted_probs, sorted_idx = torch.sort(probs,descending=True)
			cumulative_probs = torch.cumsum(sorted_probs,dim=-1)
			mask = cumulative_probs > TOP_P
			mask[1:] = mask[:-1].clone()
			mask[0] = False
			
			sorted_probs[mask] = 0
			sorted_probs /= sorted_probs.sum()
			sample = torch.multinomial(sorted_probs,1).item()
			next_token = top_indices[sorted_idx[sample]].item()
			#values, indices = torch.topk(logits, TOP_K)
			#prob = torch.softmax(values,dim=-1)
			#next_token = indices[torch.multinomial(prob, 1)].item()
			print ("next_token:",tokenizer.idx2word[next_token])
			if next_token == tokenizer.word2idx["<EOS>"]:
				if len(generated) == len(encoded):
					continue
				break
			generated.append(next_token)
	jawaban = tokenizer.decode(generated[len(encoded):])
	print("Bot :", jawaban)
