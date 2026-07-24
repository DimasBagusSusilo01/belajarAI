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
			values, indices = torch.topk(logits, TOP_K)
			prob = torch.softmax(values,dim=-1)
			next_token = indices[torch.multinomial(prob, 1)].item()
			if next_token == tokenizer.word2idx["<EOS>"]:
				break
			generated.append(next_token)
			jawaban = tokenizer.decode(generated[len(encoded):])
			print("Bot :", jawaban)
