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
	encoded = tokenizer.encode(dialog)
	input_ids = torch.tensor(
	encoded,
	dtype=torch.long).unsqueeze(0)
	
	with torch.no_grad():
		output = model(input_ids)
		next_token = torch.argmax(
			output[0, -1]).item()
		jawaban = tokenizer.decode([next_token])
		print("Bot :", jawaban)
