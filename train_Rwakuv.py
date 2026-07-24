import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChatDataset
from model_RwaKuv import Rwkv
from config_rwakuv import *


dataset = ChatDataset(
	"dataset.csv",
	"vocab.json"
)

dataloader = DataLoader(
	dataset,
	batch_size=BATCH_SIZE,
	shuffle=True
)
model = Rwkv()

criterion = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = torch.optim.Adam(
	model.parameters(),
	lr=LEARNING_RATE
)

for epoch in range(EPOCH):
	total_loss = 0
	for input_ids, target_ids in dataloader:
		optimizer.zero_grad()
		output = model(input_ids)
		output = output.view(-1, VOCAB_SIZE)
		target_ids = target_ids.view(-1)
		
		loss = criterion(output, target_ids)
		loss.backward()
		optimizer.step()
		total_loss += loss.item()

	print(
			f"Epoch {epoch+1}/{EPOCH} "
			f"Loss: {total_loss/len(dataloader):.4f}"
)

torch.save(
	model.state_dict(),
	"rwkv_chatbot.pth"
)

print("Model berhasil disimpan.")
