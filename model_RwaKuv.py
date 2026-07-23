import torch
import torch.nn as nn
from config_rwakuv import *

class TimeMix(nn.Module):
	def __init__(self,dim):
		super().__init__()
		self.key = nn.Linear(dim,dim)
		self.value = nn.Linear(dim,dim)
		self.receptance = nn.Linear(dim,dim)
		
	def forward(self,x):
		k = self.key(x)
		v = self.value(x)
		r = torch.sigmoid(self.receptance(x))
		return r* v + (1-r) * k

class ChannelMix(nn.Module):
	def __init__(self, dim, hidden_dim):
		super().__init__()
		self.fc1 = nn.Linear(dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim,dim)
		self.relu = nn.ReLU()
		
	def forward(self, x):
			x = self.fc1(x)
			x = self.relu(x)
			x = self.fc2(x)
			return x
			
class Rwkvblock(nn.Module):
	def __init__(self, dim, hidden_dim):
		super().__init__()
		self.timemix = TimeMix(dim)
		self.channelmix = ChannelMix(dim, hidden_dim)
		
	def forward(self, x):
		x = self.timemix(x)
		x = self.channelmix(x)
		return x

class Rwkv(nn.Module):
	def __init__(self):
		super().__init__()
		self.embedding = nn.Embedding(VOCAB_SIZE,EMBED_DIM)
		self.blocks = nn.ModuleList(
		[Rwkvblock(EMBED_DIM, HIDDEN_DIM)
		for _ in range (NUM_LAYERS)]
		)
		self.linear = nn.Linear(EMBED_DIM, VOCAB_SIZE)
		
	def forward(self, x):
		x = self.embedding(x)
		for balok in self.blocks:
			x = balok(x)
		x = self.linear(x)
		return x
