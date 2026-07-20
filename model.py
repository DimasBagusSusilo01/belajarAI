import torch
import torch.nn as nn


class ChatModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True)

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output)
        return output
