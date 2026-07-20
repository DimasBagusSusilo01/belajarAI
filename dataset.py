import pandas as pd
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tokenizer import Tokenizer

class ChatDataset(Dataset):

    def __init__(self, csv_file, vocab_file):

        self.df = pd.read_csv(csv_file)

        self.tokenizer = Tokenizer()
        self.tokenizer.load(vocab_file)
        self.inputs = []
        self.targets = []
        
        for _, row in self.df.iterrows():
            input_ids = self.tokenizer.encode(row["text"])
            target_ids = self.tokenizer.encode(row["respon"])
            
            self.inputs.append(input_ids)
            self.targets.append(target_ids)
            self.max_length = 0

        for seq in self.inputs + self.targets:
            if len(seq) > self.max_length:
                self.max_length = len(seq)
                
        for i in range(len(self.inputs)):
            self.inputs[i] = self.pad(self.inputs[i])
            self.targets[i] = self.pad(self.targets[i])
        
    def pad(self, sequence):
        while len(sequence) < self.max_length:
            sequence.append(
                self.tokenizer.word2idx["<PAD>"]
        )
        return sequence
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        input_tensor = torch.tensor(
            self.inputs[index],
            dtype=torch.long
    )
        target_tensor = torch.tensor(
            self.targets[index],
            dtype=torch.long
    )

        return input_tensor, target_tensor
    
dataset = ChatDataset(
    "dataset.csv",
    "vocab.json"
)

print("Jumlah data :", len(dataset))

x, y = dataset[0]

print()

print("Input")
print(x)

print()

print("Target")
print(y)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

for x, y in loader:

    print(x.shape)
    print(y.shape)

    break
