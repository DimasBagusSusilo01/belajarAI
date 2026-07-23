import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import ChatDataset
from model import ChatModel

# 1. Hyperparameters & Konfigurasi
BATCH_SIZE = 16
EMBEDDING_DIM = 64  # Dinaikkan dari 64 agar daya tampung makna kata lebih kaya
HIDDEN_DIM = 128     # Dinaikkan dari 128 agar memori sekuensial LSTM lebih kuat
EPOCHS = 80
LEARNING_RATE = 0.001
MAX_NORM = 1.0        # Batas gradien clipping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print(f"Menggunakan device: {DEVICE}")

    # 2. Load Dataset & DataLoader
    # ChatDataset sudah menginisialisasi tokenizer internal di dalamnya
    dataset = ChatDataset("dataset.csv", "vocab.json")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Ambil ukuran vocabulary langsung dari dataset
    vocab_size = len(dataset.tokenizer.word2idx)
    pad_idx = dataset.tokenizer.word2idx["<PAD>"]

    # 3. Inisialisasi Model, Loss Function, dan Optimizer
    model = ChatModel(
        vocab_size=vocab_size, 
        embedding_dim=EMBEDDING_DIM, 
        hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    
    # Abaikan padding (<PAD>) saat menghitung loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Scheduler: Menurunkan LR sebesar 10% setiap 30 epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # 4. Proses Training Loop
    print("\n--- Memulai Training ---")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            # Reset gradien
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)  # Shape: [batch_size, seq_len, vocab_size]
            
            # Reshape output & target
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # --- PENINGKATAN: Gradient Clipping ---
            # Mencegah nilai gradien meledak di pertengahan jalan
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_NORM)

            optimizer.step()
            total_loss += loss.item()

        # Update learning rate tiap akhir epoch
        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - LR: {current_lr:.6f}")

    # 5. Menyimpan Model ke File
    model_path = "chatbot_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nTraining selesai! Model berhasil disimpan ke '{model_path}'")

if __name__ == "__main__":
    main()
