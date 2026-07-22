import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# --- 1. ARSITEKTUR MODEL ---
class DynamicChatbot(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(DynamicChatbot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

# --- 2. TOKENIZER ---
class CharTokenizer:
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build_vocab(self, texts, max_vocab_size=None):
        all_chars = "".join(texts)
        char_counts = {ch: all_chars.count(ch) for ch in set(all_chars)}
        sorted_chars = sorted(char_counts.keys(), key=lambda x: char_counts[x], reverse=True)
        
        if max_vocab_size and max_vocab_size < len(sorted_chars):
            sorted_chars = sorted_chars[:max_vocab_size]

        self.char2idx = {ch: i for i, ch in enumerate(sorted_chars)}
        self.idx2char = {i: ch for i, ch in enumerate(sorted_chars)}
        self.vocab_size = len(sorted_chars)

    def encode(self, text):
        return [self.char2idx.get(ch, 0) for ch in text if ch in self.char2idx]

    def decode(self, indices):
        return "".join([self.idx2char.get(idx, "") for idx in indices])

# --- 3. FUNGSI TRAINING ---
def train_model(model, tokenizer, dataset, epochs, batch_size):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    print(f"\n[Training] Memulai pelatihan selama {epochs} epoch...")
    
    for epoch in range(epochs):
        total_loss = 0
        for q, a in dataset:
            full_text = f"Q: {q} A: {a}"
            encoded = tokenizer.encode(full_text)

            if len(encoded) < 2:
                continue

            inputs = torch.tensor(encoded[:-1]).unsqueeze(0)
            targets = torch.tensor(encoded[1:]).unsqueeze(0)

            optimizer.zero_grad()
            outputs, _ = model(inputs)

            loss = criterion(outputs.view(-1, tokenizer.vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, len(dataset))
        current_lr = optimizer.param_groups[0]['lr']
        print_interval = max(1, epochs // 5)
        
        if (epoch + 1) % print_interval == 0 or (epoch + 1) == epochs:
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Learning Rate: {current_lr}")
            
    print("[Training] Selesai!\n")

# --- 4. FUNGSI GENERATE ---
def generate_response(model, tokenizer, prompt, max_length=60, temperature=0.7, top_k=3):
    model.eval()
    start_str = f"Q: {prompt} A:"
    encoded = tokenizer.encode(start_str)
    if not encoded:
        maaf_idx = tokenizer.encode("Maaf")
        input_tensor = torch.tensor(maaf_idx if maaf_idx else [0]).unsqueeze(0)
    else:
        input_tensor = torch.tensor(encoded).unsqueeze(0)

    hidden = None
    generated_chars = []

    with torch.no_grad():
        if input_tensor.size(1) > 1:
            _, hidden = model(input_tensor[:, :-1], hidden)
        current_input = input_tensor[:, -1:]

        for _ in range(max_length):
            output, hidden = model(current_input, hidden)
            logits = output[:, -1, :] / max(temperature, 1e-5)
            
            if top_k > 0:
                values, indices = torch.topk(logits, k=min(top_k, logits.size(-1)))
                probs = torch.softmax(values, dim=-1)
                next_idx_in_top = torch.multinomial(probs, num_samples=1).item()
                predicted_idx = indices[0, next_idx_in_top].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                predicted_idx = torch.multinomial(probs, num_samples=1).item()

            next_char = tokenizer.idx2char.get(predicted_idx, "")
            if next_char == "" or next_char == "\n":
                break

            generated_chars.append(next_char)
            current_input = torch.tensor([[predicted_idx]])

    return "".join(generated_chars)

# --- 5. MAIN PROGRAM DENGAN FITUR .PTH ---
if __name__ == "__main__":
    print("=== SETUP CHATBOT PYTHON (PyTorch .pth Support) ===")
    
    csv_file = input("Masukkan nama file CSV dataset (contoh: dataset.csv): ").strip()
    dataset = []
    
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file, sep=None, engine='python')
            dataset = list(zip(df.iloc[:, 0].astype(str), df.iloc[:, 2].astype(str)))
            print(f"Berhasil memuat {len(dataset)} baris data dari {csv_file}")
        except Exception as e:
            print(f"Gagal membaca CSV: {e}.")

    if not dataset:
        dataset = [("Halo", "Halo! Ada yang bisa kubantu?"), ("Siapa namamu?", "Aku chatbot buatanmu.")]

    # Konfigurasi Hyperparameter
    try:
        max_v = int(input("Batasi Max Vocab (0 untuk semua): ") or 0)
        max_v = None if max_v <= 0 else max_v
        EMBEDDING_DIM = int(input("Embedding Dimension (default 32): ") or 32)
        HIDDEN_DIM = int(input("Hidden Dimension (default 64): ") or 64)
        BATCH_SIZE = int(input("Batch Size (default 1): ") or 1)
        EPOCHS = int(input("Jumlah Epoch Training Awal (default 100): ") or 100)
        TEMPERATURE = float(input("Temperature (default 0.7): ") or 0.7)
        TOP_K = int(input("Top-K Sampling (default 3): ") or 3)
    except ValueError:
        max_v, EMBEDDING_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS, TEMPERATURE, TOP_K = None, 32, 64, 1, 100, 0.7, 3

    # Inisialisasi Tokenizer & Model
    print("\nMembangun Kamus Tokenizer...")
    all_texts = [q + a for q, a in dataset]
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(all_texts, max_vocab_size=max_v)
    print(f"Total Vocab Terbentuk: {tokenizer.vocab_size}")

    model = DynamicChatbot(tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM)

    # Cek apakah file model .pth sudah ada sebelumnya
    model_filename = "chatbot_mandiri.pth"
    if os.path.exists(model_filename):
        muat_model = input(f"File '{model_filename}' ditemukan. Ingin langsung memuatnya tanpa training ulang? (y/n): ")
        if muat_model.lower() == 'y':
            try:
                model.load_state_dict(torch.load(model_filename))
                model.eval()
                print("Model berhasil dimuat dari file .pth!")
            except Exception as e:
                print(f"Gagal memuat file .pth ({e}), melakukan training ulang dari awal...")
                train_model(model, tokenizer, dataset, EPOCHS, BATCH_SIZE)
        else:
            train_model(model, tokenizer, dataset, EPOCHS, BATCH_SIZE)
            torch.save(model.state_dict(), model_filename)
            print(f"Model baru berhasil disimpan ke {model_filename}")
    else:
        train_model(model, tokenizer, dataset, EPOCHS, BATCH_SIZE)
        torch.save(model.state_dict(), model_filename)
        print(f"Model berhasil disimpan ke {model_filename}")

    print("=" * 50)
    print(" CHATBOT SIAP! Ketik 'exit' untuk keluar.")
    print("=" * 50)

    while True:
        user_input = input("\nAnda: ")
        if user_input.lower() == "exit":
            break

        response = generate_response(model, tokenizer, user_input, max_length=60, temperature=TEMPERATURE, top_k=TOP_K)
        print(f"Bot: {response}")

        feedback = input("Apakah jawaban ini sudah sesuai? (y/n): ")
        if feedback.lower() == "n":
            correct_answer = input("Masukkan jawaban yang benar (Dataset baru otomatis ditambah): ")
            
            dataset.append((user_input, correct_answer))
            all_texts = [q + a for q, a in dataset]
            tokenizer.build_vocab(all_texts, max_vocab_size=max_v)

            print("Menyesuaikan ulang arsitektur model dengan data baru...")
            model = DynamicChatbot(tokenizer.vocab_size, EMBEDDING_DIM, HIDDEN_DIM)
            train_model(model, tokenizer, dataset, epochs=max(30, EPOCHS // 2), batch_size=BATCH_SIZE)
            
            # Simpan pembaruan ke file .pth otomatis
            torch.save(model.state_dict(), model_filename)
            print(f"Model yang telah diperbarui berhasil disimpan ulang ke {model_filename}!")
