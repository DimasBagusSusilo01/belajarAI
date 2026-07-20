import torch
from model import ChatModel
from tokenizer import Tokenizer
import torch.nn.functional as F

# Konfigurasi harus SAMA dengan saat training
EMBEDDING_DIM = 64
HIDDEN_DIM = 64
MAX_LENGTH = 20  # Batas maksimum panjang respon
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_resources():
    # 1. Load Tokenizer
    tokenizer = Tokenizer()
    tokenizer.load("vocab.json")
    vocab_size = len(tokenizer.word2idx)

    # 2. Inisialisasi Model & Load Bobot (.pth)
    model = ChatModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(torch.load("chatbot_model.pth", map_location=DEVICE))
    model.eval()  # Ubah ke mode evaluasi/prediksi

    return tokenizer, model

def generate_response(prompt, tokenizer, model, temperature=0.7, top_k=5):
    # Prepare Input
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        
        predicted_ids = []
        
        # Iterasi untuk setiap posisi kata dalam kalimat output
        for t in range(output.shape[1]):
            logits = output[0, t, :]  # Ambil distribusi nilai untuk kata ke-t
            
            # 1. Terapkan Temperature
            # Nilai logits dibagi temperature sebelum dimasukkan ke Softmax
            logits = logits / temperature
            
            # 2. Terapkan Top-k Filtering
            if top_k > 0:
                # Cari k nilai terbesar
                top_k_values, _ = torch.topk(logits, top_k)
                # Nilai di luar top-k diubah menjadi -infinity agar probabilitasnya 0 saat di-softmax
                min_value = top_k_values[-1]
                logits[logits < min_value] = float('-inf')
            
            # 3. Hitung Probabilitas dengan Softmax
            probabilities = F.softmax(logits, dim=-1)
            
            # 4. Sampling Menggunakan Multinomial (Bukan Argmax!)
            # Ini akan mengambil 1 index kata secara acak berdasarkan bobot probabilitasnya
            next_token_id = torch.multinomial(probabilities, num_samples=1).item()
            
            predicted_ids.append(next_token_id)

    # Decode kembali angka menjadi teks
    response_text = tokenizer.decode(predicted_ids)
    return response_text

def chat():
    tokenizer, model = load_resources()
    print("= Chatbot Siap! (Ketik 'keluar' untuk berhenti) =")

    while True:
        user_input = input("\nKamu : ")
        if user_input.lower() in ["keluar", "exit", "quit"]:
            print("Chatbot: Sampai jumpa!")
            break

        if not user_input.strip():
            continue

        response = generate_response(
    user_input, 
    tokenizer, 
    model, 
    temperature=0.7, 
    top_k=5)
        print(f"Bot  : {response}")

if __name__ == "__main__":
    chat()
