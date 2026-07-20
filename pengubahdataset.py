import pandas as pd
import re

# 1. Buka file dataset asli kamu (misal: dataset_4.csv)
df = pd.read_csv("dataset.csv")

# 2. Fungsi pembersih sederhana (lowercase & rapikan tanda baca)
def bersihkan_teks(teks):
    teks = str(teks).lower()  # Ubah jadi huruf kecil semua
    teks = re.sub(r'\s+', ' ', teks)  # Rapikan spasi berlebih
    return teks.strip()

# 3. Terapkan pembersihan
df['text'] = df['text'].apply(bersihkan_teks)
df['label'] = df['label'].apply(bersihkan_teks)
df['respon'] = df['respon'].apply(bersihkan_teks)

# 4. Gabungkan kolom label + text menjadi 1 kolom input
df['input_text'] = df['label'] + " " + df['text']
df['target_text'] = df['respon']

# 5. Ambil HANYA 2 kolom yang dibutuhkan untuk training LSTM
df_siap = df[['input_text', 'target_text']]

# 6. SIMPAN ke file CSV baru (misal: dataset_siap_train.csv)
df_siap.to_csv("dataset_siap_train.csv", index=False)

print("✅ Berhasil! File 'dataset_siap_train.csv' siap digunakan untuk training.")
