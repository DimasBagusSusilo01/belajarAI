import joblib

try:
    model = joblib.load('model_emosi.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✓ Model dan Vectorizer berhasil dimuat!\n")
except FileNotFoundError:
    print("✗ Error: File .pkl tidak ditemukan. Jalankan skrip training dulu!")
    exit()

def analisis_kalimat(kalimat_baru):
    # Teks mentah wajib diubah dulu ke bentuk angka lewat vectorizer yang sudah terlatih
    kalimat_vektor = vectorizer.transform([kalimat_baru])
    
    # Tebak label emosi tertingginya
    prediksi_utama = model.predict(kalimat_vektor)[0]
    
    # Ambil persentase keyakinan model untuk masing-masing label
    probabilitas = model.predict_proba(kalimat_vektor)[0]
    
    # Gabungkan nama label dengan skor kemunculannya
    daftar_skor = dict(zip(model.classes_, probabilitas))
    
    # Cetak Hasil ke Terminal
    print(f"Input Kalimat : \"{kalimat_baru}\"")
    print(f"Hasil Prediksi: {prediksi_utama.upper()}")
    print("Detail Vibe Check:")
    for emosi, skor in daftar_skor.items():
        print(f"  - {emosi}: {skor * 100:.2f}%")
    print("-" * 40)

print("=== APLIKASI VIBE CHECK (INDO-JAWA) ===")
print("Ketik 'keluar' untuk menyudahi eksperimen.\n")

while True:
    input_user = input("Masukkan kalimat yang mau dites: ")
    if input_user.lower() == 'keluar':
        print("Eksperimen selesai! Selamat lanjut ngoding web-nya.")
        break
    
    if input_user.strip() == "":
        continue
        
    analisis_kalimat(input_user)
