import joblib

# Load kembali model dan vectorizer yang sudah disimpan
model = joblib.load('model_emosi.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def tebak_emosi(kalimat_baru):
    # Ubah kalimat baru menjadi bentuk angka menggunakan vectorizer yang sama
    kalimat_vec = vectorizer.transform([kalimat_baru])
    
    # Lakukan prediksi
    prediksi = model.predict(kalimat_vec)[0]
    
    # Mengambil probabilitas angka/skor untuk setiap label emosi
    probabilitas = model.predict_proba(kalimat_vec)[0]
    skor_emosi = dict(zip(model.classes_, probabilitas))
    
    print(f"\nKalimat: \"{kalimat_baru}\"")
    print(f"Prediksi Utama Emosi: {prediksi.upper()}")
    print("Detail Skor Emosi:")
    for emosi, skor in skor_emosi.items():
        print(f"- {emosi}: {skor * 100:.2f}%")


tebak_emosi("Gabut banget asli, ra reti arep nopo")
tebak_emosi("MBOH AH!  aku ora gumun")
