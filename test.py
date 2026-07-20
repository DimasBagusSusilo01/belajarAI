import joblib

# Load kembali model dan vectorizer yang sudah disimpan
<<<<<<< HEAD
model = joblib.load('/home/dimas/0MyNLP/model_emosi_ver2.pkl')
vectorizer = joblib.load('/home/dimas/0MyNLP/vectorizer._ver2.pkl')
=======
model = joblib.load('model_emosi.pkl')
vectorizer = joblib.load('vectorizer.pkl')
>>>>>>> 572caa218b085c42544c04a5c2eab596cfe938cb

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
<<<<<<< HEAD
tebak_emosi("MBOH AH!  aku ora gumun. Sak karepmu")
=======
tebak_emosi("MBOH AH!  aku ora gumun")
>>>>>>> 572caa218b085c42544c04a5c2eab596cfe938cb
