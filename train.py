import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Memuat Dataset
df = pd.read_csv('/home/dimas/0MyNLP/envnlp/dataset.csv')

# Split data menjadi fitur (X) dan target/label (y)
X = df['text']
y = df['label']

# 2. Membagi Data Belajar & Data Tes (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Ekstraksi Fitur (Mengubah teks menjadi angka/vektor berdasarkan bobot kata)
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Melatih Model Machine Learning (Naive Bayes)
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 5. Menguji Model dengan Data Tes
y_pred = model.predict(X_test_vec)

print("=== Hasil Evaluasi Model ===")
print(f"Akurasi Model: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# 6. Menyimpan Model dan Vectorizer agar bisa dipakai di Web/FastAPI nanti
joblib.dump(model, 'model_emosi_ver2.pkl')
joblib.dump(vectorizer, 'vectorizer._ver2.pkl')
print("Model berhasil dilatih dan disimpan!")
