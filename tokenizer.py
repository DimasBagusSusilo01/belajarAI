import pandas as pd
import json

class Tokenizer:
    def __init__(self):
        # Token khusus
        self.word2idx = {
            "<PAD>": 0,
            "<SOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "<USER>":4,
            "<BOT>":5,
            "<EMOTION>":6
        }

        self.idx2word = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

    # Membersihkan teks
    def clean(self, text):
        return str(text).strip()

    # Membuat vocabulary
    def build_vocab(self, dataframe):

        sentences = []

        sentences.extend(dataframe["text"].tolist())
        sentences.extend(dataframe["respon"].tolist())

        words = []

        for sentence in sentences:

            sentence = self.clean(sentence)

            tokens = sentence.split()

            words.extend(tokens)

        vocab = sorted(set(words))

        for word in vocab:
            if word not in self.word2idx:

                index = len(self.word2idx)

                self.word2idx[word] = index
                self.idx2word[index] = word

        print("Vocabulary berhasil dibuat")
        print("Jumlah kata :", len(self.word2idx))

    # Mengubah kalimat menjadi angka
    def encode(self, sentence):

        sentence = self.clean(sentence)

        tokens = sentence.split()

        encoded = [self.word2idx["<SOS>"]]

        for token in tokens:
            encoded.append(
                self.word2idx.get(token, self.word2idx["<UNK>"])
            )

        encoded.append(self.word2idx["<EOS>"])

        return encoded

    # Mengubah angka menjadi kalimat
    def decode(self, encoded):

        words = []

        for token in encoded:

            word = self.idx2word.get(token, "<UNK>")

            if word in ["<PAD>", "<SOS>", "<EOS>"]:
                continue

            words.append(word)

        return " ".join(words)

    # Simpan vocabulary
    def save(self, filename):

        with open(filename, "w", encoding="utf-8") as f:

            json.dump(
                self.word2idx,
                f,
                ensure_ascii=False,
                indent=4
            )

        print("Vocabulary disimpan")

    # Load vocabulary
    def load(self, filename):

        with open(filename, "r", encoding="utf-8") as f:

            self.word2idx = json.load(f)

        self.idx2word = {
            int(v): k
            for k, v in self.word2idx.items()
        }

        print("Vocabulary berhasil dimuat")


# ===========================
# Program utama
# ===========================

df = pd.read_csv("dataset.csv")

tokenizer = Tokenizer()

tokenizer.build_vocab(df)

tokenizer.save("vocab.json")

print()

contoh = "MBOH AH aku ra mudeng?"

hasil = tokenizer.encode(contoh)

print("Kalimat :", contoh)
print("Encode  :", hasil)
print("Decode  :", tokenizer.decode(hasil))
