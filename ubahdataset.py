import pandas as pd

df = pd.read_csv("dataset.csv")
df = df.drop(columns=["label"])

df.to_csv("dataset_baru.csv", index=False)
print ("Berhasil dikonversi")
