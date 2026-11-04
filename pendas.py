import pandas as pd

data = {
    "luas_rumah": [36, 45, 60, 72, 90],
    "jumlah_kamar": [2, 2, 3, 3, 4],
    "harga_juta": [350, 420, 610, 730, 950]
}

df = pd.DataFrame(data)

print(df)


import pandas as pd

data = {
    "luas_rumah": [36, 45, 60, 72, 90],
    "jumlah_kamar": [2, 2, 3, 3, 4],
    "harga_juta": [350, 420, 610, 730, 950]
}

df = pd.DataFrame(data)

print("\n=== DATASET ===")
print(df)

print("\n=== 5 BARIS PERTAMA ===")
print(df.head())

print("\n=== INFORMASI DATASET ===")
df.info()

print("\n=== STATISTIK DATASET ===")
print(df.describe())

print("\n=== UKURAN DATASET ===")
print(df.shape)

print("\n=== DATA KOSONG ===")
print(df.isnull().sum())
