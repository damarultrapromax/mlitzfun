import pandas as pd

# Membuat dataset sederhana
data = {
    "nama": ["Andi", "Budi", "Citra", "Dina", "Eka"],
    "umur": [18, 20, 19, 21, 20],
    "nilai": [85, 90, 88, 95, 78]
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Menampilkan seluruh dataset
print("=== DATASET ===")
print(df)

# Lima baris pertama
print("\n=== HEAD ===")
print(df.head())

# Informasi dataset
print("\n=== INFO ===")
df.info()

# Statistik dataset
print("\n=== DESCRIBE ===")
print(df.describe())

# Mengakses satu kolom
print("\n=== KOLOM NILAI ===")
print(df["nilai"])

# Siswa yang memiliki nilai di atas 85 (kode ini pakai >= 90)
print("\n=== NILAI DI ATAS 90 ===")
print(df[df["nilai"] >= 90])

print("\n=== UMUR DI BAWAH 20 ===")
print(df[df["umur"] < 20])

# Challenge 3: Urutkan nilai tertinggi ke terendah
print("\n=== SORTING NILAI ===")
# Masukkan hasil sorting ke dalam variabel filtered_df
filtered_df = df.sort_values(by="nilai", ascending=False)
print(filtered_df)

from pathlib import Path

# Lokasi file CSV di sebelah script
csv_path = Path(__file__).with_name("nilai_siswa.csv")

# Simpan DataFrame menjadi CSV
df.to_csv(csv_path, index=False)

# Baca kembali file CSV
new_df = pd.read_csv(csv_path)

print("\n=== DATASET FROM CSV ===")
print(new_df)

# Informasi dataset
print("\n=== CSV INFO ===")
new_df.info()