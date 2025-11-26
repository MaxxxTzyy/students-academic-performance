import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Baca dataset (perhatikan path foldernya)
df = pd.read_csv("Students_Academic_Performance_Dataset/StudentsPerformance.csv")

# 2. Lihat bentuk dasar data
print("=== 5 data teratas ===")
print(df.head())

print("\n=== Info dataset ===")
print(df.info())

print("\n=== Statistik deskriptif nilai ===")
print(df[["math score", "reading score", "writing score"]].describe())

# 3. Tambah kolom rata-rata nilai
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

print("\n=== 5 data dengan rata-rata nilai ===")
print(df[["gender", "math score", "reading score", "writing score", "avg_score"]].head())

# 4. Rata-rata nilai per gender
print("\n=== Rata-rata nilai per gender ===")
print(df.groupby("gender")["avg_score"].mean())

# 5. Plot distribusi rata-rata nilai
plt.figure()
sns.histplot(df["avg_score"], kde=True)
plt.title("Distribusi Rata-rata Nilai Siswa")
plt.xlabel("Rata-rata nilai")
plt.ylabel("Jumlah siswa")
plt.tight_layout()
plt.show()

# 6. Boxplot rata-rata nilai per test preparation course
plt.figure()
sns.boxplot(data=df, x="test preparation course", y="avg_score")
plt.title("Nilai Rata-rata vs Test Preparation Course")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
