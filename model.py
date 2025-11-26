import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Baca dataset
df = pd.read_csv("Students_Academic_Performance_Dataset/StudentsPerformance.csv")

# 2. Buat kolom rata-rata nilai
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

# 3. Buat label: 1 = nilai rata-rata >= 70, 0 = < 70
df["high_score"] = (df["avg_score"] >= 70).astype(int)

print(df[["math score", "reading score", "writing score", "avg_score", "high_score"]].head())

# 4. Pilih fitur (X) dan target (y)
#    Di sini kita pakai nilai math, reading, writing sebagai fitur sederhana
X = df[["math score", "reading score", "writing score"]]
y = df["high_score"]

# 5. Bagi data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Buat dan latih model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Prediksi di data test
y_pred = model.predict(X_test)

# 8. Lihat hasil evaluasi
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
