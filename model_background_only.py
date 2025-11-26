import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Baca dataset
df = pd.read_csv("Students_Academic_Performance_Dataset/StudentsPerformance.csv")

# 2. Hitung rata-rata nilai & buat label high_score
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["high_score"] = (df["avg_score"] >= 70).astype(int)

# 3. Fitur: hanya faktor latar belakang (tanpa nilai ujian)
cat_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

X = df[cat_cols]
y = df["high_score"]

# 4. One-hot encoding fitur kategorikal
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("=== Contoh fitur setelah encoding ===")
print(X.head())

# 5. Bagi train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Evaluasi
y_pred = model.predict(X_test)

print("\n=== Classification Report (Background Only) ===")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["low_score", "high_score"]
)
disp.plot()
plt.title("Confusion Matrix - Background Only")
plt.tight_layout()
plt.show()
