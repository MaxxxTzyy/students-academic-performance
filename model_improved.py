import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Baca dataset
df = pd.read_csv("Students_Academic_Performance_Dataset/StudentsPerformance.csv")

# 2. Fitur numerik: nilai
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

# 3. Label: 1 = nilai tinggi (>= 70), 0 = rendah
df["high_score"] = (df["avg_score"] >= 70).astype(int)

# 4. Pilih fitur numerik + kategorikal
num_cols = ["math score", "reading score", "writing score"]
cat_cols = ["gender", "lunch", "test preparation course", "parental level of education"]

X = df[num_cols + cat_cols]
y = df["high_score"]

# 5. One-hot encoding untuk fitur kategorikal
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("=== Contoh fitur setelah get_dummies ===")
print(X.head())

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Model Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 8. Prediksi dan evaluasi
y_pred = model.predict(X_test)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# 9. Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low_score", "high_score"])
disp.plot()
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
