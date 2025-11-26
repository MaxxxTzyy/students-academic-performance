import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Baca data
df = pd.read_csv("Students_Academic_Performance_Dataset/StudentsPerformance.csv")
df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["high_score"] = (df["avg_score"] >= 70).astype(int)

cat_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

X = df[cat_cols]
y = df["high_score"]

# 2. One-hot encoding
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Definisikan beberapa model
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
}

# 5. Latih & evaluasi semua model
for name, clf in models.items():
    print(f"\n=== {name} ===")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.3f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred))
