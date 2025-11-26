import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Pengaturan halaman
st.set_page_config("Dashboard Performa Akademik Siswa", layout="wide")

# =========================================================
# 1. LOAD DATA
# =========================================================
def load_data():
    # sesuaikan path kalau folder-nya beda
    df = pd.read_csv(
        "Students_Academic_Performance_Dataset/StudentsPerformance.csv"
    )
    # fitur rata-rata nilai
    df["avg_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
    # label: 1 = high score (>= 70)
    df["high_score"] = (df["avg_score"] >= 70).astype(int)
    return df


df = load_data()

# =========================================================
# 2. TRAIN MODEL SEDERHANA (BERBASIS LATAR BELAKANG)
# =========================================================
cat_cols = [
    "gender",
    "race/ethnicity",
    "parental level of education",
    "lunch",
    "test preparation course",
]

X = df[cat_cols].copy()
y = df["high_score"].copy()

# one-hot encoding
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model logistic regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# simpan nama kolom fitur untuk dipakai saat prediksi
feature_cols = X.columns

# =========================================================
# 3. DASHBOARD UTAMA
# =========================================================
st.title("Dashboard Performa Akademik Siswa")
st.sidebar.header("Filter")

# ---- Filter sidebar ----
genders = ["All"] + sorted(df["gender"].unique().tolist())
selected_gender = st.sidebar.selectbox("Gender", genders)

prep_opts = ["All"] + sorted(df["test preparation course"].unique().tolist())
selected_prep = st.sidebar.selectbox("Test preparation course", prep_opts)

filtered = df.copy()
if selected_gender != "All":
    filtered = filtered[filtered["gender"] == selected_gender]
if selected_prep != "All":
    filtered = filtered[filtered["test preparation course"] == selected_prep]

# ---- Ringkasan ----
st.subheader("Ringkasan Data Terfilter")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah siswa", len(filtered))
if len(filtered) > 0:
    col2.metric("Rata-rata nilai", f"{filtered['avg_score'].mean():.2f}")
    col3.metric(
        "Proporsi high score",
        f"{filtered['high_score'].mean() * 100:.1f}%",
    )
else:
    col2.metric("Rata-rata nilai", "-")
    col3.metric("Proporsi high score", "-")

st.write("Contoh 5 data teratas:")
st.dataframe(filtered.head())

# ---- Histogram avg_score ----
st.subheader("Distribusi Rata-rata Nilai (data terfilter)")
fig1, ax1 = plt.subplots()
if len(filtered) > 0:
    sns.histplot(filtered["avg_score"], kde=True, ax=ax1)
ax1.set_xlabel("Rata-rata nilai")
ax1.set_ylabel("Jumlah siswa")
st.pyplot(fig1)

# ---- Boxplot vs test preparation ----
st.subheader("Nilai Rata-rata vs Test Preparation Course (data terfilter)")
fig2, ax2 = plt.subplots()
if len(filtered) > 0:
    sns.boxplot(
        data=filtered,
        x="test preparation course",
        y="avg_score",
        ax=ax2,
    )
ax2.set_xlabel("Test preparation course")
ax2.set_ylabel("Rata-rata nilai")
plt.xticks(rotation=15)
st.pyplot(fig2)

# ---- Rata-rata per pendidikan orang tua ----
st.subheader("Rata-rata Nilai per Pendidikan Orang Tua (data terfilter)")
if len(filtered) > 0:
    avg_by_parent = (
        filtered.groupby("parental level of education")["avg_score"]
        .mean()
        .sort_values(ascending=False)
    )
    fig3, ax3 = plt.subplots()
    avg_by_parent.plot(kind="bar", ax=ax3)
    ax3.set_ylabel("Rata-rata nilai")
    ax3.set_xlabel("Pendidikan orang tua")
    ax3.set_title("Rata-rata Nilai per Pendidikan Orang Tua")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig3)
else:
    st.write("Tidak ada data setelah filter diterapkan.")

# =========================================================
# 4. SIMULASI PREDIKSI HIGH SCORE
# =========================================================
st.header("Simulasi Prediksi High Score (berdasarkan latar belakang)")

col_a, col_b = st.columns(2)

with col_a:
    gender_input = st.selectbox("Gender", sorted(df["gender"].unique()))
    race_input = st.selectbox(
        "Race/ethnicity", sorted(df["race/ethnicity"].unique())
    )
    parent_edu_input = st.selectbox(
        "Pendidikan orang tua",
        sorted(df["parental level of education"].unique()),
    )

with col_b:
    lunch_input = st.selectbox("Jenis lunch", sorted(df["lunch"].unique()))
    prep_input = st.selectbox(
        "Test preparation course",
        sorted(df["test preparation course"].unique()),
    )

# satu baris data sesuai input user
input_df = pd.DataFrame(
    [
        {
            "gender": gender_input,
            "race/ethnicity": race_input,
            "parental level of education": parent_edu_input,
            "lunch": lunch_input,
            "test preparation course": prep_input,
        }
    ]
)

# encode sama seperti saat training
input_encoded = pd.get_dummies(
    input_df,
    columns=cat_cols,
    drop_first=True,
)

# tambahkan kolom yang hilang dengan 0
for col in feature_cols:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# pastikan urutan kolom sama
input_encoded = input_encoded[feature_cols]

if st.button("Prediksi"):
    prob = model.predict_proba(input_encoded)[0][1]  # peluang high_score
    label = "High Score" if prob >= 0.5 else "Low Score"

    st.write(f"**Prediksi kategori:** {label}")
    st.write(f"Perkiraan peluang high score: **{prob*100:.1f}%**")
