"""
train.py - Melatih model Analisis Sentimen Bahasa Indonesia
Jalankan sekali sebelum deploy: python train.py
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import preprocess

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Memuat data...")
df = pd.read_csv("data.csv")
print(f"Total baris  : {len(df)}")
print(f"Distribusi   :\n{df['label'].value_counts()}\n")

df["clean"] = df["text"].apply(preprocess)

X = df["clean"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

# Coba 3 model, pilih yang terbaik
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Linear SVC"         : LinearSVC(max_iter=2000, C=1.0, random_state=42),
    "Naive Bayes"        : MultinomialNB(alpha=0.1),
}

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

best_name  = ""
best_acc   = 0
best_model = None
results    = {}

for name, clf in models.items():
    clf.fit(X_train_tfidf, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test_tfidf))
    results[name] = round(acc, 4)
    print(f"{name}: {acc:.4f}")
    if acc > best_acc:
        best_acc   = acc
        best_name  = name
        best_model = clf

print(f"\nModel terbaik: {best_name} ({best_acc:.4f})")
print(classification_report(y_test, best_model.predict(X_test_tfidf),
                            target_names=["Negatif", "Positif"]))

# Simpan
joblib.dump(best_model, os.path.join(MODEL_DIR, "ml_model.pkl"))
joblib.dump(tfidf,      os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

meta = {
    "best_model"   : best_name,
    "best_accuracy": best_acc,
    "all_results"  : results,
    "train_size"   : len(X_train),
    "test_size"    : len(X_test),
    "vocab_size"   : len(tfidf.vocabulary_),
}
with open(os.path.join(MODEL_DIR, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print("\nSelesai! File tersimpan di folder models/")
