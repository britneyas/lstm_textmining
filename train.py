"""
train.py - Melatih model LSTM untuk Analisis Sentimen Bahasa Indonesia
Jalankan sekali sebelum deploy: python train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import re

# ── Konfigurasi ───────────────────────────────────────────────────────────────
DATA_PATH   = "data.csv"
MODEL_DIR   = "models"
MAX_WORDS   = 10000   # vocab size
MAX_LEN     = 50      # panjang sequence
EMBED_DIM   = 64
LSTM_UNITS  = 64
DROPOUT     = 0.3
BATCH_SIZE  = 32
EPOCHS      = 20

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Stopwords Bahasa Indonesia (ringkas) ──────────────────────────────────────
STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan",
    "untuk", "pada", "adalah", "atau", "juga", "karena", "saya",
    "aku", "kamu", "kami", "kita", "mereka", "dia", "tidak", "bisa",
    "ada", "sudah", "akan", "bisa", "lebih", "sangat", "sekali",
    "jadi", "kalau", "tapi", "agar", "supaya", "bagi", "oleh",
    "lagi", "pun", "sih", "deh", "dong", "nih", "ya", "nya", "si"
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Memuat data...")
df = pd.read_csv(DATA_PATH)
print(f"  Total baris  : {len(df)}")
print(f"  Distribusi   :\n{df['label'].value_counts()}")

df["clean"] = df["text"].apply(preprocess)

X = df["clean"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("\nMembangun tokenizer...")
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_train),
    maxlen=MAX_LEN, padding="post", truncating="post"
)
X_test_seq = pad_sequences(
    tokenizer.texts_to_sequences(X_test),
    maxlen=MAX_LEN, padding="post", truncating="post"
)

# ── Build Model LSTM ──────────────────────────────────────────────────────────
print("\nMembangun model LSTM...")
model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
    Dropout(DROPOUT),
    GlobalMaxPooling1D(),
    Dense(64, activation="relu"),
    Dropout(DROPOUT),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# ── Train ─────────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
]

print("\nMelatih model...")
history = model.fit(
    X_train_seq, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# ── Evaluasi ──────────────────────────────────────────────────────────────────
print("\nEvaluasi pada data test...")
y_pred_proba = model.predict(X_test_seq, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

print(f"Akurasi: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Negatif", "Positif"]))

# ── Simpan artifact ───────────────────────────────────────────────────────────
print("\nMenyimpan model dan tokenizer...")

# Simpan model LSTM
model.save(os.path.join(MODEL_DIR, "lstm_model.keras"))

# Simpan tokenizer + config
config = {
    "MAX_LEN"   : MAX_LEN,
    "MAX_WORDS" : MAX_WORDS,
    "EMBED_DIM" : EMBED_DIM,
}
with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
with open(os.path.join(MODEL_DIR, "config.pkl"), "wb") as f:
    pickle.dump(config, f)

# Simpan history untuk visualisasi di Streamlit
history_data = {
    "accuracy"    : history.history.get("accuracy", []),
    "val_accuracy": history.history.get("val_accuracy", []),
    "loss"        : history.history.get("loss", []),
    "val_loss"    : history.history.get("val_loss", []),
}
with open(os.path.join(MODEL_DIR, "training_history.pkl"), "wb") as f:
    pickle.dump(history_data, f)

print("Selesai! File tersimpan di folder 'models/':")
print("  - models/lstm_model.keras")
print("  - models/tokenizer.pkl")
print("  - models/config.pkl")
print("  - models/training_history.pkl")
