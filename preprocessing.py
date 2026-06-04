"""
preprocessing.py - Fungsi pembersihan teks Bahasa Indonesia
Digunakan bersama oleh train.py dan streamlit_app.py
"""

import re

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan",
    "untuk", "pada", "adalah", "atau", "juga", "karena", "saya",
    "aku", "kamu", "kami", "kita", "mereka", "dia", "tidak", "bisa",
    "ada", "sudah", "akan", "bisa", "lebih", "sangat", "sekali",
    "jadi", "kalau", "tapi", "agar", "supaya", "bagi", "oleh",
    "lagi", "pun", "sih", "deh", "dong", "nih", "ya", "nya", "si"
}


def preprocess(text: str) -> str:
    """Bersihkan dan normalisasi teks Bahasa Indonesia."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # hapus URL
    text = re.sub(r"@\w+|#\w+", " ", text)               # hapus mention/hashtag
    text = re.sub(r"[^a-z\s]", " ", text)                # hapus non-alfabet
    text = re.sub(r"\s+", " ", text).strip()              # normalisasi spasi
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)
