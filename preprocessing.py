import re

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "dengan",
    "untuk", "pada", "adalah", "atau", "juga", "karena", "saya",
    "aku", "kamu", "kami", "kita", "mereka", "dia", "bisa",
    "ada", "sudah", "akan", "lebih", "jadi", "kalau", "tapi",
    "agar", "supaya", "bagi", "oleh", "lagi", "pun", "sih",
    "deh", "dong", "nih", "ya", "nya", "si"
}

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)
