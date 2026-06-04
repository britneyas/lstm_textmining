# 🧠 SentiMind ID - Analisis Sentimen Bahasa Indonesia

Model Bidirectional LSTM untuk klasifikasi sentimen teks Bahasa Indonesia.

## Struktur Folder

```
.
├── data.csv              # Dataset ulasan Bahasa Indonesia
├── train.py              # Script training model LSTM
├── preprocessing.py      # Fungsi preprocessing teks
├── streamlit_app.py      # UI Streamlit
├── requirements.txt
└── models/               # Dibuat otomatis setelah train
    ├── lstm_model.keras
    ├── tokenizer.pkl
    ├── config.pkl
    └── training_history.pkl
```

## Cara Penggunaan

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2. Latih model (wajib dilakukan sekali)

```bash
python train.py
```

Proses ini akan membuat folder `models/` dengan semua artifact yang dibutuhkan.

### 3. Jalankan aplikasi

```bash
streamlit run streamlit_app.py
```

## Deploy ke Streamlit Cloud

1. Push semua file ke GitHub (termasuk folder `models/` beserta isinya)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repo, pilih `streamlit_app.py` sebagai main file
4. Deploy

> Catatan: folder `models/` harus ikut di-push ke GitHub agar Streamlit Cloud bisa memuat model tanpa perlu training ulang.

## Fitur Aplikasi

- **Analisis Teks** - Input teks tunggal, tampilkan label + probabilitas
- **Analisis Batch** - Upload CSV atau ketik banyak teks sekaligus, download hasil
- **Statistik Model** - Grafik training history (akurasi & loss per epoch)

## Arsitektur Model

```
Embedding (10000, 64)
Bidirectional LSTM (64 units, return_sequences=True)
Dropout (0.3)
GlobalMaxPooling1D
Dense (64, relu)
Dropout (0.3)
Dense (1, sigmoid)
```
