"""
streamlit_app.py - Analisis Sentimen Bahasa Indonesia (LSTM)
Deploy: streamlit run streamlit_app.py
"""

import os
import pickle
import re
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import preprocess

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentiMind ID",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-dark:    #0a0e1a;
    --bg-card:    #111827;
    --bg-card2:   #1a2236;
    --accent:     #6366f1;
    --accent2:    #818cf8;
    --green:      #22d3a8;
    --red:        #f87171;
    --yellow:     #fbbf24;
    --text:       #e2e8f0;
    --text-muted: #64748b;
    --border:     rgba(99,102,241,0.2);
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-dark) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Headings */
h1, h2, h3 { font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 800; }

/* Input */
.stTextArea textarea {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
    padding: 14px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 0.6rem 1.2rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.3px;
}
.stButton button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-accent {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(124,58,237,0.1));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
}

/* Sentiment badges */
.badge-pos {
    display:inline-block;
    background: rgba(34,211,168,0.15);
    color: var(--green);
    border: 1px solid rgba(34,211,168,0.4);
    border-radius: 99px;
    padding: 4px 16px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-neg {
    display:inline-block;
    background: rgba(248,113,113,0.15);
    color: var(--red);
    border: 1px solid rgba(248,113,113,0.4);
    border-radius: 99px;
    padding: 4px 16px;
    font-weight: 700;
    font-size: 13px;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Metric boxes */
.metric-box {
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: var(--accent2);
    line-height: 1.1;
}
.metric-label {
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Progress bar custom */
.bar-wrap { background: var(--bg-card2); border-radius: 8px; height: 10px; overflow: hidden; margin-top: 6px; }
.bar-fill-pos { height: 100%; background: linear-gradient(90deg, var(--green), #10b981); border-radius: 8px; transition: width 0.8s ease; }
.bar-fill-neg { height: 100%; background: linear-gradient(90deg, var(--red), #ef4444); border-radius: 8px; transition: width 0.8s ease; }

/* Result emoji */
.result-emoji { font-size: 72px; text-align: center; display: block; animation: bounceIn 0.5s ease; }
@keyframes bounceIn {
    0%   { transform: scale(0.3); opacity: 0; }
    60%  { transform: scale(1.1); }
    80%  { transform: scale(0.95); }
    100% { transform: scale(1); opacity: 1; }
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Selectbox */
.stSelectbox div[data-baseweb="select"] > div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
MODEL_DIR = "models"

@st.cache_resource(show_spinner="Memuat model LSTM...")
def load_model_artifacts():
    from tensorflow.keras.models import load_model

    # Coba .keras dulu, fallback ke .h5
    keras_path = os.path.join(MODEL_DIR, "lstm_model.keras")
    h5_path    = os.path.join(MODEL_DIR, "lstm_model.h5")

    if os.path.exists(keras_path):
        model = load_model(keras_path)
    elif os.path.exists(h5_path):
        model = load_model(h5_path)
    else:
        return None, None, None

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    return model, tokenizer, config


@st.cache_data
def load_history():
    path = os.path.join(MODEL_DIR, "training_history.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Predict helper ────────────────────────────────────────────────────────────
def predict_sentiment(text: str, model, tokenizer, config):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    clean = preprocess(text)
    seq   = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=config["MAX_LEN"],
                           padding="post", truncating="post")
    proba = float(model.predict(padded, verbose=0)[0][0])
    label = "Positif" if proba >= 0.5 else "Negatif"
    conf  = proba if label == "Positif" else 1.0 - proba
    return label, conf, proba, clean


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 20px 0;'>
      <span style='font-size:2.5rem'>🧠</span>
      <h2 style='margin:8px 0 4px 0; color:#e2e8f0; font-size:1.3rem'>SentiMind ID</h2>
      <p style='color:#64748b; font-size:12px; margin:0'>Analisis Sentimen Bahasa Indonesia</p>
    </div>
    <hr/>
    """, unsafe_allow_html=True)

    st.markdown("**Model**")
    st.markdown("""
    <div class='metric-box' style='text-align:left; padding:14px 16px;'>
      <span style='font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:1px'>Arsitektur</span><br/>
      <span style='font-weight:700; color:#818cf8'>Bidirectional LSTM</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("**Tentang**")
    st.markdown("""
    <p style='color:#94a3b8; font-size:13px; line-height:1.7'>
    Model deep learning berbasis <b>Bidirectional LSTM</b> 
    untuk klasifikasi sentimen teks Bahasa Indonesia.<br/><br/>
    Dilatih pada dataset ulasan produk, restoran, dan film berbahasa Indonesia.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.caption("v2.0.0 · LSTM · Streamlit")


# ── Main content ──────────────────────────────────────────────────────────────
model, tokenizer, config = load_model_artifacts()

st.markdown("""
<div style='padding: 32px 0 24px 0;'>
  <h1 style='font-size:2.2rem; margin:0 0 8px 0; background: linear-gradient(135deg,#818cf8,#22d3a8); -webkit-background-clip:text; -webkit-text-fill-color:transparent'>
    Analisis Sentimen Bahasa Indonesia
  </h1>
  <p style='color:#64748b; font-size:15px; margin:0'>
    Deteksi sentimen positif & negatif menggunakan model Deep Learning Bidirectional LSTM
  </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("""
    **Model belum tersedia!**

    Jalankan perintah berikut untuk melatih model terlebih dahulu:

    ```bash
    python train.py
    ```

    Setelah selesai, folder `models/` akan berisi:
    - `lstm_model.keras`
    - `tokenizer.pkl`
    - `config.pkl`
    - `training_history.pkl`
    """)
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["  🔍 Analisis Teks  ", "  📦 Analisis Batch  ", "  📊 Statistik Model  "])

# ─── Tab 1: Analisis teks tunggal ─────────────────────────────────────────────
with tab1:
    col_main, col_result = st.columns([1.2, 1], gap="large")

    with col_main:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("##### ✍️ Masukkan Teks untuk Dianalisis")

        # Contoh teks
        eg_col1, eg_col2, eg_col3 = st.columns(3)
        EXAMPLES = {
            "😊 Positif": "Produknya sangat berkualitas, pengiriman cepat dan pelayanan seller ramah banget! Puas banget beli di sini.",
            "😞 Negatif": "Barang tidak sesuai gambar, kualitas buruk dan pengiriman lama sekali. Sangat kecewa dan menyesal beli.",
            "😐 Netral" : "Produk sudah sampai sesuai pesanan. Lumayan lah untuk harganya."
        }
        selected = ""
        with eg_col1:
            if st.button("😊 Contoh Positif", use_container_width=True):
                selected = EXAMPLES["😊 Positif"]
        with eg_col2:
            if st.button("😞 Contoh Negatif", use_container_width=True):
                selected = EXAMPLES["😞 Negatif"]
        with eg_col3:
            if st.button("😐 Contoh Netral", use_container_width=True):
                selected = EXAMPLES["😐 Netral"]

        default_text = selected if selected else st.session_state.get("input_text", "")
        if selected:
            st.session_state["input_text"] = selected

        user_text = st.text_area(
            "Teks ulasan:",
            value=default_text,
            height=140,
            placeholder="Ketik atau paste teks ulasan Bahasa Indonesia di sini...",
            label_visibility="collapsed",
            key="input_text"
        )

        btn_col1, btn_col2 = st.columns([2, 1])
        with btn_col1:
            analyze = st.button("🔍 Analisis Sentimen", use_container_width=True)
        with btn_col2:
            if st.button("🗑️ Hapus", use_container_width=True):
                st.session_state["input_text"] = ""
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if analyze and user_text.strip():
            with st.spinner("Menganalisis..."):
                label, conf, proba_pos, clean_text = predict_sentiment(
                    user_text, model, tokenizer, config
                )

            st.markdown("<div class='card-accent'>", unsafe_allow_html=True)
            st.markdown("##### 🎯 Hasil Analisis")

            emoji = "😊" if label == "Positif" else "😞"
            st.markdown(f"<span class='result-emoji'>{emoji}</span>", unsafe_allow_html=True)

            badge = "badge-pos" if label == "Positif" else "badge-neg"
            st.markdown(
                f"<div style='text-align:center; margin:12px 0'>"
                f"<span class='{badge}'>{label}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

            # Confidence bar
            st.markdown("**Distribusi Probabilitas**")
            proba_neg = 1.0 - proba_pos
            pos_pct = int(proba_pos * 100)
            neg_pct = int(proba_neg * 100)

            st.markdown(f"""
            <div style='margin:8px 0'>
              <div style='display:flex; justify-content:space-between; font-size:12px; color:#94a3b8; margin-bottom:4px'>
                <span>😊 Positif</span><span><b style='color:#22d3a8'>{pos_pct}%</b></span>
              </div>
              <div class='bar-wrap'>
                <div class='bar-fill-pos' style='width:{pos_pct}%'></div>
              </div>
            </div>
            <div style='margin:8px 0'>
              <div style='display:flex; justify-content:space-between; font-size:12px; color:#94a3b8; margin-bottom:4px'>
                <span>😞 Negatif</span><span><b style='color:#f87171'>{neg_pct}%</b></span>
              </div>
              <div class='bar-wrap'>
                <div class='bar-fill-neg' style='width:{neg_pct}%'></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='metric-box' style='margin-top:12px'>
              <span class='metric-value' style='font-size:1.6rem; color:{"#22d3a8" if label=="Positif" else "#f87171"}'>{conf:.1%}</span>
              <div class='metric-label'>Tingkat Kepercayaan</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔎 Token setelah preprocessing"):
                st.code(clean_text if clean_text else "(teks kosong setelah preprocessing)", language=None)

            st.markdown("</div>", unsafe_allow_html=True)

        elif analyze and not user_text.strip():
            st.warning("Masukkan teks terlebih dahulu!")
        else:
            st.markdown("""
            <div class='card' style='text-align:center; padding:40px 20px;'>
              <span style='font-size:3rem'>🤔</span>
              <p style='color:#64748b; margin:12px 0 0 0'>Masukkan teks dan klik <b>Analisis Sentimen</b></p>
            </div>
            """, unsafe_allow_html=True)


# ─── Tab 2: Analisis Batch ────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("##### 📦 Analisis Banyak Teks Sekaligus")
    st.markdown("<p style='color:#64748b; font-size:13px'>Upload file CSV dengan kolom <code>text</code>, atau paste teks (satu per baris).</p>", unsafe_allow_html=True)

    batch_mode = st.radio("Mode input:", ["Upload CSV", "Ketik manual"], horizontal=True)

    texts_batch = []
    if batch_mode == "Upload CSV":
        uploaded = st.file_uploader("Upload file CSV", type=["csv"])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            if "text" in df_up.columns:
                texts_batch = df_up["text"].astype(str).tolist()
                st.success(f"{len(texts_batch)} baris ditemukan.")
            else:
                st.error("Kolom 'text' tidak ditemukan di CSV.")
    else:
        raw = st.text_area("Teks (satu per baris):", height=150,
                           placeholder="Produk bagus sekali!\nBarang rusak parah.\n...")
        if raw.strip():
            texts_batch = [t.strip() for t in raw.strip().split("\n") if t.strip()]

    if st.button("🚀 Jalankan Analisis Batch", use_container_width=True) and texts_batch:
        results = []
        prog = st.progress(0, "Menganalisis...")
        for i, txt in enumerate(texts_batch):
            lbl, cf, pp, _ = predict_sentiment(txt, model, tokenizer, config)
            results.append({"Teks": txt, "Sentimen": lbl, "Kepercayaan": f"{cf:.1%}", "Prob. Positif": round(pp, 4)})
            prog.progress((i + 1) / len(texts_batch))
        prog.empty()

        df_res = pd.DataFrame(results)
        pos_count = (df_res["Sentimen"] == "Positif").sum()
        neg_count = (df_res["Sentimen"] == "Negatif").sum()

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{len(df_res)}</div><div class='metric-label'>Total Teks</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown(f"<div class='metric-box'><div class='metric-value' style='color:#22d3a8'>{pos_count}</div><div class='metric-label'>Positif</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown(f"<div class='metric-box'><div class='metric-value' style='color:#f87171'>{neg_count}</div><div class='metric-label'>Negatif</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_res, use_container_width=True, height=350)

        csv_out = df_res.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Hasil CSV", csv_out, "hasil_sentimen.csv", "text/csv")

    st.markdown("</div>", unsafe_allow_html=True)


# ─── Tab 3: Statistik Model ────────────────────────────────────────────────────
with tab3:
    history = load_history()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("##### 📊 Informasi & Statistik Model")

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.markdown("""<div class='metric-box'>
        <div class='metric-value'>BiLSTM</div>
        <div class='metric-label'>Arsitektur</div></div>""", unsafe_allow_html=True)
    with info_col2:
        st.markdown(f"""<div class='metric-box'>
        <div class='metric-value'>{config.get('MAX_WORDS', '-'):,}</div>
        <div class='metric-label'>Vocab Size</div></div>""", unsafe_allow_html=True)
    with info_col3:
        st.markdown(f"""<div class='metric-box'>
        <div class='metric-value'>{config.get('MAX_LEN', '-')}</div>
        <div class='metric-label'>Max Sequence</div></div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if history:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        epochs = list(range(1, len(history["accuracy"]) + 1))

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Akurasi Training", "Loss Training"))
        fig.add_trace(go.Scatter(x=epochs, y=history["accuracy"],
                                 name="Train Acc", line=dict(color="#818cf8", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"],
                                 name="Val Acc", line=dict(color="#22d3a8", width=2.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["loss"],
                                 name="Train Loss", line=dict(color="#f87171", width=2.5)), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],
                                 name="Val Loss", line=dict(color="#fbbf24", width=2.5, dash="dot")), row=1, col=2)

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
            legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"),
            height=360,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        for axis in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
            fig.update_layout(**{axis: dict(gridcolor="rgba(99,102,241,0.1)",
                                            linecolor="rgba(99,102,241,0.2)")})

        st.plotly_chart(fig, use_container_width=True)

        best_val_acc = max(history["val_accuracy"])
        best_epoch   = history["val_accuracy"].index(best_val_acc) + 1
        st.markdown(f"""
        <div class='card' style='margin-top:8px; background:rgba(34,211,168,0.07); border-color:rgba(34,211,168,0.3);'>
          <p style='margin:0; color:#94a3b8; font-size:14px'>
            Model terbaik dicapai pada <b style='color:#22d3a8'>epoch {best_epoch}</b> 
            dengan val accuracy <b style='color:#22d3a8'>{best_val_acc:.4f}</b>
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Training history tidak ditemukan. Jalankan `python train.py` terlebih dahulu.")
