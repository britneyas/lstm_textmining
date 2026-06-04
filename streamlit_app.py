"""
streamlit_app.py - Analisis Sentimen Bahasa Indonesia (LSTM)
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from preprocessing import preprocess

st.set_page_config(
    page_title="SentiMind ID",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:     #0a0e1a;
    --card:   #111827;
    --card2:  #1a2236;
    --accent: #6366f1;
    --accent2:#818cf8;
    --green:  #22d3a8;
    --red:    #f87171;
    --text:   #e2e8f0;
    --muted:  #64748b;
    --border: rgba(99,102,241,0.2);
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: var(--card) !important;
    border-right: 1px solid var(--border);
}
#MainMenu, footer, header { visibility: hidden; }
.stTextArea textarea {
    background: var(--card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton button {
    background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    transition: all 0.2s !important;
}
.stButton button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important; }
.card  { background: var(--card);  border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 16px; }
.card2 { background: linear-gradient(135deg,rgba(99,102,241,0.15),rgba(124,58,237,0.1)); border:1px solid rgba(99,102,241,0.35); border-radius:16px; padding:24px; margin-bottom:16px; }
.badge-pos { display:inline-block; background:rgba(34,211,168,0.15); color:var(--green); border:1px solid rgba(34,211,168,0.4); border-radius:99px; padding:4px 16px; font-weight:700; font-size:13px; letter-spacing:1px; text-transform:uppercase; }
.badge-neg { display:inline-block; background:rgba(248,113,113,0.15); color:var(--red);   border:1px solid rgba(248,113,113,0.4); border-radius:99px; padding:4px 16px; font-weight:700; font-size:13px; letter-spacing:1px; text-transform:uppercase; }
.mbox { background:var(--card2); border:1px solid var(--border); border-radius:12px; padding:18px 20px; text-align:center; }
.mval { font-size:2rem; font-weight:800; color:var(--accent2); line-height:1.1; }
.mlbl { font-size:11px; color:var(--muted); font-weight:500; margin-top:4px; text-transform:uppercase; letter-spacing:0.8px; }
.bar-wrap { background:var(--card2); border-radius:8px; height:10px; overflow:hidden; margin-top:6px; }
.bar-pos  { height:100%; background:linear-gradient(90deg,var(--green),#10b981); border-radius:8px; }
.bar-neg  { height:100%; background:linear-gradient(90deg,var(--red),#ef4444);   border-radius:8px; }
.emoji-big { font-size:72px; text-align:center; display:block; animation:pop 0.4s ease; }
@keyframes pop { 0%{transform:scale(0.3);opacity:0} 70%{transform:scale(1.1)} 100%{transform:scale(1);opacity:1} }
.stTabs [data-baseweb="tab-list"] { background:var(--card)!important; border-radius:12px!important; padding:4px!important; }
.stTabs [data-baseweb="tab"]      { background:transparent!important; color:var(--muted)!important; font-weight:600!important; border-radius:8px!important; }
.stTabs [aria-selected="true"]    { background:var(--accent)!important; color:white!important; }
hr { border-color:var(--border)!important; }
</style>
""", unsafe_allow_html=True)

MODEL_DIR = "models"

@st.cache_resource(show_spinner="Memuat model LSTM...")
def load_artifacts():
    from tensorflow.keras.models import load_model

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

def predict(text, model, tokenizer, config):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    clean  = preprocess(text)
    seq    = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=config["MAX_LEN"], padding="post", truncating="post")
    proba_pos = float(model.predict(padded, verbose=0)[0][0])
    label = "Positif" if proba_pos >= 0.5 else "Negatif"
    conf  = proba_pos if label == "Positif" else 1 - proba_pos
    return label, conf, proba_pos, clean

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:8px 0 20px 0'>
      <span style='font-size:2.5rem'>🧠</span>
      <h2 style='margin:8px 0 4px 0; color:#e2e8f0; font-size:1.3rem'>SentiMind ID</h2>
      <p style='color:#64748b; font-size:12px; margin:0'>Analisis Sentimen Bahasa Indonesia</p>
    </div><hr/>
    """, unsafe_allow_html=True)

    model, tokenizer, config = load_artifacts()

    st.markdown("**Model Aktif**")
    st.markdown("""
    <div class='mbox' style='text-align:left; padding:14px 16px; margin-bottom:12px'>
      <span style='font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:1px'>Arsitektur</span><br/>
      <span style='font-weight:700; color:#818cf8'>Bidirectional LSTM</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding:12px; background:rgba(99,102,241,0.1);
    border:1px solid rgba(99,102,241,0.25); border-radius:10px; margin-top:8px'>
      <p style='margin:0; font-size:13px; font-weight:700; color:#818cf8'>Britney Angeline</p>
      <p style='margin:2px 0 0 0; font-size:11px; color:#64748b; font-family:monospace'>NIM: 2702333586</p>
    </div>
    """, unsafe_allow_html=True)
    st.caption("v2.0.0 · Bidirectional LSTM · Streamlit")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:32px 0 24px 0'>
  <h1 style='font-size:2.2rem; margin:0 0 8px 0;
    background:linear-gradient(135deg,#818cf8,#22d3a8);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent'>
    Analisis Sentimen Bahasa Indonesia
  </h1>
  <p style='color:#64748b; font-size:15px; margin:0'>
    Deteksi sentimen positif &amp; negatif menggunakan Deep Learning · Bidirectional LSTM
  </p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("Model belum ada! Pastikan folder `models/` berisi `lstm_model.keras`, `tokenizer.pkl`, `config.pkl`.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["  🔍 Analisis Teks  ", "  📦 Analisis Batch  ", "  📊 Statistik Model  "])

# ── Tab 1 ─────────────────────────────────────────────────────────────────────
with tab1:
    col_in, col_out = st.columns([1.2, 1], gap="large")

    with col_in:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("##### ✍️ Masukkan Teks untuk Dianalisis")

        EXAMPLES = {
            "pos": "Produknya sangat berkualitas, pengiriman cepat dan pelayanan seller ramah banget! Puas banget beli di sini.",
            "neg": "Barang tidak sesuai gambar, kualitas buruk dan pengiriman lama sekali. Sangat kecewa dan menyesal beli.",
            "net": "Produk sudah sampai sesuai pesanan. Lumayan lah untuk harganya."
        }
        c1, c2, c3 = st.columns(3)
        sel = ""
        with c1:
            if st.button("😊 Positif", use_container_width=True): sel = EXAMPLES["pos"]
        with c2:
            if st.button("😞 Negatif", use_container_width=True): sel = EXAMPLES["neg"]
        with c3:
            if st.button("😐 Netral",  use_container_width=True): sel = EXAMPLES["net"]

        if sel:
            st.session_state["txt"] = sel

        user_text = st.text_area("Teks:", value=st.session_state.get("txt",""),
                                  height=140, placeholder="Ketik ulasan Bahasa Indonesia...",
                                  label_visibility="collapsed", key="txt")

        b1, b2 = st.columns([2,1])
        with b1: run = st.button("🔍 Analisis Sentimen", use_container_width=True)
        with b2:
            if st.button("🗑️ Hapus", use_container_width=True):
                st.session_state["txt"] = ""
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col_out:
        if run and user_text.strip():
            with st.spinner("Menganalisis..."):
                label, conf, proba_pos, clean = predict(user_text, model, tokenizer, config)
            proba_neg = 1 - proba_pos
            pos_pct   = int(proba_pos * 100)
            neg_pct   = int(proba_neg * 100)
            emoji     = "😊" if label == "Positif" else "😞"
            badge     = "badge-pos" if label == "Positif" else "badge-neg"
            conf_color = "#22d3a8" if label=="Positif" else "#f87171"

            st.markdown("<div class='card2'>", unsafe_allow_html=True)
            st.markdown("##### 🎯 Hasil Analisis")
            st.markdown(f"<span class='emoji-big'>{emoji}</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;margin:12px 0'><span class='{badge}'>{label}</span></div>", unsafe_allow_html=True)

            st.markdown("**Distribusi Probabilitas**")
            st.markdown(f"""
            <div style='margin:8px 0'>
              <div style='display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;margin-bottom:4px'>
                <span>😊 Positif</span><span><b style='color:#22d3a8'>{pos_pct}%</b></span>
              </div>
              <div class='bar-wrap'><div class='bar-pos' style='width:{pos_pct}%'></div></div>
            </div>
            <div style='margin:8px 0'>
              <div style='display:flex;justify-content:space-between;font-size:12px;color:#94a3b8;margin-bottom:4px'>
                <span>😞 Negatif</span><span><b style='color:#f87171'>{neg_pct}%</b></span>
              </div>
              <div class='bar-wrap'><div class='bar-neg' style='width:{neg_pct}%'></div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='mbox' style='margin-top:12px'>
              <span class='mval' style='color:{conf_color}'>{conf:.1%}</span>
              <div class='mlbl'>Tingkat Kepercayaan</div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("🔎 Token setelah preprocessing"):
                st.code(clean or "(kosong setelah preprocessing)")
            st.markdown("</div>", unsafe_allow_html=True)

        elif run:
            st.warning("Masukkan teks terlebih dahulu!")
        else:
            st.markdown("""
            <div class='card' style='text-align:center;padding:40px 20px'>
              <span style='font-size:3rem'>🤔</span>
              <p style='color:#64748b;margin:12px 0 0 0'>Masukkan teks dan klik <b>Analisis Sentimen</b></p>
            </div>""", unsafe_allow_html=True)

# ── Tab 2 ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("##### 📦 Analisis Banyak Teks Sekaligus")
    st.markdown("<p style='color:#64748b;font-size:13px'>Upload CSV dengan kolom <code>text</code>, atau ketik satu per baris.</p>", unsafe_allow_html=True)

    mode = st.radio("Mode:", ["Upload CSV", "Ketik manual"], horizontal=True)
    texts = []
    if mode == "Upload CSV":
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up:
            dfu = pd.read_csv(up)
            if "text" in dfu.columns:
                texts = dfu["text"].astype(str).tolist()
                st.success(f"{len(texts)} baris ditemukan.")
            else:
                st.error("Kolom 'text' tidak ditemukan.")
    else:
        raw = st.text_area("Teks (satu per baris):", height=150)
        if raw.strip():
            texts = [t.strip() for t in raw.strip().split("\n") if t.strip()]

    if st.button("🚀 Jalankan Analisis Batch", use_container_width=True) and texts:
        rows = []
        bar  = st.progress(0)
        for i, t in enumerate(texts):
            lbl, cf, pp, _ = predict(t, model, tokenizer, config)
            rows.append({"Teks": t, "Sentimen": lbl, "Kepercayaan": f"{cf:.1%}", "Prob. Positif": round(pp,4)})
            bar.progress((i+1)/len(texts))
        bar.empty()
        df_r = pd.DataFrame(rows)
        pos_n = (df_r["Sentimen"]=="Positif").sum()
        neg_n = (df_r["Sentimen"]=="Negatif").sum()

        m1,m2,m3 = st.columns(3)
        with m1: st.markdown(f"<div class='mbox'><div class='mval'>{len(df_r)}</div><div class='mlbl'>Total</div></div>", unsafe_allow_html=True)
        with m2: st.markdown(f"<div class='mbox'><div class='mval' style='color:#22d3a8'>{pos_n}</div><div class='mlbl'>Positif</div></div>", unsafe_allow_html=True)
        with m3: st.markdown(f"<div class='mbox'><div class='mval' style='color:#f87171'>{neg_n}</div><div class='mlbl'>Negatif</div></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_r, use_container_width=True, height=350)
        st.download_button("⬇️ Download Hasil CSV", df_r.to_csv(index=False).encode(), "hasil_sentimen.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

# ── Tab 3 ─────────────────────────────────────────────────────────────────────
with tab3:
    history = load_history()

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("##### 📊 Informasi & Statistik Model")

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f"<div class='mbox'><div class='mval' style='font-size:1.2rem'>BiLSTM</div><div class='mlbl'>Arsitektur</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='mbox'><div class='mval'>{config.get('MAX_WORDS',10000):,}</div><div class='mlbl'>Vocab Size</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='mbox'><div class='mval'>{config.get('MAX_LEN',50)}</div><div class='mlbl'>Max Sequence</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if history:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        epochs = list(range(1, len(history["accuracy"]) + 1))
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Akurasi Training", "Loss Training"))
        fig.add_trace(go.Scatter(x=epochs, y=history["accuracy"],     name="Train Acc",  line=dict(color="#818cf8", width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_accuracy"], name="Val Acc",    line=dict(color="#22d3a8", width=2.5, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs, y=history["loss"],         name="Train Loss", line=dict(color="#f87171", width=2.5)), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"],     name="Val Loss",   line=dict(color="#fbbf24", width=2.5, dash="dot")), row=1, col=2)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8", height=360,
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=20,r=20,t=50,b=20),
        )
        for axis in ["xaxis","yaxis","xaxis2","yaxis2"]:
            fig.update_layout(**{axis: dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)")})

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
        st.info("Training history tidak ditemukan.")
