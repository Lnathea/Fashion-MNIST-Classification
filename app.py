import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👗",
    layout="wide"
)

CLASS_NAMES = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Ankle boot']
CLASS_EMOJI = ['👕','👖','🧥','👗','🧥','👡','👔','👟','👜','🥾']
NUM_CLASSES  = 10

# ── Load Model (ONNX) ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    session = ort.InferenceSession('fashion_model.onnx')
    return session

@st.cache_data
def load_sample_data():
    X = np.load('sample_images.npy')  # (200, 28, 28, 1) float32
    y = np.load('sample_labels.npy')  # (200,) int32
    return X, y

def predict(session, img_array):
    img        = img_array.reshape(1, 28, 28, 1).astype(np.float32)
    input_name = session.get_inputs()[0].name
    probs      = session.run(None, {input_name: img})[0][0]
    pred_idx   = int(np.argmax(probs))
    return pred_idx, probs

def preprocess_uploaded(img):
    img = img.convert('L')
    img = img.resize((28, 28))
    arr = np.array(img).astype(np.float32) / 255.0
    return arr

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("👗 Fashion Classifier")
st.sidebar.markdown("Klasifikasi pakaian dengan **CNN** berbasis Fashion MNIST")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigasi",
    ["🏠 Home", "🔮 Klasifikasi Gambar", "📊 Eksplorasi Dataset", "🧠 Tentang Model"]
)

st.sidebar.divider()
st.sidebar.markdown("**Muhammad Afriza Hidayat**")
st.sidebar.markdown("Mahasiswa Teknik Informatika | Data & AI Enthusiast")
st.sidebar.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/afriza) [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://github.com/Lnathea)")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("👗 Fashion MNIST Image Classifier")
    st.markdown("**Klasifikasi 10 kategori pakaian menggunakan Convolutional Neural Network (CNN)**")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Test Accuracy", "94.12%")
    col2.metric("📉 Test Loss",     "0.1908")
    col3.metric("🖼️ Total Dataset", "70,000 gambar")
    col4.metric("🏷️ Jumlah Kelas",  "10 kategori")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("📌 Tentang Project")
        st.markdown("""
        Project ini membangun model CNN untuk mengklasifikasikan gambar pakaian
        dari **Fashion MNIST Dataset** oleh Zalando Research.

        **Pipeline yang dibangun:**
        - Data Exploration & Preprocessing
        - Dense NN (baseline) vs CNN (model utama)
        - Evaluasi: Confusion Matrix, F1-Score per kelas
        - Visualisasi filter dan feature maps CNN

        **Model terbaik: CNN (Test Accuracy = 94.12%)**
        """)

    with col_b:
        st.subheader("🏷️ 10 Kategori Pakaian")
        cols = st.columns(2)
        for i, (name, emoji) in enumerate(zip(CLASS_NAMES, CLASS_EMOJI)):
            cols[i % 2].markdown(f"{emoji} **{i}** — {name}")

    st.divider()
    st.subheader("🏆 Perbandingan Model")
    import pandas as pd
    results = pd.DataFrame({
        'Model'         : ['Dense NN (Baseline)', 'CNN (Main) ⭐'],
        'Test Accuracy' : ['~88%', '94.12%'],
        'Test Loss'     : ['~0.33', '0.1908'],
        'Parameter'     : ['~670K', '~420K']
    })
    st.dataframe(results.set_index('Model'), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: KLASIFIKASI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Klasifikasi Gambar":
    st.title("🔮 Klasifikasi Gambar Pakaian")
    st.divider()

    session = load_model()
    tab1, tab2 = st.tabs(["📤 Upload Gambar", "🎲 Coba Gambar Contoh"])

    # ── Tab 1: Upload ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("Upload gambar pakaian (JPG/PNG) — model akan memprediksi kategorinya.")
        st.info("💡 Gambar terbaik: latar belakang putih/polos, objek di tengah")

        uploaded = st.file_uploader("Pilih gambar...", type=["jpg","jpeg","png"])

        if uploaded:
            img_pil = Image.open(uploaded)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(img_pil, caption="Gambar yang diupload", width=200)
                img_arr     = preprocess_uploaded(img_pil)
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                ax.imshow(img_arr, cmap='gray')
                ax.set_title("Setelah preprocessing\n(28×28 grayscale)", fontsize=8)
                ax.axis('off')
                st.pyplot(fig, use_container_width=False)

            pred_idx, probs = predict(session, img_arr)

            with col2:
                st.subheader(f"Hasil: {CLASS_EMOJI[pred_idx]} {CLASS_NAMES[pred_idx]}")
                conf = probs[pred_idx] * 100
                if conf >= 80:
                    st.success(f"Confidence: **{conf:.1f}%** — Model sangat yakin!")
                elif conf >= 50:
                    st.warning(f"Confidence: **{conf:.1f}%** — Model cukup yakin")
                else:
                    st.error(f"Confidence: **{conf:.1f}%** — Model kurang yakin")

                st.markdown("**Probabilitas semua kelas:**")
                sorted_idx = np.argsort(probs)[::-1]
                for idx in sorted_idx:
                    bar_pct = int(probs[idx] * 100)
                    label   = f"{CLASS_EMOJI[idx]} {CLASS_NAMES[idx]}"
                    color   = "#0052D9" if idx == pred_idx else "#E8E8E8"
                    st.markdown(
                        f"{label}: **{bar_pct}%**"
                        f"<div style='background:{color};height:8px;width:{max(bar_pct,2)}%;"
                        f"border-radius:4px;margin-bottom:4px'></div>",
                        unsafe_allow_html=True
                    )

    # ── Tab 2: Contoh ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("Pilih kategori dan gambar contoh dari dataset Fashion MNIST:")
        X_samples, y_samples = load_sample_data()

        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "Pilih kategori:",
                range(NUM_CLASSES),
                format_func=lambda x: f"{CLASS_EMOJI[x]} {CLASS_NAMES[x]}"
            )
        with col2:
            sample_no = st.slider("Pilih gambar ke-", 1, 20, 1)

        mask      = y_samples == selected_class
        class_imgs= X_samples[mask]
        img_arr   = class_imgs[sample_no - 1]

        pred_idx, probs = predict(session, img_arr)
        col_img, col_res = st.columns([1, 2])

        with col_img:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img_arr.reshape(28, 28), cmap='gray')
            ax.set_title(f"Label: {CLASS_NAMES[selected_class]}", fontsize=10)
            ax.axis('off')
            st.pyplot(fig)

        with col_res:
            correct = pred_idx == selected_class
            st.subheader(f"Prediksi: {CLASS_EMOJI[pred_idx]} {CLASS_NAMES[pred_idx]}")
            if correct:
                st.success(f"✅ BENAR — Confidence: {probs[pred_idx]*100:.1f}%")
            else:
                st.error(f"❌ SALAH — Model prediksi {CLASS_NAMES[pred_idx]} "
                         f"({probs[pred_idx]*100:.1f}%)")

            st.markdown("**Top 3 prediksi:**")
            for rank, idx in enumerate(np.argsort(probs)[::-1][:3]):
                st.markdown(f"{rank+1}. {CLASS_EMOJI[idx]} {CLASS_NAMES[idx]} — **{probs[idx]*100:.1f}%**")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EKSPLORASI DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Eksplorasi Dataset":
    st.title("📊 Eksplorasi Fashion MNIST Dataset")
    st.divider()

    X_samples, y_samples = load_sample_data()
    tab1, tab2 = st.tabs(["🖼️ Contoh Gambar", "📉 Hasil Evaluasi"])

    with tab1:
        st.subheader("Satu Contoh per Kategori")
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        axes = axes.flatten()
        for i in range(NUM_CLASSES):
            img = X_samples[y_samples == i][0].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"{CLASS_EMOJI[i]} {CLASS_NAMES[i]}", fontsize=9)
            axes[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        st.subheader("Variasi dalam Satu Kategori")
        sel = st.selectbox("Pilih kategori:", range(NUM_CLASSES),
                           format_func=lambda x: f"{CLASS_EMOJI[x]} {CLASS_NAMES[x]}",
                           key="var_sel")
        cat_imgs = X_samples[y_samples == sel]
        fig2, axes2 = plt.subplots(2, 10, figsize=(16, 4))
        axes2 = axes2.flatten()
        for i in range(min(20, len(cat_imgs))):
            axes2[i].imshow(cat_imgs[i].reshape(28, 28), cmap='gray')
            axes2[i].axis('off')
        plt.suptitle(f"20 Variasi: {CLASS_NAMES[sel]}", fontsize=12)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with tab2:
        st.subheader("F1-Score per Kelas (Hasil Aktual)")
        import pandas as pd
        f1_data = {
            'T-shirt/top': 0.906, 'Trouser': 0.992, 'Pullover': 0.911,
            'Dress': 0.935,       'Coat': 0.898,     'Sandal': 0.985,
            'Shirt': 0.811,       'Sneaker': 0.975,  'Bag': 0.988,
            'Ankle boot': 0.973
        }
        fig3, ax = plt.subplots(figsize=(10, 5))
        colors = ['#1DB954' if v > 0.92 else '#FF6B35' if v > 0.88 else '#FF3B6B'
                  for v in f1_data.values()]
        bars = ax.barh(list(f1_data.keys()), list(f1_data.values()),
                       color=colors, alpha=0.85)
        ax.axvline(np.mean(list(f1_data.values())), color='black', linestyle='--',
                   label=f"Rata-rata = {np.mean(list(f1_data.values())):.3f}")
        ax.set_xlim(0.75, 1.0)
        ax.set_title('F1-Score per Kategori — CNN (94.12%)', fontsize=12)
        ax.legend()
        for bar, val in zip(bars, f1_data.values()):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.success("**Terbaik**: Trouser (F1=0.992)")
        c2.error("**Tersulit**: Shirt (F1=0.811)")
        c3.warning("**Paling tertukar**: Shirt ↔ T-shirt")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: TENTANG MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Tentang Model":
    st.title("🧠 Tentang Model CNN")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏗️ Arsitektur CNN")
        st.code("""
Input (28 × 28 × 1)
│
├── Conv2D(32) + BatchNorm + ReLU
├── Conv2D(32) + BatchNorm + ReLU
├── MaxPooling(2×2) + Dropout(0.25)
│
├── Conv2D(64) + BatchNorm + ReLU
├── Conv2D(64) + BatchNorm + ReLU
├── MaxPooling(2×2) + Dropout(0.25)
│
├── Flatten
├── Dense(256) + BatchNorm + ReLU
├── Dropout(0.5)
│
└── Dense(10, Softmax)
        """, language="text")

    with col2:
        st.subheader("⚙️ Training Setup")
        import pandas as pd
        params = pd.DataFrame({
            'Parameter': ['Optimizer','Learning Rate','Batch Size',
                          'Max Epochs','Early Stopping','Loss Function'],
            'Nilai': ['Adam','0.001','128','50','patience=7','Categorical Crossentropy']
        })
        st.dataframe(params.set_index('Parameter'), use_container_width=True)

        st.subheader("📈 Hasil")
        st.metric("Test Accuracy",  "94.12%", "+6% vs Dense NN")
        st.metric("Test Loss",      "0.1908")

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.info("**Conv Layer**\n\nBelajar fitur visual: tepi, tekstur, pola")
    c2.info("**MaxPooling**\n\nReduksi dimensi, pertahankan fitur penting")
    c3.info("**BatchNorm + Dropout**\n\nCegah overfitting, training lebih stabil")
