import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io
import os

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

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    model = tf.keras.models.load_model('best_cnn_model.keras')
    return model

@st.cache_data
def load_sample_data():
    """Load beberapa gambar contoh dari Fashion MNIST untuk demo."""
    import tensorflow as tf
    (_, _), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    X_test = X_test.astype('float32') / 255.0
    return X_test, y_test

def predict(model, img_array):
    """Prediksi dari array gambar (28x28, grayscale, 0-1)."""
    img = img_array.reshape(1, 28, 28, 1)
    probs = model.predict(img, verbose=0)[0]
    pred_idx = np.argmax(probs)
    return pred_idx, probs

def preprocess_uploaded(img):
    """Preprocess gambar yang diupload user → (28,28) grayscale, 0-1."""
    img = img.convert('L')          # ke grayscale
    img = img.resize((28, 28))      # resize ke 28x28
    arr = np.array(img).astype('float32') / 255.0
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

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 Test Accuracy",  "94.12%")
    col2.metric("📉 Test Loss",      "0.1908")
    col3.metric("🖼️ Total Dataset",  "70,000 gambar")
    col4.metric("🏷️ Jumlah Kelas",   "10 kategori")

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
        'Model': ['Dense NN (Baseline)', 'CNN (Main) ⭐'],
        'Test Accuracy': ['~88%', '94.12%'],
        'Test Loss':     ['~0.33', '0.1908'],
        'Parameter':     ['~670K', '~420K']
    })
    st.dataframe(results.set_index('Model'), use_container_width=True)
    st.caption("CNN lebih akurat dan lebih efisien (lebih sedikit parameter)")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: KLASIFIKASI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Klasifikasi Gambar":
    st.title("🔮 Klasifikasi Gambar Pakaian")
    st.divider()

    tab1, tab2 = st.tabs(["📤 Upload Gambar", "🎲 Coba Gambar Contoh"])

    model = load_model()

    # ── Tab 1: Upload ─────────────────────────────────────────────────────────
    with tab1:
        st.markdown("Upload gambar pakaian (JPG/PNG) — model akan memprediksi kategorinya.")
        st.info("💡 Gambar terbaik: latar belakang putih/polos, objek di tengah, grayscale atau berwarna")

        uploaded = st.file_uploader("Pilih gambar...", type=["jpg","jpeg","png"])

        if uploaded:
            img_pil = Image.open(uploaded)
            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(img_pil, caption="Gambar yang diupload", width=200)

            img_arr     = preprocess_uploaded(img_pil)
            pred_idx, probs = predict(model, img_arr)

            with col2:
                st.subheader(f"Hasil Prediksi: {CLASS_EMOJI[pred_idx]} {CLASS_NAMES[pred_idx]}")
                conf = probs[pred_idx] * 100
                if conf >= 80:
                    st.success(f"Confidence: **{conf:.1f}%** — Model sangat yakin!")
                elif conf >= 50:
                    st.warning(f"Confidence: **{conf:.1f}%** — Model cukup yakin")
                else:
                    st.error(f"Confidence: **{conf:.1f}%** — Model kurang yakin")

                st.markdown("**Probabilitas semua kelas:**")
                import pandas as pd
                prob_df = pd.DataFrame({
                    'Kategori': [f"{CLASS_EMOJI[i]} {CLASS_NAMES[i]}" for i in range(NUM_CLASSES)],
                    'Probabilitas': probs
                }).sort_values('Probabilitas', ascending=False)

                for _, row in prob_df.iterrows():
                    bar_pct = int(row['Probabilitas'] * 100)
                    color   = "#0052D9" if row['Kategori'].split(' ', 1)[1] == CLASS_NAMES[pred_idx] else "#E0E0E0"
                    st.markdown(
                        f"{row['Kategori']}: **{bar_pct}%** "
                        f"<div style='background:{color};height:8px;width:{bar_pct}%;border-radius:4px'></div>",
                        unsafe_allow_html=True
                    )

            # Tampilkan gambar setelah preprocessing
            st.divider()
            st.markdown("**Preview gambar setelah diproses model (28×28 grayscale):**")
            fig, ax = plt.subplots(figsize=(2, 2))
            ax.imshow(img_arr, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            st.pyplot(fig, use_container_width=False)

    # ── Tab 2: Contoh ─────────────────────────────────────────────────────────
    with tab2:
        st.markdown("Pilih kategori dan gambar contoh dari dataset Fashion MNIST:")

        X_test, y_test = load_sample_data()

        col1, col2 = st.columns(2)
        with col1:
            selected_class = st.selectbox(
                "Pilih kategori:",
                range(NUM_CLASSES),
                format_func=lambda x: f"{CLASS_EMOJI[x]} {CLASS_NAMES[x]}"
            )
        with col2:
            n_available = np.sum(y_test == selected_class)
            sample_no   = st.slider("Pilih gambar ke-", 1, min(20, n_available), 1)

        # Ambil gambar
        indices    = np.where(y_test == selected_class)[0]
        sample_idx = indices[sample_no - 1]
        img_arr    = X_test[sample_idx]

        pred_idx, probs = predict(model, img_arr)

        col_img, col_res = st.columns([1, 2])

        with col_img:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img_arr, cmap='gray')
            ax.set_title(f"Label: {CLASS_NAMES[selected_class]}", fontsize=10)
            ax.axis('off')
            st.pyplot(fig)

        with col_res:
            correct = pred_idx == selected_class
            status  = "✅ BENAR" if correct else "❌ SALAH"
            st.subheader(f"Prediksi: {CLASS_EMOJI[pred_idx]} {CLASS_NAMES[pred_idx]}")
            if correct:
                st.success(f"{status} — Confidence: {probs[pred_idx]*100:.1f}%")
            else:
                st.error(f"{status} — Model memprediksi {CLASS_NAMES[pred_idx]} "
                         f"(confidence {probs[pred_idx]*100:.1f}%)")

            # Top 3 prediksi
            st.markdown("**Top 3 prediksi:**")
            top3 = np.argsort(probs)[::-1][:3]
            for rank, idx in enumerate(top3):
                st.markdown(f"{rank+1}. {CLASS_EMOJI[idx]} {CLASS_NAMES[idx]} — **{probs[idx]*100:.1f}%**")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: EKSPLORASI DATASET
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Eksplorasi Dataset":
    st.title("📊 Eksplorasi Fashion MNIST Dataset")
    st.divider()

    X_test, y_test = load_sample_data()

    tab1, tab2, tab3 = st.tabs(["🖼️ Contoh Gambar", "📈 Distribusi", "📉 Hasil Evaluasi"])

    with tab1:
        st.subheader("Satu Contoh per Kategori")
        fig, axes = plt.subplots(2, 5, figsize=(14, 6))
        axes = axes.flatten()
        for i in range(NUM_CLASSES):
            idx = np.where(y_test == i)[0][0]
            axes[i].imshow(X_test[idx], cmap='gray')
            axes[i].set_title(f"{CLASS_EMOJI[i]} {CLASS_NAMES[i]}", fontsize=9)
            axes[i].axis('off')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab2:
        st.subheader("Distribusi Kelas di Test Set")
        import pandas as pd
        counts = pd.Series(y_test).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(12, 4))
        colors  = plt.cm.Set3(np.linspace(0, 1, NUM_CLASSES))
        ax.bar([f"{CLASS_EMOJI[i]}\n{CLASS_NAMES[i]}" for i in range(NUM_CLASSES)],
               counts.values, color=colors, alpha=0.85)
        ax.axhline(counts.mean(), color='red', linestyle='--',
                   label=f'Rata-rata ({counts.mean():.0f})')
        ax.set_title('Distribusi Kelas — Test Set (10,000 gambar)', fontsize=12)
        ax.set_ylabel('Jumlah Gambar')
        ax.legend()
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.success("Dataset sangat seimbang — setiap kelas punya 1,000 gambar di test set")

    with tab3:
        st.subheader("F1-Score per Kelas (Hasil Aktual)")
        f1_scores = {
            'T-shirt/top': 0.906, 'Trouser': 0.992, 'Pullover': 0.911,
            'Dress': 0.935,       'Coat': 0.898,     'Sandal': 0.985,
            'Shirt': 0.811,       'Sneaker': 0.975,  'Bag': 0.988,
            'Ankle boot': 0.973
        }
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_f1 = ['#1DB954' if v > 0.92 else '#FF6B35' if v > 0.88 else '#FF3B6B'
                     for v in f1_scores.values()]
        bars = ax.barh(list(f1_scores.keys()), list(f1_scores.values()),
                       color=colors_f1, alpha=0.85)
        ax.axvline(np.mean(list(f1_scores.values())), color='black',
                   linestyle='--', label=f"Rata-rata = {np.mean(list(f1_scores.values())):.3f}")
        ax.set_xlim(0.75, 1.0)
        ax.set_xlabel('F1-Score')
        ax.set_title('F1-Score per Kategori — CNN (Test Accuracy: 94.12%)', fontsize=12)
        ax.legend()
        for bar, val in zip(bars, f1_scores.values()):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.success("**Terbaik**: Trouser (F1=0.992)")
        col2.error("**Tersulit**: Shirt (F1=0.811)")
        col3.warning("**Paling tertukar**: Shirt ↔ T-shirt")

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
├── Conv2D(32, 3×3) + BatchNorm + ReLU
├── Conv2D(32, 3×3) + BatchNorm + ReLU
├── MaxPooling(2×2) + Dropout(0.25)
│
├── Conv2D(64, 3×3) + BatchNorm + ReLU
├── Conv2D(64, 3×3) + BatchNorm + ReLU
├── MaxPooling(2×2) + Dropout(0.25)
│
├── Flatten
├── Dense(256) + BatchNorm + ReLU
├── Dropout(0.5)
│
└── Dense(10, Softmax) → Output
        """, language="text")

    with col2:
        st.subheader("⚙️ Hyperparameter Training")
        import pandas as pd
        params = pd.DataFrame({
            'Parameter': ['Optimizer','Learning Rate','Batch Size',
                          'Max Epochs','EarlyStopping patience',
                          'ReduceLR factor','Loss Function'],
            'Nilai': ['Adam','0.001','128','50','7','0.5','Categorical Crossentropy']
        })
        st.dataframe(params.set_index('Parameter'), use_container_width=True)

        st.subheader("📈 Hasil Training")
        st.metric("Test Accuracy",  "94.12%", "+6.12% vs Dense NN")
        st.metric("Test Loss",      "0.1908")
        st.metric("Total Parameter","~420K")

    st.divider()
    st.subheader("💡 Kenapa CNN lebih baik dari Dense NN?")
    c1, c2, c3 = st.columns(3)
    c1.info("**Convolutional Layer**\n\nBelajar fitur lokal: tepi, tekstur, pola — tidak bisa dilakukan Dense NN")
    c2.info("**MaxPooling**\n\nMereduksi dimensi sambil mempertahankan fitur penting → lebih efisien")
    c3.info("**Parameter Sharing**\n\nFilter yang sama diterapkan ke seluruh gambar → lebih sedikit parameter")

    st.divider()
    st.subheader("🚀 Potensi Peningkatan")
    st.markdown("""
    - **Data Augmentation** (flip, zoom, rotation) → model lebih robust
    - **Transfer Learning** (ResNet50, EfficientNet) → akurasi >97%
    - **Fine-tuning Learning Rate** → konvergensi lebih cepat
    - **Label Smoothing** → mengurangi overconfidence pada prediksi
    """)
