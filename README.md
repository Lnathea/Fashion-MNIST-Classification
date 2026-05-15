# 👗 Fashion MNIST Image Classification

> End-to-end Deep Learning project — klasifikasi 10 kategori pakaian menggunakan CNN berbasis TensorFlow/Keras

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?logo=keras&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.12%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Tentang Project

Project ini membangun model **Convolutional Neural Network (CNN)** untuk mengklasifikasikan gambar pakaian dari **Fashion MNIST Dataset** oleh Zalando Research.

Motivasi utama: *"Seberapa akurat model CNN sederhana bisa membedakan 10 kategori pakaian — dan di mana model ini masih gagal?"*

---

## 📊 Dataset

| Info | Detail |
|---|---|
| Sumber | [Fashion MNIST — Zalando Research](https://github.com/zalandoresearch/fashion-mnist) |
| Total gambar | 70,000 (60K train + 10K test) |
| Ukuran gambar | 28 × 28 pixel (grayscale) |
| Jumlah kelas | 10 kategori |
| Built-in | ✅ Tersedia langsung di `tf.keras.datasets` |

### 🏷️ 10 Kategori

| Label | Kategori | Label | Kategori |
|---|---|---|---|
| 0 | 👕 T-shirt/top | 5 | 👡 Sandal |
| 1 | 👖 Trouser | 6 | 👔 Shirt |
| 2 | 🧥 Pullover | 7 | 👟 Sneaker |
| 3 | 👗 Dress | 8 | 👜 Bag |
| 4 | 🧥 Coat | 9 | 🥾 Ankle boot |

---

## 🗂️ Struktur Project

```
fashion-mnist-classification/
│
├── 📓 fashion_mnist_complete.ipynb   ← Notebook utama (semua part)
├── 🌐 fashion_app.py                 ← Streamlit web app
├── 📦 best_cnn_model.keras           ← Model CNN terbaik
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🔬 Alur Analisis

### Part 1 — Data Exploration & Preprocessing
- Load dataset langsung dari Keras (tidak perlu download manual)
- Visualisasi sampel gambar per kategori
- Analisis distribusi kelas → dataset sangat seimbang (6,000/kelas)
- Analisis distribusi pixel dan rata-rata gambar per kelas
- **Preprocessing**: Normalisasi (0–255 → 0–1), Reshape (28,28) → (28,28,1), One-hot encoding

### Part 2 — Modeling & Training

**Model 1: Dense Neural Network (Baseline)**
```
Flatten → Dense(512) → Dropout → Dense(256) → Dropout → Dense(10, Softmax)
```

**Model 2: CNN (Model Utama)**
```
Conv2D(32) × 2 → MaxPool → Dropout
Conv2D(64) × 2 → MaxPool → Dropout
Dense(256) → Dropout → Dense(10, Softmax)
```

Training menggunakan:
- **EarlyStopping** (patience=7, restore best weights)
- **ModelCheckpoint** (save best model)
- **ReduceLROnPlateau** (factor=0.5, patience=3)

### Part 3 — Evaluasi & Visualisasi
- Confusion matrix (absolut + persentase per kelas)
- Classification report: Precision, Recall, F1 per kelas
- Visualisasi prediksi benar vs salah
- Analisis error — pasangan kelas yang paling sering tertukar
- Visualisasi filter CNN dan feature maps

---

## 🏆 Hasil

| Model | Test Accuracy | Test Loss | Parameter |
|---|---|---|---|
| Dense NN (Baseline) | ~88% | ~0.33 | ~670K |
| **CNN (Main)** ⭐ | **94.12%** | **0.1908** | **~420K** |

> CNN lebih akurat **dan** lebih efisien dari Dense NN

### 📊 F1-Score per Kelas

| Kelas | F1-Score | Keterangan |
|---|---|---|
| 👖 Trouser | 0.992 | 🏆 Terbaik |
| 👜 Bag | 0.988 | ✅ Sangat Baik |
| 👡 Sandal | 0.985 | ✅ Sangat Baik |
| 👟 Sneaker | 0.975 | ✅ Sangat Baik |
| 🥾 Ankle boot | 0.973 | ✅ Sangat Baik |
| 👗 Dress | 0.935 | ✅ Baik |
| 🧥 Pullover | 0.911 | ✅ Baik |
| 👕 T-shirt/top | 0.906 | ✅ Baik |
| 🧥 Coat | 0.898 | 🟡 Cukup |
| 👔 Shirt | 0.811 | ⚠️ Tersulit |

> Shirt paling sulit karena secara visual mirip dengan T-shirt dan Pullover

---

## 💡 Key Insights

- 👖 **Trouser, Bag, Sneaker** paling mudah diklasifikasi karena bentuknya unik
- 👔 **Shirt** paling sering salah — tertukar dengan T-shirt/top karena sangat mirip secara visual
- 🧠 **CNN lebih efisien dari Dense NN** — parameter lebih sedikit tapi akurasi lebih tinggi
- 📦 **BatchNormalization + Dropout** sangat membantu mencegah overfitting

---

## 🛠️ Tech Stack

| Library | Versi | Kegunaan |
|---|---|---|
| `tensorflow` | 2.x | Build & train model |
| `keras` | Built-in | API neural network |
| `numpy` | 1.23+ | Manipulasi array gambar |
| `matplotlib` | 3.6+ | Visualisasi |
| `seaborn` | 0.12+ | Confusion matrix |
| `scikit-learn` | 1.2+ | Classification report |
| `streamlit` | 1.28+ | Web app deployment |
| `pillow` | 9.0+ | Image processing |

---

## 🚀 Cara Menjalankan

**Opsi 1 — Google Colab (Rekomendasi)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lnathea/Fashion-MNIST-Classification/blob/main/fashion_mnist_complete.ipynb)

1. Upload `fashion_mnist_complete.ipynb` ke Google Colab
2. Aktifkan GPU: `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all` — selesai dalam ~10 menit

**Opsi 2 — Lokal**
```bash
git clone https://github.com/Lnathea/Fashion-MNIST-Classification.git
cd Fashion-MNIST-Classification
pip install -r requirements.txt
streamlit run fashion_app.py
```

---

## 📄 Referensi

- Han Xiao et al., *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms*, 2017
- [Fashion MNIST GitHub Repository](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)

---

## 👤 Author

**Muhammad Afriza Hidayat**
Mahasiswa Teknik Informatika | Data & AI Enthusiast

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/afriza)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Lnathea)
