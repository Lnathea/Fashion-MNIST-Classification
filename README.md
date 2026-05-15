# 👗 Fashion MNIST Image Classification

> End-to-end Deep Learning project — klasifikasi 10 kategori pakaian menggunakan CNN berbasis TensorFlow/Keras

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-93.93%25-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lnathea/Fashion-MNIST-Classification/blob/main/fashion_mnist_complete.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fashion-mnist-classification-g8w4yaleg6vy3y9kbpme3n.streamlit.app)

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
├── 📓 fashion_mnist_complete.ipynb   ← Notebook utama (Part 1 + 2 + 3)
├── 🌐 app.py                         ← Streamlit web app
├── 📦 fashion_model.onnx             ← Model CNN (format ONNX)
├── 🗃️ sample_images.npy              ← Sample gambar untuk demo
├── 🗃️ sample_labels.npy              ← Label sample gambar
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🔬 Alur Analisis

### Part 1 — Data Exploration & Preprocessing
- Load dataset langsung dari Keras (tidak perlu download manual)
- Visualisasi sampel gambar tiap kategori (6,000 gambar/kelas, sangat seimbang)
- Analisis distribusi pixel dan rata-rata gambar per kelas
- **Preprocessing**: Normalisasi (0–255 → 0–1), Reshape → (28,28,1), One-hot encoding
- Split: 48,000 train / 12,000 validation / 10,000 test

### Part 2 — Modeling & Training

**Model 1: Dense Neural Network (Baseline)**
```
Flatten(784) → Dense(512) → Dropout(0.3) → Dense(256) → Dropout(0.3) → Dense(10, Softmax)
Parameter: 535,818
```

**Model 2: CNN (Model Utama)**
```
Conv2D(32)×2 → MaxPool → Dropout(0.25)
Conv2D(64)×2 → MaxPool → Dropout(0.25)
Dense(256) → Dropout(0.5) → Dense(10, Softmax)
Parameter: 872,426
```

Training menggunakan:
- **EarlyStopping** (patience=7, restore best weights) — berhenti di epoch 28
- **ModelCheckpoint** (save best val_accuracy)
- **ReduceLROnPlateau** (factor=0.5, patience=3)

### Part 3 — Evaluasi & Visualisasi
- Confusion matrix (absolut + persentase per kelas)
- Classification report: Precision, Recall, F1 per kelas
- Visualisasi prediksi benar vs salah
- Analisis error — kelas yang paling sering tertukar
- Visualisasi filter CNN dan feature maps

---

## 🏆 Hasil

| Model | Test Accuracy | Test Loss | Parameter |
|---|---|---|---|
| Dense NN (Baseline) | 89.93% | 0.3170 | 535,818 |
| **CNN (Main)** ⭐ | **93.93%** | **0.1830** | **872,426** |

> CNN lebih akurat **+4.00%** dibanding Dense NN

### 📊 F1-Score per Kelas (Hasil Aktual)

| Kelas | F1-Score | Error Rate |
|---|---|---|
| 👖 Trouser | 0.993 | 1.0% |
| 👜 Bag | 0.989 | 1.3% |
| 👡 Sandal | 0.989 | 1.2% |
| 👟 Sneaker | 0.973 | 2.2% |
| 🥾 Ankle boot | 0.975 | 2.9% |
| 👗 Dress | 0.943 | 5.2% |
| 🧥 Pullover | 0.913 | 8.2% |
| 🧥 Coat | 0.910 | 9.5% |
| 👕 T-shirt/top | 0.888 | 11.5% |
| 👔 Shirt | 0.821 | **17.7%** ⚠️ |

---

## 💡 Key Insights

- 👖 **Trouser, Bag, Sandal** paling mudah diklasifikasi (F1 > 0.98) karena bentuknya unik dan berbeda jauh dari kelas lain
- 👔 **Shirt paling sulit** (error rate 17.7%) — sering tertukar dengan T-shirt/top (81 kali dari 1,000 gambar)
- 🎯 **9,393 dari 10,000 gambar** diprediksi benar
- 🧠 CNN lebih efisien: parameter lebih banyak tapi pola yang dipelajari jauh lebih relevan untuk data gambar

---

## 🛠️ Tech Stack

| Library | Kegunaan |
|---|---|
| `tensorflow` / `keras` | Build & train model CNN |
| `onnxruntime` | Inferensi model di Streamlit (ringan, tanpa TensorFlow) |
| `numpy` | Manipulasi array gambar |
| `matplotlib` | Visualisasi training history & confusion matrix |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Classification report |
| `streamlit` | Web app deployment |
| `pillow` | Image processing (upload gambar) |

---

## 🚀 Cara Menjalankan

**Opsi 1 — Streamlit Web App (Live Demo)**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fashion-mnist-classification-g8w4yaleg6vy3y9kbpme3n.streamlit.app)

**Opsi 2 — Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Lnathea/Fashion-MNIST-Classification/blob/main/fashion_mnist_complete.ipynb)

1. Upload `fashion_mnist_complete.ipynb` ke Google Colab
2. Aktifkan GPU: `Runtime → Change runtime type → T4 GPU`
3. `Runtime → Run all` — selesai dalam ~10 menit

**Opsi 3 — Lokal**
```bash
git clone https://github.com/Lnathea/Fashion-MNIST-Classification.git
cd Fashion-MNIST-Classification
pip install -r requirements.txt
streamlit run app.py
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
[![Streamlit](https://img.shields.io/badge/Live%20Demo-Streamlit-red?logo=streamlit)](https://fashion-mnist-classification-g8w4yaleg6vy3y9kbpme3n.streamlit.app)
