# RVL-CDIP Document AI & Decision Threshold Tuning
### Klasifikasi Multimodal Dokumen Bisnis & Mitigasi Risiko Finansial (Tugas 06 — Integrasi Menuju UTS)

**Mata Kuliah:** Pembelajaran Mesin (S1 Informatika - 3 SKS)  
**Kelas Gabungan:** KG.043.10.A (12.8A.10 & 12.8A.12)  
**Dosen Pengampu:** Dr. Irwansyah Saputra, M.Kom. (NIP: 201909149)  
**Semester:** Gasal 2026/2027 — Universitas Nusa Mandiri  

---

### Profil Tim Peneliti (Kelompok 05)
| No | NIM | Nama Mahasiswa | Peran / Pembagian Tugas Teknis |
|:---:|:---:|:---|:---|
| 1 | **12250131** | **Jakariyana** | Koordinator Tim & Pipeline Modeling / Threshold Tuning |
| 2 | **12250167** | **Nopa Aji Prasetio** | Eksperimen Hyperparameter Optuna & Cross Validation |
| 3 | **12250153** | **Muhammad Taufik** | Evaluasi Confusion Matrix, Bab I–III Proposal UTS & Analisis Risiko Bisnis |

---

## 1. Ringkasan Riset & Studi Kasus
* **Instansi Studi Kasus:** PT Mobil Sejuta Umat (Manufaktur Otomotif Multinasional)
* **Objek Riset:** Otomasi pengarsipan berkas pindaian citra dokumen fisik (*Invoice, Form, Letter, Memo*) berbasis representasi multimodal (vektor teks TF-IDF N-gram dan parameter struktural spasial).
* **Dataset Ground Truth:** Benchmark terbuka industri RVL-CDIP (*Ryerson Vision Lab Complex Document Information Processing*) dengan 800 dokumen seimbang (200 sampel per kelas) guna mematuhi etika kepatuhan *Non-Disclosure Agreement* (NDA) perusahaan.
* **Model Terpilih:** *Tuned Random Forest Classifier* terintegrasi dalam *scikit-learn ColumnTransformer* bebas kebocoran data (*anti-leakage*).

---

## 2. Kinerja Model & Kalibrasi Ambang Batas (Tugas 06)
* **Optimasi Bayesian:** 30 Iterasi Trial dengan Optuna (`TPESampler`) pada Stratified 5-Fold Cross Validation.
* **Parameter Terbaik:** `n_estimators=225`, `max_depth=16`, `min_samples_split=9`, `min_samples_leaf=1`, `max_features='sqrt'`.
* **Kinerja Kunci:**
  * **Test Set Accuracy:** **70.63%**
  * **Macro F1-Score:** **0.7017**
  * **ROC-AUC Macro (One-vs-Rest):** **0.9012**
  * **Invoice PR-AUC:** **0.8148**
* **Kalibrasi Ambang Batas (Decision Threshold = 0.35):**
  * Memangkas kasus *False Negative* (FN) faktur tagihan dari 15 kasus menjadi 9 kasus (**reduksi 40%**).
  * Menghindari denda keterlambatan pembayaran vendor rata-rata Rp 2.500.000 per kejadian, menghasilkan **efisiensi biaya operasional bersih Rp 14.940.000** pada data pengujian.

---

## 3. Struktur Berkas Repositori
```text
├── app.py                         # Antarmuka Web Interaktif Streamlit (Dashboard & Threshold Slider)
├── api.py                         # Layanan Microservice REST API Mandiri (FastAPI)
├── ocr_preprocessor.py            # Modul Pra-Pemrosesan Citra (Binarisasi Otsu & Morfologi) & Text Cleaner
├── tugas06_optuna_threshold.py    # Skrip Training Optuna, 5-Fold CV, Evaluasi Matrix & Visualisasi
├── model_final_tugas06.joblib     # Artefak Serialized Scikit-Learn Model Pipeline Terkalibrasi
├── model_metadata_tugas06.json    # Rekap Metadata Metrik Evaluasi, Parameter, & Simulasi Threshold
├── optuna_trials.json             # Log Riwayat Lengkap 30 Iterasi Bayesian Optimization
├── requirements.txt               # Daftar Dependensi Pustaka Lingkungan Produksi
├── README.md                      # Panduan Reprodusibilitas & Cara Menjalankan Sistem
├── sample_documents/              # Sampel Citra Scan Pindaian Asli RVL-CDIP (.tif) untuk Pengujian
│   ├── 0000023361.tif             # Sampel: Faktur Tagihan (Invoice)
│   ├── 0000953276.tif             # Sampel: Formulir Isian (Form)
│   ├── 0000000700.tif             # Sampel: Surat Resmi (Letter)
│   └── 0000001053.tif             # Sampel: Memorandum Internal (Memo)
├── gambar1_confusion_matrix.png   # Heatmap Confusion Matrix Multikelas 4x4 & Matriks Biner Th=0.35
├── gambar2_roc_pr_curve.png       # Kurva Multi-Class ROC-AUC (OVR) & Precision-Recall Curve Invoice
├── gambar3_optuna_optimization.png# Riwayat Konvergensi Optuna & Hyperparameter Importance
├── gambar4_flowchart_otsu_morphology.png # Diagram Alir Pra-Pemrosesan Citra Pra-OCR
├── gambar5_web_app_ui.png         # Tangkapan Layar Prototipe Web (Form Masukan & Slider)
└── gambar6_web_app_results.png    # Tangkapan Layar Panel Prediksi, Confidence & Explainable AI (XAI)
```

---

## 4. Panduan Menjalankan Sistem (Streamlit Web App)

### Langkah 1: Instalasi Dependensi
Buka terminal pada direktori proyek, lalu jalankan:
```bash
pip install -r requirements.txt
```

### Langkah 2: Menjalankan Aplikasi Web
Jalankan perintah berikut:
```bash
streamlit run app.py
```
Aplikasi interaktif akan otomatis terbuka pada peramban web: **`http://localhost:8501`**

### Fitur Utama Antarmuka:
1. **Interactive Decision Threshold Slider (0.10 s.d. 0.90):** Menggeser ambang batas klasifikasi kelas kritis *Invoice* secara langsung untuk mengamati mitigasi risiko *False Negative*.
2. **Fleksibilitas Input Citra:** Uji instan dengan sampel dokumen RVL-CDIP bawaan, upload berkas pindaian mandiri (`.tif`, `.png`, `.jpg`, `.pdf`), atau input teks manual.
3. **Zero-Crash OCR Fallback:** Terintegrasi engine OCR Tesseract dengan proteksi fail-safe jika OCR engine sistem belum terpasang.
4. **Visualisasi Explainability (XAI):** Menampilkan skor *confidence*, probabilitas per kategori, dan kata kunci penentu prediksi.

---

## 5. Menjalankan Uji Replikasi Model & Optuna
Untuk mereplikasi proses pencarian hiperparameter Optuna dan evaluasi threshold:
```bash
python tugas06_optuna_threshold.py
```

---

## 6. Referensi Ilmiah Terpublikasi
1. **Huang, Y., et al. (2022).** *LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking.* ACM MM '22, pp. 4083–4092. DOI: [10.1145/3503161.3548112](https://doi.org/10.1145/3503161.3548112).
2. **Kim, G., et al. (2022).** *OCR-free Document Understanding Transformer (Donut).* ECCV 2022, LNCS vol. 13688, pp. 498–517. Springer Nature. DOI: [10.1007/978-3-031-19815-1_29](https://doi.org/10.1007/978-3-031-19815-1_29).
3. **Harley, A. W., et al. (2015).** *Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval.* ICDAR 2015, pp. 991–995. IEEE. DOI: [10.1109/ICDAR.2015.7333910](https://doi.org/10.1109/ICDAR.2015.7333910).
4. **Akiba, T., et al. (2019).** *Optuna: A Next-generation Hyperparameter Optimization Framework.* ACM KDD '19, pp. 2623–2631. DOI: [10.1145/3292500.3330701](https://doi.org/10.1145/3292500.3330701).
5. **Breiman, L. (2001).** *Random Forests.* Machine Learning, 45(1), pp. 5–32. Springer. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).
