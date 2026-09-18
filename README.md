# Skripsi: Klasifikasi Tingkat Kematangan Buah Kopi Menggunakan Citra Digital Berbasis CNN MobileNetV2

**Penulis:** Yohanes Darman Tala (NIM 22/23492/TP)
**Program Studi:** Teknologi Pertanian, Institut Pertanian Stiper (INSTIPER) Yogyakarta

## Ringkasan

Repository ini memuat notebook pelatihan model untuk skripsi klasifikasi tingkat kematangan buah kopi (mentah, setengah matang, matang) menggunakan Convolutional Neural Network (CNN) dengan arsitektur MobileNetV2 berbasis transfer learning.

## Isi Notebook

Notebook `PELATIHAN_ULANG_Skenario_1-4.ipynb` mencakup seluruh proses penelitian:

1. Persiapan lingkungan dan pemuatan dataset (900 citra, 3 kelas, kondisi kebun)
2. Pra-pemrosesan citra (resize 224x224, normalisasi)
3. Pembagian data latih/validasi/uji
4. **Skenario 1** — MobileNetV2 pra-latih (baseline, tanpa augmentasi)
5. **Skenario 2** — MobileNetV2 pra-latih dengan augmentasi data
6. **Skenario 3** — MobileNetV2 dengan augmentasi + fine-tuning 30 lapisan terakhir
7. **Skenario 4** — CNN sederhana dari nol (pembanding)
8. Evaluasi model (confusion matrix, precision, recall, F1-score) pada data uji
9. Visualisasi Grad-CAM untuk interpretasi fokus perhatian model

## Lingkungan

- Google Colab, GPU Tesla T4
- TensorFlow/Keras 2.20.0
