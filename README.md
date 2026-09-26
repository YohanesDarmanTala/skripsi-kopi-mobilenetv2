# Klasifikasi Tingkat Kematangan Buah Kopi dengan CNN MobileNetV2

Notebook pelatihan dan evaluasi untuk skripsi **"Klasifikasi Tingkat Kematangan Buah Kopi Menggunakan Citra Digital Convolutional Neural Network (CNN) MobileNetV2"**.

**Penulis:** Yohanes Darman Tala (NIM 22/23492/TP)
**Program Studi:** Teknik Pertanian, Fakultas Teknologi Pertanian
**Institusi:** Institut Pertanian Stiper (INSTIPER) Yogyakarta

---

## Ringkasan Penelitian

Penelitian ini mengklasifikasikan buah kopi ke dalam tiga tingkat kematangan berdasarkan citra digital:

| Kelas | Label | Deskripsi warna |
|---|---|---|
| Mentah | `green` | Kulit buah masih hijau |
| Setengah Matang | `yellowish` | Warna peralihan, hijau kekuningan hingga kemerahan |
| Matang | `red` | Kulit buah sudah merah penuh |

Pendekatan utama adalah **transfer learning** menggunakan arsitektur **MobileNetV2** yang dilatih sebelumnya pada ImageNet, dibandingkan dengan CNN sederhana yang dilatih dari nol sebagai baseline pembanding. Empat skenario pelatihan diuji secara bertingkat untuk mengisolasi pengaruh augmentasi data dan fine-tuning terhadap performa model.

**Hasil akhir:** model terbaik (Skenario 3 — augmentasi + fine-tuning 30 lapisan terakhir) mencapai **akurasi 87%** pada 135 citra data uji, dengan ukuran berkas 22,52 MB dan waktu inferensi rata-rata 103,33 ms per citra.

---

## Dataset

- **Sumber:** [Coffee Cherry Ripeness Dataset](https://www.kaggle.com/) oleh Haris Yunanda (Kaggle), lisensi Open Database License (ODbL).
- **Jumlah citra:** 900 citra, terbagi rata 300 citra per kelas.
- **Ukuran input model:** 224 × 224 piksel, RGB.
- **Pembagian data:** *stratified split* 70% latih : 15% validasi : 15% uji, dengan `random_state = 42` agar dapat direproduksi.

| Subset | Jumlah total | Per kelas |
|---|---|---|
| Latih | 630 citra | 210 |
| Validasi | 135 citra | 45 |
| Uji | 135 citra | 45 |

Struktur folder dataset yang diharapkan pada Google Drive (lihat SEL 2–3 pada notebook):

```
Skripsi_Kopi/
└── dataset_kopi/
    ├── MENTAH (green)/
    ├── SETENGAH MATANG (yellowish)/
    └── MATANG (red)/
```

---

## Empat Skenario Pengujian

Keempat skenario dirancang bertingkat agar pengaruh tiap perlakuan (augmentasi, fine-tuning, bobot pra-latih) dapat diamati secara terpisah melalui perbandingan berpasangan.

| Skenario | Arsitektur | Augmentasi | Fine-Tuning | Keterangan |
|---|---|---|---|---|
| **Skenario 1 — Baseline** | MobileNetV2 (bobot ImageNet, dibekukan) | Tidak | Tidak | Titik acuan tanpa perlakuan tambahan |
| **Skenario 2 — Augmentasi** | MobileNetV2 (bobot ImageNet, dibekukan) | Ya | Tidak | Mengisolasi pengaruh augmentasi |
| **Skenario 3 — Fine-Tuning** | MobileNetV2 (30 lapisan terakhir dibuka) | Ya | Ya (`learning_rate = 1e-4`) | **Model terbaik**; mengisolasi pengaruh fine-tuning |
| **Skenario 4 — CNN Sederhana** | CNN 3 blok konvolusi, dilatih dari nol | Ya | — | Pembanding untuk mengukur manfaat transfer learning |

Augmentasi data latih meliputi rotasi (±20°), zoom (0,8–1,2×), flip horizontal dan vertikal, serta penyesuaian kecerahan (rentang 0,8–1,2). Ukuran batch tetap 32 pada seluruh skenario, dengan 20 batch per epoch (19 batch berisi 32 citra, satu batch sisa berisi 22 citra) — jumlah batch dibaca otomatis oleh Keras dari panjang generator data, bukan dihitung manual, sehingga tidak ada citra latih yang tertinggal pada tiap epoch.

---

## Hasil Akhir — Perbandingan Empat Skenario

Performa pada 135 citra data uji (tidak pernah dilihat selama pelatihan):

| Skenario | Akurasi | Macro F1 | F1 Mentah | F1 Setengah Matang | F1 Matang |
|---|---|---|---|---|---|
| S1 — Baseline | 0,81 | 0,81 | 0,88 | 0,76 | 0,81 |
| S2 — Augmentasi | 0,85 | 0,85 | 0,89 | 0,80 | 0,86 |
| **S3 — Fine-Tuning** | **0,87** | **0,87** | **0,91** | **0,81** | **0,90** |
| S4 — CNN Sederhana | 0,67 | 0,67 | 0,76 | 0,53 | 0,72 |

Setiap perlakuan yang ditambahkan secara bertingkat pada MobileNetV2 menaikkan akurasi (0,81 → 0,85 → 0,87). Ketiga skenario berbasis bobot pra-latih ImageNet unggul 14–20 poin atas model yang dilatih dari nol, menunjukkan pentingnya transfer learning pada jumlah data latih yang terbatas (630 citra).

### Confusion Matrix — Model Terbaik (Skenario 3)

```
                    Diprediksi
                Mentah  Setengah  Matang
Aktual Mentah      43      0        2
Aktual Setengah     7     31        7
Aktual Matang        0     1       44
```

118 dari 135 citra uji (87%) diklasifikasikan benar. Kesalahan terpusat pada kelas Setengah Matang (14 dari 17 kesalahan, ±82%), sejalan dengan sifat kematangan buah kopi yang berlangsung sebagai proses biologis berkelanjutan tanpa batas warna yang tegas antar tahap. Praktis tidak ada kesalahan antara dua kelas yang paling berjauhan (Mentah ↔ Matang): hanya 2 dari 135 citra.

<details>
<summary>Confusion matrix skenario lain (klik untuk membuka)</summary>

**Skenario 1 — Baseline** (110/135 benar)
```
[[40  3  2]
 [ 5 34  6]
 [ 1  8 36]]
```

**Skenario 2 — Augmentasi** (115/135 benar)
```
[[41  2  2]
 [ 5 33  7]
 [ 1  3 41]]
```

**Skenario 4 — CNN Sederhana** (91/135 benar)
```
[[31  9  5]
 [ 4 21 20]
 [ 2  4 39]]
```
</details>

### Kinerja Komputasi (Model Terbaik)

| Parameter | Nilai |
|---|---|
| Jumlah parameter | 2.422.339 (1.690.755 dapat dilatih; 731.584 dibekukan) |
| Ukuran berkas model | 22,52 MB |
| Waktu inferensi | 103,33 ms/citra (rata-rata 5 pengukuran; rentang 97,52–105,97 ms) |
| Lingkungan pengukuran | Google Colab, GPU NVIDIA Tesla T4 |

### Interpretasi Grad-CAM

Visualisasi Grad-CAM pada lima citra uji menunjukkan bahwa peta panas model konsisten terpusat pada buah kopi itu sendiri, bukan pada latar dedaunan atau tanah di sekitarnya — termasuk pada citra dengan latar dedaunan hijau rimbun yang warnanya menyerupai buah kelas mentah. Ini mengindikasikan model membedakan buah dari daun berdasarkan gabungan warna, bentuk, dan tekstur, bukan warna semata.

---

## Isi Notebook (`PELATIHAN_ULANG_Skenario_1-4.ipynb`)

Notebook berjalan berurutan dari SEL 1 hingga SEL 33, dapat dijalankan penuh (*Run all*) di Google Colab dengan Google Drive terhubung.

| Sel | Isi |
|---|---|
| 1–3 | Persiapan lingkungan, koneksi Google Drive, verifikasi jumlah citra per kelas |
| 4–6 | Pemuatan citra ke array NumPy, pembagian data (70:15:15), normalisasi |
| 7 | Augmentasi data latih beserta contoh visualnya |
| 8 | Arsitektur model MobileNetV2 + lapisan klasifikasi |
| 9–12 | Pelatihan, evaluasi, confusion matrix, dan learning curve — **Skenario 1** |
| 13–16 | Idem — **Skenario 2** (augmentasi) |
| 17–21 | Pembukaan 30 lapisan terakhir, pelatihan, evaluasi — **Skenario 3** (fine-tuning) |
| 22–26 | Arsitektur CNN sederhana, pelatihan, evaluasi — **Skenario 4** |
| 27 | Tabel perbandingan keempat skenario |
| 28–29 | Analisis kesalahan klasifikasi pada model terbaik, contoh citra salah prediksi |
| 30–31 | Fungsi dan visualisasi Grad-CAM |
| 32 | Pengukuran waktu inferensi dan ukuran model |
| 33 | Diagram batang ringkasan akhir |

Seluruh keluaran (log pelatihan, grafik, confusion matrix, visualisasi Grad-CAM) sudah tersimpan di dalam notebook ini dan langsung tampil saat dibuka di GitHub, tanpa perlu menjalankan ulang.

---

## Catatan Metodologis

Model Skenario 2–4 pada awalnya melatih setiap epoch genap hanya dengan 1 dari 20 batch (akibat penulisan `steps_per_epoch = 630 // 32 = 19` yang membulatkan ke bawah, padahal generator menyediakan 20 batch mencakup seluruh 630 citra). Setelah ditemukan bahwa hal ini menyebabkan sebagian besar epoch hanya melihat sebagian kecil data latih, kode diperbaiki menjadi `steps_per_epoch = len(train_generator)` agar jumlah batch selalu mengikuti panjang generator secara otomatis, dan seluruh model dilatih ulang dari awal dalam satu sesi yang konsisten. Seluruh angka pada notebook dan README ini adalah hasil setelah perbaikan tersebut.

---

## Cara Menjalankan

1. Buka notebook di [Google Colab](https://colab.research.google.com/).
2. Pastikan runtime menggunakan GPU (*Runtime → Change runtime type → GPU*).
3. Siapkan dataset pada Google Drive sesuai struktur folder pada bagian **Dataset** di atas.
4. Jalankan seluruh sel secara berurutan (*Runtime → Run all*).

## Lisensi Dataset

Dataset bersumber dari Kaggle (Haris Yunanda) di bawah lisensi Open Database License (ODbL).
