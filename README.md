# Wood Defect Classifier

Aplikasi klasifikasi cacat permukaan kayu menggunakan metode **Gray Level Co-occurrence Matrix (GLCM)** sebagai ekstraksi fitur dan **K-Nearest Neighbor (KNN)** sebagai algoritma klasifikasi.

## Kelas yang Digunakan

* Normal
* Knot
* Dark Knot

## Struktur Proyek

```text
wood-defect-classifier/
├── dataset/
│   ├── normal/
│   ├── knot/
│   └── dark_knot/
├── model/
│   └── knn_model.pkl
├── app.py
├── train.py
└── requirements.txt
```

## Membuat Virtual Environment

Jalankan seluruh perintah berikut dari folder root proyek (folder yang berisi `app.py`, `train.py`, dan folder `dataset`).

Jika menggunakan Visual Studio Code:

1. Buka folder proyek.
2. Pilih **Terminal → New Terminal**.
3. Pastikan terminal berada pada folder root proyek.
4. Buat virtual environment:

```powershell
python -m venv .venv
```

5. Aktifkan virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jika berhasil, prompt terminal akan berubah menjadi:

```powershell
(.venv) PS ...\wood-defect-classifier>
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

Jika file `requirements.txt` belum tersedia:

```powershell
pip install numpy pandas scikit-image scikit-learn streamlit pillow
```

## Training Model

Jalankan proses training untuk membuat model KNN:

```powershell
python train.py
```

Model yang telah dilatih akan disimpan pada:

```text
model/knn_model.pkl
```

## Menjalankan Aplikasi

Setelah model tersedia, jalankan aplikasi Streamlit:

```powershell
streamlit run app.py
```

atau

```powershell
python -m streamlit run app.py
```

Aplikasi akan terbuka pada browser melalui alamat:

```text
http://localhost:8501
```

Jika sudah selesai menggunakan, bisa tekan Ctrl + C ke terminal yang dijalankan

## Cara Penggunaan

1. Jalankan aplikasi Streamlit.
2. Upload gambar kayu (.jpg, .jpeg, .png, atau .bmp).
3. Sistem akan mengekstraksi fitur GLCM:

   * Contrast
   * Energy
   * Homogeneity
   * Correlation
4. Sistem akan menampilkan hasil klasifikasi:

   * Normal
   * Knot
   * Dark Knot

## Metode

* Ekstraksi Fitur: Gray Level Co-occurrence Matrix (GLCM)
* Fitur: Contrast, Energy, Homogeneity, Correlation
* Algoritma Klasifikasi: K-Nearest Neighbor (KNN)
* Bahasa Pemrograman: Python
* Framework Antarmuka: Streamlit

```
```
