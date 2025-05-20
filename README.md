# Laporan Proyek Machine Learning

## Domain Proyek: Heart Diseases Analytics

Penyakit jantung merupakan penyebab utama kematian di dunia dan sering tidak terdeteksi hingga gejala serius muncul atau terjadi serangan jantung mendadak tanpa peringatan. Kondisi ini disebut sebagai "penyakit tersembunyi" karena gejala awal seperti kelelahan, nyeri dada ringan, atau sesak napas sering disalahartikan sebagai masalah ringan, sehingga banyak penderita tidak memperoleh diagnosis dan pengobatan tepat waktu. Kurangnya kesadaran ini meningkatkan risiko komplikasi serius yang sebenarnya dapat dicegah dengan deteksi dini dan penanganan yang tepat.

Penyakit jantung dikenal sebagai salah satu penyebab utama kematian di dunia, termasuk di Indonesia. Berdasarkan data dari World Health Organization (WHO) tahun 2021, tercatat sekitar 17,8 juta kematian setiap tahunnya disebabkan oleh penyakit jantung, atau setara dengan satu dari tiga kematian secara global. Jumlah kasus penyakit jantung terbaru mencapai 21,2 juta, dengan prevalensi lebih tinggi pada laki-laki dibandingkan perempuan.

Penggunaan teknologi seperti machine learning dan analisis data kesehatan berperan penting dalam deteksi dini penyakit jantung. Dengan menganalisis data klinis dan gaya hidup, algoritma dapat mengenali pola dan faktor risiko secara akurat. Model prediktif ini membantu tenaga medis membuat keputusan lebih cepat dan tepat, serta memungkinkan pemantauan kesehatan secara berkala untuk mencegah kondisi memburuk. Pendekatan ini mendukung sistem kesehatan yang lebih preventif dan berpotensi menurunkan angka kematian akibat penyakit jantung.

**Referensi:**
- [World Heart Federation. (2024). Heart Failure](https://world-heart-federation.org/what-we-do/heart-failure/)
- [Kementerian Kesehatan RI. (2018). Laporan Nasional Riskesdas 2018](https://repository.badankebijakan.kemkes.go.id/id/eprint/3514/1/Laporan%20Riskesdas%202018%20Nasional.pdf)

## Business Understanding

### Problem Statements

1. Banyak pasien yang berisiko mengalami penyakit jantung tidak terdeteksi secara dini, sehingga penanganan medis menjadi terlambat dan meningkatkan risiko kematian.
2. Faktor risiko penyakit jantung yang kompleks dan beragam membuat proses identifikasi pasien berisiko tinggi menjadi sulit tanpa bantuan teknologi.
3. Minimnya metode atau alat yang efektif untuk melakukan skrining dan deteksi dini risiko penyakit jantung di masyarakat luas sehingga pencegahan belum optimal.

### Goals

1. Mengembangkan model prediktif untuk mengidentifikasi risiko penyakit jantung berdasarkan data klinis dan demografis pasien.
2. Menentukan faktor-faktor klinis dan gaya hidup yang paling berpengaruh dalam meningkatkan risiko penyakit jantung.
3. Memberikan rekomendasi berbasis data kepada tenaga medis untuk mendukung upaya pencegahan dan deteksi dini penyakit jantung.

### Solution Statements

1. Melakukan analisis data eksploratif (EDA) untuk mengidentifikasi pola risiko dan keterkaitan antara variabel klinis seperti tekanan darah, kadar kolesterol, dan jenis nyeri dada dengan kejadian penyakit jantung.
2. Mengimplementasikan serta membandingkan berbagai algoritma klasifikasi, termasuk `Logistic Regression`, `Random Forest`, dan `Gradient Boosting`, untuk mengembangkan model prediksi dengan tingkat akurasi terbaik.
3. Melakukan evaluasi performa setiap model menggunakan metrik F1-Score untuk memastikan keseimbangan antara precision dan recall, serta menggunakan Area Under ROC Curve (AUC-ROC) untuk menilai kemampuan model dalam membedakan antara kasus positif dan negatif secara keseluruhan.

## Data Understanding

Dataset ini berisi informasi klinis dan demografis pasien yang digunakan untuk mendeteksi kemungkinan penyakit jantung. Berikut adalah penjelasan detailnya:

- Jumlah data: 918 baris
- Jumlah fitur: 12 kolom (termasuk target)
- Tipe data: Gabungan numerik (int64, float64) dan kategorikal (object)
- Sumber data: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

### Variabel-variabel pada Student Depression Dataset adalah sebagai berikut:

| No. | Nama Kolom       | Deskripsi                                                        |
| --- | ---------------- | ---------------------------------------------------------------- |
| 1   | `Age`            | Usia pasien dalam tahun                                          |
| 2   | `Sex`            | Jenis kelamin pasien (`M` = Laki-laki, `F` = Perempuan)          |
| 3   | `ChestPainType`  | Tipe nyeri dada pasien (`TA`, `ATA`, `NAP`, `ASY`)               |
| 4   | `RestingBP`      | Tekanan darah saat istirahat (mm Hg)                             |
| 5   | `Cholesterol`    | Kadar kolesterol dalam darah (mg/dl)                             |
| 6   | `FastingBS`      | Gula darah puasa > 120 mg/dl (`1` = Ya, `0` = Tidak)             |
| 7   | `RestingECG`     | Hasil elektrokardiogram saat istirahat (`Normal`, `ST`, `LVH`)   |
| 8   | `MaxHR`          | Detak jantung maksimum yang dicapai selama tes                   |
| 9   | `ExerciseAngina` | Apakah mengalami angina saat olahraga (`Y` = Ya, `N` = Tidak)    |
| 10  | `Oldpeak`        | Depresi segmen ST dibandingkan dengan kondisi istirahat          |
| 11  | `ST_Slope`       | Kemiringan segmen ST saat puncak olahraga (`Up`, `Flat`, `Down`) |
| 12  | `HeartDisease`   | Variabel target: status penyakit jantung (`1` = Ya, `0` = Tidak) |


Dari hasil analisis awal deskripsi data, terlihat bahwa data siap untuk tahap selanjutnya.

### Distribusi Data Target (Depression)

<br>
<image src='image/distribusi_target.png' width= 500/>
<br>

Distribusi pada data target (Depression) sedikit **imbalance** (tidak seimbang), yang mungkin dapat menyebabkan model cenderung memprediksi kelas mayoritas. Oleh karena itu, untuk mengantisipasi permasalahan ini, akan dilakukan percobaan menggunakan teknik **oversampling** pada **data train** (*setelah proses pembagian data menjadi data latih dan data uji*).

#### Analisis Univariat

**1. Distribusi Data Kategorikal**
<br>
<image src='image/distribusi_data_kategorik.png' width= 500/>
<br>

Berdasarkan distribusi nilai pada kolom-kolom kategorikal, ditemukan bahwa kolom `City` dan `Profession` memiliki beberapa kategori dengan jumlah data yang sangat sedikit (dominan pada satu kategori saja). Selain itu, kolom City juga memiliki terlalu banyak kategori, yang ***dapat menyebabkan curse of dimensionality***. Oleh karena itu, kedua kolom tersebut akan dihapus dari dataset.

Selain itu, pada kolom `Sleep Duration`, `Dietary Habits`, dan `Degree`, terdapat kategori bernilai "Others" yang tidak merepresentasikan informasi yang jelas serta jumlahnya sangat sedikit. Maka dari itu, baris data yang memiliki nilai "Others" pada fitur-fitur tersebut akan dihapus dari dataset.

**2. Distribusi Data Numerik**
<br>
<image src='image/distribusi_data_numerik.png' width= 500/>
<image src='image/boxplot_data_numerik.png' width= 500/>
<br> 
Kolom `Work Pressure` dan `Job Satisfaction` juga menunjukkan dominasi pada satu nilai tertentu, sehingga tidak memberikan variasi yang signifikan untuk analisis. Oleh karena itu, kedua kolom tersebut akan dihapus dari dataset.

Sementara itu, kolom `Age` dan `CGPA` teridentifikasi **memiliki nilai outlier** yang dapat memengaruhi hasil analisis. Outlier pada kedua kolom tersebut akan dihapus pada tahap praproses selanjutnya.

#### Analisis Multivariat

**1. Distribusi Jenis Kelamin dan Pengaruhnya terhadap Status Depresi**
<br>
<image src='image/gender.png' width= 500/>
<br>
Baik pria maupun wanita memiliki proporsi yang hampir sama dalam hal mengalami depresi, dengan sekitar 58% dari masing-masing gender tercatat mengalami depresi. Persentase pria yang mengalami depresi sedikit lebih tinggi (58,62%) dibanding wanita (58,47%). Ini mengindikasikan bahwa dalam data ini, status depresi tidak terlalu dipengaruhi oleh perbedaan gender.

**2. Pengaruh Pemikiran Bunuh Diri Terhadap Status Depresi**
<br>
<image src='image/pemikiran_bunuh_diri.png' width= 500/>
<br>
Seperti yang dapat diduga, data menunjukkan bahwa responden yang memiliki pemikiran untuk melakukan bunuh diri memiliki kemungkinan jauh lebih besar untuk mengalami depresi.

**3. Pengaruh Tekanan Akademik Terhadap Status Depresi**
<br>
<image src='image/tekanan_akademik.png' width= 500/>
<br>
Tekanan akademik yang tinggi dapat menjadi salah satu faktor yang meningkatkan risiko seseorang mengalami depresi. Semakin besar beban dan stres yang dirasakan, semakin tinggi pula kemungkinan individu mengalami gangguan kesehatan mental seperti depresi.

**4. Umur Pelajar Dengan Kemungkinan Status Depresi**
<br>
<image src='image/umur.png' width= 500/>
<br> 
Grafik menunjukkan bahwa responden dengan status depresi cenderung berusia lebih muda, dengan rata-rata usia 24 tahun, dibandingkan yang tidak depresi dengan rata-rata 27 tahun. Hal ini mengindikasikan bahwa depresi lebih banyak dialami oleh kelompok usia muda.

**5. Jam Belajar/Kerja dan Status Depresi**
<br>
<image src='image/waktu_belajar.png' width= 500/>
<br> 
Responden dengan depresi memiliki rata-rata jam kerja/belajar lebih tinggi (7,81 jam) dibandingkan yang tidak depresi (6,24 jam). Ini mengindikasikan bahwa semakin banyak jam kerja/belajar, potensi mengalami depresi cenderung meningkat.

## Data Preparation

Pada tahap ini dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Beberapa tahap persiapan data yang dilakukan adalah:

### 1. Menghapus Kolom & Kategori Values yang Tidak Penting

Pertama akan dilakukakn penghapusan pada kolom - kolom yang tidak akan digunakan lebih lanjut seperti kolom `Id`, `City`, `Profession`, `Job Satisfaction`, `Work Pressure` dan juga mengubah kategori nilai pada kolom `Financial Stress` yang bertipe data **object** menjadi **float**. dan tidak lupa untuk menghapus baris data yang memiliki nilai "Others" pada kolom `Sleep Duration`, `Dietary Habits`, dan `Degree` karana nilai "Others" tidak merepresentasikan informasi yang jelas.
<br>
<image src='image/menghapus_kolom_id.png' width= 500/>
<image src='image/mengubah_kategori_nilai.png' width= 500/>
<br> 

### 2. Menangani Missing Values

Pada dataset terdapat missing value pada kolom `Financial Stress` sebanyak 3 data. Dikarenakan jumlahnya yang sedikit dan untuk menjaga keaslian data, maka diputuskan untuk menghapus baris-baris tersebut dari dataset.
<img src='image/null_value.png' align="center"><br>

### 3. Menghapus Outlier Values

Untuk menangani outlier, dilakukan penghapusan outlier pada kolom `Age` dan `CGPA` menggunakan metode IQR (Interquartile Range). Metode ini digunakan untuk menghilangkan nilai-nilai yang berada di luar batas bawah dan batas atas yang ditentukan, sehingga data menjadi lebih bersih dan representatif.
<img src='image/outlier.png' align="center"><br>

### 4. Encoding Fitur Kategori

Pada bagian ini, dilakukan transformasi data kategori (yang berbentuk teks atau label) menjadi format numerik agar dapat diproses oleh algoritma machine learning. Encoding fitur kategorikal dilakukan dalam 2 bagian:

1. **Label Encoding**: mengonversi nilai kategori menjadi angka integer (`0` dan `1`) untuk variabel biner seperti:
   - `Gender`
   - `Have you ever had suicidal thoughts ?`
   - `Family History of Mental Illness`

2. **One Hot Encoding**: mengubah setiap kategori menjadi kolom biner terpisah untuk data tidak terurut seperti:
   - `Sleep Duration`
   - `Dietary Habits`
   - `Degree`

<img src="image/encoding.png" align="center"><br>

### 5. Train-Test-Split

Data dibagi dengan proporsi 80:20, dimana 80% digunakan untuk training model dan 20% digunakan untuk testing model, untuk memastikan evaluasi yang objektif terhadap performa model.
<img src="image/spliting_data.png" align="center"><br>

### 6. Transformasi Values

Dilakukan scaling value dengan MinMaxScaler untuk menyamaratakan skala dari setiap fitur, sehingga tidak ada fitur yang mendominasi karena memiliki skala nilai yang lebih besar.
<img src="image/transformation_value.png" align="center"><br>

### 7. Menangani Data Imbalance

SMOTE (Synthetic Minority Over-sampling Technique) digunakan untuk mengatasi ketidakseimbangan kelas pada data latih. Pengujian juga dilakukan pada data tanpa SMOTE untuk membandingkan akurasi dan menilai efektivitas metode tersebut.
<img src="image/imbalance_data.png" align="center"><br>

## Modeling

Pada project kali ini akan dilakukan percobaan terhadap beberapa algoritma machine learning yaitu:

### 1. Logistic Regression
Logistic Regression adalah algoritma machine learning yang digunakan untuk klasifikasi, terutama klasifikasi biner. Algoritma ini memodelkan probabilitas suatu data masuk ke kelas tertentu menggunakan fungsi logistik (sigmoid). Logistic Regression mencoba menemukan garis pemisah (decision boundary) linear antara kelas.
**Kelebihan:**
* Sederhana dan cepat untuk dilatih.
* Mudah diinterpretasikan melalui nilai koefisien fitur.
* Cocok untuk kasus klasifikasi biner.

**Kekurangan:**
* Sensitif terhadap multikolinearitas.
* Kurang efektif jika banyak outlier.

### 2. Decision Tree
Decision Tree adalah algoritma klasifikasi yang bekerja dengan membagi dataset berdasarkan fitur menjadi cabang-cabang seperti struktur pohon. Setiap node dalam pohon mewakili fitur, dan setiap daun mewakili kelas. Algoritma ini menggunakan pemisahan berdasarkan kriteria tertentu (seperti Gini atau Entropy) untuk membuat keputusan.

**Kelebihan:**
* Bisa menangani data numerik dan kategorikal.
* Tidak memerlukan normalisasi atau scaling data.
* Dapat menangkap hubungan non-linear antara fitur.

**Kekurangan:**
* Rentan terhadap overfitting, terutama pada data kompleks.
* Sensitif terhadap perubahan kecil pada data.
* Bisa membuat model yang terlalu dalam atau kompleks.

### 3. XGBoost
XGBoost adalah algoritma boosting yang sangat efisien dan akurat. Algoritma ini bekerja dengan membangun banyak pohon keputusan secara bertahap, di mana setiap pohon mencoba memperbaiki kesalahan dari pohon sebelumnya. XGBoost dilengkapi dengan teknik regularisasi dan optimisasi untuk mencegah overfitting dan meningkatkan performa.

**Kelebihan:**
* Mampu menangani missing value secara otomatis.
* Dilengkapi dengan regularisasi untuk menghindari overfitting.
* Cepat dalam pelatihan dan prediksi.

**Kekurangan:**
* Lebih kompleks dan sulit untuk dipahami secara menyeluruh.
* Membutuhkan waktu tuning hyperparameter yang tidak sedikit.
* Mengonsumsi memori dan waktu lebih banyak dibanding model sederhana.

## Evaluation

Untuk mengevaluasi kinerja model dalam mendeteksi risiko depresi pada mahasiswa, digunakan metrik **F1 Score**. Pemilihan metrik ini didasarkan pada karakteristik masalah yang memiliki distribusi kelas tidak seimbang dan dampak serius apabila terjadi kesalahan klasifikasi.

### F1 Score

**F1 Score** merupakan metrik yang menggabungkan **Precision** dan **Recall** dalam satu nilai harmonis, dengan rumus:

<img src="image/f1score_image.png" align="center"><br>

di mana:
- **Precision**: Persentase prediksi positif yang benar-benar positif.  
<img src="image/precision_formulas.png" align="center"><br>
- **Recall**: Persentase kasus positif yang berhasil diprediksi sebagai positif.  
<img src="image/recall_formulas.png" align="center"><br>

Penggunaan F1 Score sangat sesuai untuk situasi di mana keseimbangan antara **False Positive** dan **False Negative** penting untuk dipertahankan, seperti dalam kasus deteksi risiko depresi pada mahasiswa. Berdasarkan hasil evaluasi, model **Logistic Regression** memperoleh nilai F1 Score tertinggi sebesar **0,87**, menunjukkan bahwa model ini mampu menjaga keseimbangan terbaik antara ketepatan dalam mendeteksi depresi dan kepekaan dalam menjangkau kasus positif dibandingkan model lainnya.

<img src="image/perbandingan_score.png" align="center"><br>

## Referensi
1. World Health Organization. (2020). Depression. Retrieved from: https://www.who.int/news-room/fact-sheets/detail/depression
2. American College Health Association. (2021). ACHA-National College Health Assessment III: Undergraduate Student Reference Group Executive Summary Spring 2021.Retrieved from: https://www.sjsu.edu/wellness/docs/ncha-spring-2021-executive-summary.pdf
3. Kementerian Kesehatan RI. (2018). Laporan Nasional Riskesdas 2018. Badan Penelitian dan Pengembangan Kesehatan, Kemenkes RI. Retrieved from: https://repository.badankebijakan.kemkes.go.id/id/eprint/3514/1/Laporan%20Riskesdas%202018%20Nasional.pdf
4. https://medium.com/@andimrinaldisaputraa/memahami-dan-menerapkan-matriks-evaluasi-roc-auc-dalam-machine-learning-4468e5fcb9a