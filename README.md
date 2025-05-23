# Laporan Proyek Machine Learning

## Domain Proyek: Heart Diseases Analytics

Penyakit jantung merupakan penyebab utama kematian di dunia dan sering tidak terdeteksi hingga gejala serius muncul atau terjadi serangan jantung mendadak tanpa peringatan. Kondisi ini disebut sebagai "penyakit tersembunyi" karena gejala awal seperti kelelahan, nyeri dada ringan, atau sesak napas sering disalahartikan sebagai masalah ringan, sehingga banyak penderita tidak memperoleh diagnosis dan pengobatan tepat waktu. Kurangnya kesadaran ini meningkatkan risiko komplikasi serius yang sebenarnya dapat dicegah dengan deteksi dini dan penanganan yang tepat.

Penyakit jantung dikenal sebagai salah satu penyebab utama kematian di dunia, termasuk di Indonesia. Berdasarkan data dari World Health Organization (WHO) tahun 2021, tercatat sekitar 17,8 juta kematian setiap tahunnya disebabkan oleh penyakit jantung, atau setara dengan satu dari tiga kematian secara global. Jumlah kasus penyakit jantung terbaru mencapai 21,2 juta, dengan prevalensi lebih tinggi pada laki-laki dibandingkan perempuan.

Penggunaan teknologi seperti machine learning dan analisis data kesehatan berperan penting dalam deteksi dini penyakit jantung. Dengan menganalisis data klinis dan gaya hidup, algoritma dapat mengenali pola dan faktor risiko secara akurat. Model prediktif ini membantu tenaga medis membuat keputusan lebih cepat dan tepat, serta memungkinkan pemantauan kesehatan secara berkala untuk mencegah kondisi memburuk. Pendekatan ini mendukung sistem kesehatan yang lebih preventif dan berpotensi menurunkan angka kematian akibat penyakit jantung.

**Referensi:**
- [Columbia Asia. (2024). Penyakit Jantung: Ancaman Tersembunyi](https://columbiaasia.co.id/artikel/kesehatan/penyakit-jantung-ancaman-tersembunyi/)
- [Kementerian Kesehatan Republik Indonesia. (2021). Penyakit Jantung, Penyebab Kematian Nomor Satu di Dunia](https://lms.kemkes.go.id/courses/35bff824-437e-4557-b37a-94b128c43333)

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

## Exploratory Data Analysis

#### Analisis Univariat

**1. Distribusi Data Target (HeartDisease)**

<br>
<image src='images/distribusi_datatarget.png' width= 500/>
<br>

Distribusi data pada label `HeartDisease` adalah cukup imbalance (55.3% vs 44.7%). Hal ini harus diperhatikan saat pelatihan model, agar tidak menghasilkan prediksi yang berat sebelah. Metrik evaluasi yang tepat dan teknik penyeimbangan data seperti oversampling pada data train (saat setelah membagi data) bisa membantu mengatasi masalah ini.


**2. Distribusi Data Kategorikal**
<br>
<image src='images/barchart_cat.png' width= 500/>
<br>

Distribusi data kategorikal menunjukkan ketimpangan pada beberapa fitur, seperti `ChestPainType` dan `ST_Slope`. Ketimpangan ini dapat memengaruhi kinerja model, sehingga perlu penanganan khusus. Salah satu cara efektif adalah menggunakan **One-Hot Encoding**, yang mengubah setiap kategori menjadi kolom biner tanpa memberi makna urutan. Pendekatan ini membantu model memahami setiap kategori secara adil dan mencegah bias terhadap kategori mayoritas.

**3. Distribusi Data Numerik**
<br>
<image src='images/Histogram_num.png' width= 500/>
<image src='images/Boxplot_num.png' width= 500/>
<br> 
Dari gambar histogram menggambarkan sebaran nilai dan frekuensi kemunculannya. Sebagian besar distribusi menunjukkan bentuk mendekati normal (seperti `Age` dan `MaxHR`), namun beberapa memiliki sebaran yang condong atau outlier ekstrem, seperti `Cholesterol` dan `Oldpeak`

Boxplot menunjukkan bahwa sebagian besar fitur memiliki outlier, terutama pada RestingBP, Cholesterol, dan Oldpeak, sedangkan Age dan MaxHR memiliki sebaran yang relatif normal. Fitur FastingBS didominasi oleh nilai 0 dan minim variasi, sehingga dianggap tidak informatif dan akhirnya dihapus karena berpotensi tidak berkontribusi signifikan terhadap performa model.

**4. Matriks korelasi kolom numerik**
<br>
<image src='images/Correlation_matrix.png' width= 500/>
<br> 
Berdasarkan korelasi antar fitur numerik, tidak ditemukan fitur yang redundant dalam dataset ini. Nilai korelasi antar fitur mayoritas rendah, di bawah 0.3, sehingga setiap fitur memberikan informasi yang unik dan layak dipertahankan untuk analisis atau pemodelan.

#### Analisis Multivariat
## Kategorikal
**1. Sex vs HeartDisease**
<br>
<image src='images/Sex vs HeartDisease.png' width= 500/>
<br>
- Pria (M) cenderung memiliki jumlah kasus penyakit jantung (HeartDisease = Yes) sebesar 90.2% yang jauh lebih banyak dibandingkan wanita (F) yang sebesar 9.8%.
- Wanita memiliki jumlah kasus negatif sebesar 34.9% (tidak ada penyakit jantung) yang lebih tinggi dibandingkan kasus positif yang sebesar 9.8%.

Indikasi: Jenis kelamin pria memiliki risiko lebih tinggi terkena penyakit jantung.

**2. ChestPainType vs HeartDisease**
<br>
<image src='images/ChestPainType vs HeartDisease.png' width= 500/>
<br>
- Tipe nyeri dada ASY (Asymptomatic) memiliki jumlah kasus penyakit jantung yang sangat tinggi dibandingkan tipe lainnya.
- Tipe ATA (Atypical angina), NAP (Non-anginal pain), dan TA (Typical angina) cenderung lebih banyak ditemukan pada pasien tanpa penyakit jantung.

Indikasi: Nyeri dada tipe Asymptomatic berpotensi menjadi indikator kuat penyakit jantung.

**3. RestingECG vs HeartDisease**
<br>
<image src='images/RestingECG vs HeartDisease.png' width= 500/>
<br>
- Pada kategori Normal, distribusi pasien dengan dan tanpa penyakit jantung tampak seimbang.
- Kategori ST dan LVH lebih banyak ditemukan pada pasien dengan penyakit jantung.

Indikasi: Abnormalitas pada Resting ECG (ST, LVH) bisa mengindikasikan adanya penyakit jantung.

**4. ExerciseAngina vs HeartDisease**
<br>
<image src='images/ExerciseAngina vs HeartDisease.png' width= 500/>
<br> 
- Pasien dengan ExerciseAngina = Y (ya) cenderung lebih banyak yang positif penyakit jantung.
- Sebaliknya, pasien tanpa exercise angina (N) mayoritas tidak memiliki penyakit jantung.

Indikasi: Exercise angina mungkin menjadi tanda yang cukup kuat adanya penyakit jantung.

**5. ST_Slope vs HeartDisease**
<br>
<image src='images/ST_Slope vs HeartDisease.png' width= 500/>
<br> 
- Kategori Flat pada ST slope tampak terkait dengan jumlah kasus penyakit jantung yang tinggi.
- Kategori Up lebih sering ditemukan pada pasien tanpa penyakit jantung.
- Kategori Down juga lebih banyak pada pasien dengan penyakit jantung, meskipun jumlahnya lebih kecil dari Flat.

Indikasi: ST slope Flat dan Down bisa menjadi indikator risiko penyakit jantung yang lebih tinggi dibandingkan slope Up.

## Numerikal
**1. Age vs HeartDisease**
<br>
<image src='images/Distribution of Age by HeartDisease.png' width= 500/>
<br> 
- Distribusi: Terlihat pergeseran ke kanan pada grup penderita HeartDisease = Yes, artinya penderita cenderung lebih tua dibandingkan yang tidak.
- Rata-rata:
  - All: 53.51
  - No: 50.55
  - Yes: 54.90

Pengaruh: Umur yang lebih tua meningkatkan kemungkinan terkena penyakit jantung, karena risiko penyakit jantung memang meningkat seiring bertambahnya usia.

**2. RestingBP vs HeartDisease**
<br>
<image src='images/Distribution of RestingBP by HeartDisease.png' width= 500/>
<br> 
- Distribusi: Kedua kelompok (HeartDisease = No dan Yes) memiliki distribusi yang mirip, namun terdapat sedikit pergeseran ke kanan pada kelompok HeartDisease = Yes, yang menunjukkan tekanan darah lebih tinggi.
- Rata-rata:
  - All: 132.40
  - No: 130.18
  - Yes: 134.19

Pengaruh: enderita penyakit jantung cenderung memiliki tekanan darah istirahat yang sedikit lebih tinggi. Meskipun perbedaannya tidak terlalu besar, tekanan darah yang meningkat dapat menjadi faktor risiko tambahan terhadap penyakit jantung, terutama jika dikombinasikan dengan faktor risiko lainnya.

**3. Cholesterol vs HeartDisease**
<br>
<image src='images/Distribution of Cholesterol by HeartDisease.png' width= 500/>
<br> 
- Distribusi: Terlihat bahwa kelompok HeartDisease = No memiliki distribusi kolesterol yang lebih tinggi dibandingkan dengan kelompok HeartDisease = Yes, yang cukup mengejutkan karena bertentangan dengan asumsi umum.
- Rata-rata:
  - All: 198.80
  - No: 227.12
  - Yes: 175.94

Pengaruh: Secara mengejutkan, kolesterol rata-rata pada penderita penyakit jantung justru lebih rendah. Hal ini bisa disebabkan oleh berbagai faktor, seperti efek pengobatan yang menurunkan kolesterol, gaya hidup setelah diagnosis, atau distribusi data yang tidak merata. Oleh karena itu, kolesterol dalam dataset ini mungkin kurang representatif sebagai indikator tunggal untuk mendeteksi penyakit jantung.

**4. MaxHR vs HeartDisease**
<br>
<image src='images/Distribution of MaxHR by HeartDisease.png' width= 500/>
<br> 
- Distribusi: Distribusi menunjukkan bahwa kelompok HeartDisease = No memiliki detak jantung maksimum yang lebih tinggi, sedangkan kelompok HeartDisease = Yes cenderung memiliki MaxHR yang lebih rendah.
- Rata-rata:
  - All: 136.81
  - No: 148.15
  - Yes: 127.66

Pengaruh: Penderita penyakit jantung cenderung tidak mampu mencapai detak jantung maksimum yang tinggi saat beraktivitas fisik, yang bisa menjadi indikasi keterbatasan fungsi jantung. MaxHR yang lebih rendah merupakan sinyal penting adanya potensi gangguan jantung, sehingga fitur ini cukup signifikan dalam membedakan antara penderita dan non-penderita.

**5. Oldpeak vs HeartDisease**
<br>
<image src='images/Distribution of Oldpeak by HeartDisease.png' width= 500/>
<br> 
- Distribusi: Sangat berbeda antar grup. Grup HeartDisease = Yes punya distribusi lebih menyebar ke kanan (nilai oldpeak lebih tinggi).
- Rata-rata:
  - All: 0.89
  - No: 0.41
  - Yes: 1.27

Pengaruh: Oldpeak tinggi mengindikasikan abnormalitas EKG akibat iskemia (kurangnya aliran darah ke jantung), sehingga sangat erat kaitannya dengan penyakit jantung. Ini adalah salah satu fitur yang paling membedakan kedua grup.

## Data Preparation

adalah proses mengubah dan mengolah data mentah agar siap digunakan dalam tahap pemodelan. Proses ini meliputi pembersihan, transformasi, dan penyesuaian format data agar sesuai dengan kebutuhan algoritma yang akan diterapkan. Dengan persiapan data yang tepat, model dapat belajar secara efektif dan menghasilkan prediksi yang akurat. Berikut adalah beberapa tahapannya:

### 1. Menghapus Kolom yang Tidak digunakan

Pertama yang akan dilakukan yaitu menghapus kolom `FastingBS` yang sudah tidak digunakan lagi untuk tahap yang lebih lanjut.
<br>
<image src='images/drop_colomn.png' width= 500/>
<br> 

### 2. Menangani Missing Values

Pada dataset terlihat tidak terdapat missing value. 
<br>
<img src='images/handle_missingvalues.png' align="center"><br>
<br> 

### 3. Menangani Data Duplikat

Pada dataset terlihat tidak terdapat data duplikat. 

![alternative text](images/handle_duplicated.png)

### 4. Menghapus Outlier Values

Untuk menangani outlier, dilakukan penghapusan outlier pada kolom `RestingBP`, `Cholesterol`, `MaxHR` dan `Oldpeak` menggunakan metode IQR (Interquartile Range). Metode ini diterapkan untuk menghapus data yang berada di luar rentang batas bawah dan batas atas yang telah ditetapkan, sehingga menghasilkan data yang lebih rapi dan menggambarkan kondisi sebenarnya dengan lebih baik.
<img src='images/handle_outlier.png' align="center"><br>

### 5. Encoding Fitur Kategori

Di tahap ini, data kategori berupa teks atau label diubah menjadi format angka supaya bisa diproses oleh algoritma machine learning. Proses encoding pada fitur kategorikal dilakukan dalam dua tahap:

1. **Label Encoding**: mengubah nilai kategori menjadi bilangan bulat (seperti `0` dan `1`) pada variabel yang hanya memiliki dua kelas, berikut adalah kolomnya:
   - `Sex`
   - `ExerciseAngina`

2. One Hot Encoding: mengonversi setiap kategori menjadi kolom-kolom biner yang terpisah, digunakan untuk data kategori yang tidak memiliki urutan, berikut adalah kolomnya:
   - `ChestPainType`
   - `RestingECG`
   - `ST_Slope`

<img src="images/encoding.png" align="center"><br>

### 6. Train-Test-Split

Data dibagi dengan rasio 80:20, di mana 80% dari data digunakan sebagai data pelatihan (training) untuk membangun dan mengoptimalkan model, sedangkan 20% sisanya dialokasikan sebagai data pengujian (testing). Pembagian ini bertujuan untuk memastikan proses evaluasi model berjalan secara objektif dan hasil pengujian mencerminkan kemampuan model dalam menghadapi data baru yang belum pernah dilihat sebelumnya. Dengan demikian, performa model dapat diukur secara akurat dan generalisasi model terhadap data nyata dapat dipantau.
<img src="images/data_split.png" align="center"><br>

### 7. Transformasi Values

Dilakukan proses scaling menggunakan MinMaxScaler untuk menormalkan rentang nilai pada setiap fitur. Dengan metode ini, semua fitur diubah ke dalam skala yang seragam, biasanya antara 0 hingga 1, sehingga tidak ada fitur yang memiliki pengaruh berlebihan akibat perbedaan skala nilai. Hal ini membantu algoritma machine learning bekerja lebih efektif dan menghasilkan model yang lebih stabil.
<img src="images/minmaxscaler.png" align="center"><br>

### 8. Menangani Data Imbalance

Untuk mengatasi ketidakseimbangan kelas pada data latih, digunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) yang menambahkan data sintetis pada kelas yang jumlahnya lebih sedikit. Selain itu, pengujian juga dilakukan pada data tanpa penerapan SMOTE guna membandingkan tingkat akurasi dan menilai sejauh mana metode ini efektif dalam meningkatkan performa model.
<img src="images/smote.png" align="center"><br>

## Modeling

Pada project kali ini akan dilakukan percobaan terhadap beberapa algoritma machine learning yaitu:

### 1. Logistic Regression
Logistic Regression adalah algoritma machine learning yang digunakan untuk tugas klasifikasi, terutama klasifikasi biner. Algoritma ini memodelkan probabilitas suatu data termasuk dalam kelas tertentu dengan menggunakan fungsi logistik (sigmoid). Logistic Regression mencari garis batas keputusan (decision boundary) linier antara kelas-kelas.
**Kelebihan:**
* Sederhana dan cepat dalam pelatihan.
* Mudah diinterpretasikan melalui koefisien fitur.
* Efektif untuk data yang memiliki hubungan linier.

**Kekurangan:**
* Kurang cocok untuk data dengan hubungan non-linier yang kompleks.
* Sensitif terhadap multikolinearitas antar fitur.
* Kinerjanya menurun jika terdapat banyak outlier.

### 2. Random Forest
Random Forest adalah algoritma ensemble learning yang membentuk banyak pohon keputusan (decision tree) dan menggabungkan hasilnya untuk meningkatkan akurasi dan mengurangi overfitting. Setiap pohon dilatih pada subset acak dari data dan fitur, sehingga menciptakan model yang lebih stabil dan tahan terhadap noise.

**Kelebihan:**
* Dapat menangani data numerik maupun kategorikal.
* Kurang rentan terhadap overfitting dibanding single decision tree.
* Memberikan estimasi pentingnya fitur (feature importance).

**Kekurangan:**
* Kurang interpretatif dibanding model yang lebih sederhana.
* Membutuhkan lebih banyak waktu dan sumber daya dibanding algoritma dasar.
* Tidak secepat model linear pada dataset besar.

### 3. Gradient Boosting
Gradient Boosting adalah algoritma ensemble yang membangun model secara bertahap, di mana setiap model baru mencoba memperbaiki kesalahan dari model sebelumnya. Model ini sangat kuat karena mampu menangani data kompleks dan non-linier, serta memiliki performa tinggi pada banyak tugas klasifikasi.

**Kelebihan:**
* Performa tinggi dalam banyak kasus klasifikasi dan regresi.
* Mampu menangkap hubungan non-linier antar fitur.
* Dapat dikombinasikan dengan teknik regularisasi untuk mengurangi overfitting.

**Kekurangan:**
* Proses pelatihan relatif lambat.
* Memerlukan tuning hyperparameter yang cermat untuk hasil optimal.
* Lebih kompleks dan sulit untuk diinterpretasikan dibanding model sederhana.

## Evaluation

Untuk mengevaluasi kinerja model dalam mendeteksi risiko penyakit jantung, digunakan beberapa metrik evaluasi, yaitu recall, F1-score, dan ROC (Receiver Operating Characteristic). Penggunaan metrik ini didasarkan pada karakteristik masalah yang memiliki distribusi kelas tidak seimbang serta potensi dampak serius jika terjadi kesalahan klasifikasi. Recall digunakan untuk menilai kemampuan model dalam mengidentifikasi seluruh kasus positif (individu yang berisiko penyakit jantung), F1-score memberikan keseimbangan antara presisi dan recall, sementara ROC membantu mengevaluasi performa model pada berbagai ambang batas klasifikasi.

### Hasil perbandingan model sebelum dan sesudah dilakukan hyperparameter tuning
| Model                               | Acc Train | Acc Test | Prec Train | Prec Test | Rec Train | Rec Test | F1 Train | F1 Test | ROC Train | ROC Test |
| ----------------------------------- | --------- | -------- | ---------- | --------- | --------- | -------- | -------- | ------- | --------- | -------- |
| Random Forest (Before Tuning)       | 1.000     | 0.929    | 1.000      | 0.910     | 1.000     | 0.959    | 1.000    | 0.934   | 1.000     | 0.927    |
| Random Forest (Tuned)               | 0.902     | 0.908    | 0.888      | 0.877     | 0.892     | 0.959    | 0.890    | 0.916   | 0.901     | 0.905    |
| Gradient Boosting (Before Tuning)   | 0.964     | 0.887    | 0.953      | 0.892     | 0.968     | 0.892    | 0.960    | 0.892   | 0.965     | 0.886    |
| Gradient Boosting (Tuned)           | 0.941     | 0.915    | 0.925      | 0.919     | 0.944     | 0.919    | 0.935    | 0.919   | 0.941     | 0.915    |
| K-Nearest Neighbors (Before Tuning) | 0.875     | 0.894    | 0.844      | 0.873     | 0.884     | 0.932    | 0.863    | 0.902   | 0.876     | 0.892    |
| K-Nearest Neighbors (Tuned)         | 1.000     | 0.908    | 1.000      | 0.886     | 1.000     | 0.946    | 1.000    | 0.915   | 1.000     | 0.906    |

### F1 Score

**F1 Score** adalah metrik yang menggabungkan precision dan recall. F1-score sangat berguna ketika kita menghadapi ketidakseimbangan kelas dalam dataset. Nilai F1-score adalah rata-rata harmonis antara precision dan recall, memberikan keseimbangan antara kedua metrik tersebut. F1-score memberikan keseimbangan antara precision dan recall, yang berguna ketika keduanya sama pentingnya, dengan rumus:

<img src="images/F1-Score.png" align="center"><br>

di mana:
- **Precision**: Precision adalah metrik yang lebih spesifik daripada akurasi untuk mengukur salah-satu label sebagai entitas terpisah. Precision mengukur: dari semua prediksi pada label tertentu (dalam hal ini label positif), berapa persen yang prediksinya benar. Atau dalam bahasa yang lebih teknis, precision adalah rasio antara prediksi positif yang benar (true positive) dengan total prediksi positif (baik yang benar maupun salah).  
<img src="images/Precision.png" align="center"><br>
- **Recall**: Recall mengukur seberapa banyak kasus positif dari semua data yang aktualnya beneran positif. Metrik ini penting ketika kita ingin meminimalkan kesalahan negatif (false negative), seperti pada deteksi penyakit di mana kita tidak ingin ada pasien yang sakit tetapi diklasifikasikan sebagai sehat. Secara teknis dalam confusion matrix, recall adalah rasio antara prediksi positif yang benar dengan total jumlah data yang sebenarnya positif.
<br><img src="images/Recall.png" align="center"><br>

**ROC (Receiver Operating Characteristic)** adalah grafik yang digunakan untuk mengevaluasi kinerja model klasifikasi biner. Grafik ini menggambarkan kemampuan model dalam membedakan antara kelas positif dan negatif pada berbagai threshold (ambang batas) probabilitas.

Penggunaan metrik recall sangat penting untuk memastikan model mampu menangkap sebanyak mungkin kasus positif, sedangkan F1 Score sangat sesuai untuk menjaga keseimbangan antara False Positive dan False Negative, terutama dalam deteksi risiko penyakit jantung. Berdasarkan hasil evaluasi, model Gradient Boosting mencapai recall sebesar 0,92, F1 Score sebesar 0,92, dan ROC AUC sebesar 0,91, yang menunjukkan bahwa model ini memiliki kepekaan tinggi dalam mendeteksi kasus positif sekaligus performa klasifikasi yang kuat dibandingkan model lain.

<img src="images/visualisasi_alltuning.png" align="center"><br>

## Referensi
1. Columbia Asia. (2024). Penyakit Jantung: Ancaman Tersembunyi. Retrieved from: https://columbiaasia.co.id/artikel/kesehatan/penyakit-jantung-ancaman-tersembunyi/
2. Kementerian Kesehatan Republik Indonesia. (2021). Penyakit Jantung, Penyebab Kematian Nomor Satu di Dunia. Retrieved from: https://lms.kemkes.go.id/courses/35bff824-437e-4557-b37a-94b128c43333
3. https://haloryan.com/blog/apa-itu-akurasi-precision-recall-f1-score-rumus-cara-menghitungnya
4. https://www.haloryan.com/blog/roc-auc-pengertian-fungsi-dan-cara-menggunakannya-dalam-machine-learning
