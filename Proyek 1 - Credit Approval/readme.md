# Laporan Proyek Machine Learning - Faisal Mashuri
## Domain Proyek
Kegagalan untuk mengidentifikasi risiko kredit menyebabkan hilangnya pendapatan dan memperluas risiko kredit macet menjadi ancaman bagi profitabilitas. Kesalahan dalam analisis kredit menimbulkan risiko kredit, seperti kehilangan nasabah, ketidakpastian pengembalian pinjaman, bahkan ketidakmampuan nasabah untuk mengembalikan pinjaman (Zurada & Kunene, 2011).


Teknik klasifikasi pada machine learning dapat digunakan untuk menentukan risiko kredit. Misalnya di bank, sifat nasabah yang bisa membayar pinjamannya dapat di prediksi dan model dapat dibuat dengan menggunakan kumpulan data sebelumnya tentang pendanaan nasabah. Setelah itu model dapat digunakan pada yang baru. Pelanggan untuk menentukan kemungkinan membayar kembali kredit mereka. Dalam metode deskriptif, hubungan dapat dicari antara dua kumpulan data, misalnya kebiasaan berbelanja dari dua budaya yang berbeda dapat diselidiki kesamaannya.

Dalam memprediksi resiko pemberian kredit, maka diperlukannya sebuah sistem yang dapat memprediksi hal tersebut seperti machine learning. Teknik machine learning menjadi sangat populer saat ini karena ketersediaannya luas dalam kuantitas data yang besar serta kebutuhan kita untuk dapat mengubah data menjadi suatu pengetahuan. Kebutuhan akan pengetahuan inilah yang akhirnya mendorong industri TI untuk menggunakan machine learning. Data mining adalah proses mencari pola atau informasi. 

## Business Understanding
### Problem Statements
Berdasarkan latar belakang yang sudah dipaparkan sebelumnya, berikut rincian masalah yang dapat diselesaikan dalam proyek ini :
- Bagaimana cara melakukan preprocessing pada data yang tidak seimbang atau imbalance data untuk membuat model machine learning yang baik?
- Bagaimana cara membuat model machine learning untuk menentukan persetujuan pengajuan kredit?

## Goals:
- Melakukan preprocessing pada data yang tidak seimbang atau imlanace data untuk membuat model machine learning yang baik.
- Membuat model machine learning untuk menentukan persetujuan pengajuan kredit

## Solution Statements
- Menggunakan EDA untuk melihat fitur – fitur yang berkorelasi dan berpengaruh terhadap persetujuan pengajuan kredit
- Menggunakan berbagai model machine learning untuk mengetahui kemungkinan terbesar penumpang – penumpang yang selamat lewat data yang diberikan. Model - model yang akan dipakai adalah model - model machine learning klasifikasi. Kemudian kita akan memilih model mana yang memiliki tingkat akurasi paling tinggi, model – model yang akan dibuat adalah:
    * Support Vector Machine
    * XGBClassifier

## Data Understanding
Untuk data set saya menggunakan dataset dari kaggle yang dapat dilihat pada dibawah ini :
| Jenis | Keterangan | 
| ----------- | :---------: | 
| Sumber | https://www.kaggle.com/rikdifos/credit-card-approval-prediction | 
| Kategori | Layak (0) dan Tidak Layak(1)| 
 
Variabel-variabel pada dataset adalah sebagai berikut:
1. ID: id unik pada tiap baris
2. CODE_GENDER: kode jenis kelamin. M adalah laki- laki dan F dan perempuan.
3. FLAG_OWN_CAR: kolom yang menunjukan kepemilikan mobil. Y adalah punya dan N adalah tidak.
4. FLAG_OWN_REALTY: kolom yang menunjukan properti. Y adalah iya dan N adalah tidak.
5. CNT_CHILDREN: Jumlah anak.
6. AMT_INCOME_TOTAL: Jumlah pemasukan.
7. NAME_INCOME_TYPE: jenis pemasukan.
8. NAME_EDUCATION_TYPE: jenis edukasi.
9. NAME_FAMILY_STATUS: jenis status keluarga.
10. DAYS_BIRTH: Hari lahir.
11. DAYS_EMPLOYED: jumlah hari menjadi pegawai.
12. FLAG_MOBILE: kolom yang menunjukan kepemilikan mobile phone. 1 artinya punya and 0 berarti tidak.
13. FLAG_WORK_PHONE: kolom yang menunjukan kepemilikan telpon kantor1 artinya punya and 0 berarti tidak.
14. FLAG_PHONE: kolom yang menunjukan kepemilikan telpon rumah. 1 artinya punya and 0 berarti tidak.
15. FLAG_EMAIL: kolom yang menunjukan kepemilikan email. 1 artinya punya and 0 berarti tidak.
16. OCCUPATION_TYPE: jenis pekerjaan. 
17. CNT_FAM_MEMBERS: Jumlah anggota keluarga.

Statistik deskriptif data :
![stats-deskripsi](https://user-images.githubusercontent.com/62064078/158396570-89ece62d-adba-43a7-946c-f28f84d4dc51.png)


Kolom Numerik dan Categorical :
![kolom-numerik-dan-categorical](https://user-images.githubusercontent.com/62064078/158396628-e7abf22d-1667-4d24-b17d-7da331a27fa0.jpg)


Null values :
![null-value](https://user-images.githubusercontent.com/62064078/158396680-ba3fee11-7adb-49ed-a448-48b0caf470e4.png)


Outliers :
![outliers](https://user-images.githubusercontent.com/62064078/158396708-6088be00-80dd-40d5-a477-d00cec6bba12.png)




### Visualisasi

- Aplikasi yang dikirim berdasarkan gender <br />
![download1](https://user-images.githubusercontent.com/62064078/158396767-31df1982-3d39-4d15-abc2-e74bc400a8cf.png)


- Aplikasi yang disetujui berdasarkan gender <br />
![download2](https://user-images.githubusercontent.com/62064078/158396785-bae828b1-0a1d-4cbd-a7d0-d743a92c15f9.png)


- Aplikasi yang dikirim berdasarkan kepemilikan mobil <br />
![download3](https://user-images.githubusercontent.com/62064078/158396798-4ab7734a-a938-4f74-998d-0c6b221fd112.png)


- Aplikasi yang dikirim berdasarkan kepemilikan porperti <br />
![download4](https://user-images.githubusercontent.com/62064078/158396806-9d984ae8-3788-4dec-aca4-661e140ea8c1.png)


- Aplikasi yang dikirim berdasarkan jumlah anak <br />
![download5](https://user-images.githubusercontent.com/62064078/158396811-000aac31-9b61-437e-bd10-688b21582782.png)



- Histogram total income  <br />
![download6](https://user-images.githubusercontent.com/62064078/158396835-423bcbb8-f96f-47ff-a3b5-203f2fc52cc2.png)


- Aplikasi yang dikirim berdasarkan jenis income <br />
![download7](https://user-images.githubusercontent.com/62064078/158396858-a2f9204f-8187-4826-99f4-93851052c082.png)

1 artinya punya and 0 berarti tidak

- Aplikasi yang dikirim berdasarkan status keluarga <br />
![download8](https://user-images.githubusercontent.com/62064078/158396911-375aff95-02e0-4838-846c-af981d8a5f6b.png)


- Aplikasi yang dikirim berdasarkan tipe tempat tinggal <br />
![download9](https://user-images.githubusercontent.com/62064078/158396923-4ef67e2d-ec97-42f8-9a1e-6ba01ab61d61.png)


- Aplikasi yang dikirim berdasarkan age <br />
![download10](https://user-images.githubusercontent.com/62064078/158396934-04890dbd-92e5-44f7-99a1-6a9d159655c4.png)


- Heatmap correlation dari tiap atribut <br />
![download](https://user-images.githubusercontent.com/62064078/158396952-81fab74d-e9ac-4e70-9eca-d9f0cc2520cf.png)



## Data Preparation

- Data Cleaning <br />
Data Cleaning adalah proses pembersihan data. Pada step kali ini kita akan membersihkan fitur yang tidak terlalu berpengaruh terhadap sebuah keputusan diterimanya sebuah pengajuan kredit. Disini kita akan menghapus beberapa fitur karena tidak berpengaruh terhadap persetujuan pengajuan kredit seperti :
    - Occupation Type : jenis pekerjaan
    - FLAG_MOBILE : kepemilikan telpon seluler
    - FLAG_WORK_PHONE: kepemilikan telpon kantor
    - FLAG_PHONE : kepemilikan telpon rumah
    - FLAG_EMAIL : kepemilikan email


- Mengatasi pencilan atau outlier<br /> 
Pada tahapan ini berguna untuk menghapus data tidak normal pada dataset dengan menggunakan teknik IQR method. IQR merupakan interquartile yang dapat diformulasikan Q3 - Q1. 
Untuk mendeteksi pencilan langkah pertama mengalikan IQR dengan 1.5. Kemudian tentukan batas bawah dan batas atas dengan cara, batas bawah = Q1 - 1.5 * IQR dan batas atas = Q3 + 1.5 * IQR. 
Maka data yang bukan outlier adalah data yang berapa pada rentang batas bawah hingga batas atas.

- Membagi data menjadi data training dan testing <br />
Tahapan ini bertujuan agar model yang dilatih dapat diuji dengan data yang berbeda dari data yang digunakan dalam pelatihan. Data dapat dibagi menjadi dua atau tiga, pada proyek ini data dibagi menjadi dua dengan persentase untuk training sebesar 80% dan sisanya 20% untuk testing. Fungsi train_test_split pada library sklearn yang akan digunakan untuk menangani tahapan ini.

- Transformasi pada kolom categorical <br />
Mesin hanya dapat membaca angka berupa nol dan satu, oleh karenanya perlu adanya mekanisme untuk membuat mesin bisa menerima masukan berupa karakter, string maupun object. Proses transformasi pada fitur kategorikal sering kali disebut encoding. pada kali ini encoding yang digunakan adalah OneHotEncoder

- Melakukan scaling <br />
pada percobaan kali ini saya menggunakan MinMaxScaler. Min-Max Scaling, yang sering dikenal juga dengan normalisasi data atau normalization (karena z-score juga sering disebut normalization, maka sering terjadi ambiguitas atau tertukar-tukar.
Min-Max Scaling bekerja dengan scaling data/menyesuaikan data dalam rentang/range tertentu (range nilai minimum hingga nilai maksimum), dengan rentang yang biasa digunakan adalah 0 hingga 1

- Merge Dataset <br>
disini saya melakuakan merge data credit_record yang berisi riwayat pengajuan kredit dengan dataset apllication_record yang berisi tentang status approval pengajuan kredit dari data credit_record 


- Membuat Pipeline<br />
Pada tahapan sebelumnya telah dipaparkan mengapa penting membagi data menjadi data training dan testing. Data testing haruslah data yang benar-benar belum pernah dilihat oleh model. Oleh karenanya pada tahap pre-processing seperti transformasi, penyekalaan harus dilakukan pada data testing saja. Jika penyelakaan atau transformasi dilakukan terhadap semua data dalam tanda kutip dilakukan sebelum pembagian data train dan testing akan menyebabkan kebocoran data yang dapat menyebabkan model menjadi overfit kedepannya. Kebocoran data adalah suatu momen ketika model sudah pernah melihat data testing. Bisa dianalogikan seperti seorang yang sedang berlatih untuk ujian dan mendapatkan bocoran mengenai ujian pertama, maka pada ujian pertama dia akan mendapatkan nilai yang baik namun jika ada ujian kedua belum tentu akan mendapat nilai yang baik. Oleh karenanya untuk mengukur seseorang menguasai materi dengan baik haruslah diuji dengan soal atau ujian yang belum pernah dikerjakan sebelumnya. Apa hubungannya dengan pipeline ? jika dalam preprocessing tersebut dilakukan secara manual akan banyak sekali tahapan dan variabel yang perlu diingat. Sebagai contoh misal dalam menangani data kosong dengan rata-rata, nilai rata-rata tersebut harus diingat ketika ingin menyekalakan data testing. Berikut ilustrasi dari pipeline dalam proyek ini. <br> <br>
![pipeline-diagram](https://user-images.githubusercontent.com/62064078/158397264-e92b7fda-9d6d-4ac7-b809-1e2d8954651f.jpg)

<br>
<br>
<br>

## Modeling<br />
Model – model yang saya pakai dalam projek ini adalah:

### SVM
Support Vector Machine (SVM) merupakan salah satu metode dalam supervised learning yang biasanya digunakan untuk regresi (Support Vector Regression).SVM digunakan untuk mencari hyperplane terbaik dengan memaksimalkan jarak antar kelas. Hyperplane adalah sebuah fungsi yang dapat digunakan untuk pemisah antar kelas. Dalam 2-D fungsi yang digunakan untuk klasifikasi antar kelas disebut sebagai line whereas, fungsi yang digunakan untuk klasifikasi antas kelas dalam 3-D disebut plane similarly, sedangan fungsi yang digunakan untuk klasifikasi di dalam ruang kelas dimensi yang lebih tinggi di sebut hyperplane.
- Pipeline SVM <br>
![pipeline-svm](https://user-images.githubusercontent.com/62064078/158397296-56f956bc-a91b-4247-8eac-76ebc1ac66f1.jpg)

<br>
Parameter yang digunakan pada model SVM antara lain : <br >
disini saya menggunakan parameter default dari SVM untuk regresi dan hanya mengganti kernelnya menjadi linear.<br> <br> <br>

### XGBoost
XGBoost atau eXtreme Gradient Boosting adalah algoritma berbasis pohon. XGBoost adalah bagian dari keluarga pohon (Decision tree, Random Forest, bagging, boosting, gradient boosting). Kekuatan XGBoost adalah paralelisme dan pengoptimalan perangkat keras. Data disimpan dalam memori, disebut blok, dan disimpan dalam format kolom terkompresi [CSC]. Algoritma tersebut dapat melakukan pemangkasan pohon untuk menghilangkan cabang yang probabilitasnya rendah. Fungsi kerugian model memiliki istilah untuk menghukum kompleksitas model dengan regularisasi untuk memperlancar proses pembelajaran (mengurangi kemungkinan overfitting).
- Pipeline XGBoost<br>
![pipeline-xgb](https://user-images.githubusercontent.com/62064078/158397511-e7df4d1c-bb66-4439-9c85-b52ef25059ca.jpg)


## Evaluasi
Metrik evaluasi adalah dasar yang digunakan untuk menentukan performa model dan model pemenang ditempatkan di papan peringkat.
Matriks konfusi adalah metode umum yang digunakan untuk menentukan dan memvisualisasikan kinerja model klasifikasi. Matriks konfusi adalah matriks NxN dengan N adalah jumlah kelas atau nilai target.

Baris dalam matriks kebingungan mewakili nilai aktual sedangkan kolom mewakili nilai prediksi.

Istilah yang perlu diperhatikan dalam Confusion Matrix
 - Positif benar : Positif benar terjadi jika model telah memprediksi instance Benar dengan benar.
 - True Negatives : True negative adalah kasus ketika model secara akurat memprediksi instance False.
 - Positif palsu : Positif palsu adalah situasi di mana model telah memprediksi nilai True ketika nilai sebenarnya adalah False.
 - Negatif palsu : Negatif palsu adalah situasi di mana model memprediksi nilai False ketika nilai sebenarnya adalah True.
 

Berikut adalah tabel evaluasi dari kedua model:

- Metrik akurasi mengukur jumlah kelas yang diprediksi dengan benar oleh model - true positive dan true negative.

    ![acc_formula](https://user-images.githubusercontent.com/62064078/158397564-a983e3df-7a64-43f8-9a2b-67bc99d7bf24.png)


- Mean Squared Error (MSE) adalah Rata-rata Kesalahan kuadrat diantara nilai aktual dan nilai peramalan. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Metode Mean Squared Error biasanya digunakan untuk mengevaluasi metode pengukuran dengan model regressi

    ![rumus_MSE](https://user-images.githubusercontent.com/62064078/158397598-748f1c31-6b49-4e34-811e-4836d761b8c0.jpg)


<br>

![acc](https://user-images.githubusercontent.com/62064078/158397643-ca245d39-ab91-4004-94b8-bc984331a73e.jpg)






dan berikut adalah confusion matrix dari model SVM


![coff-matrix-svm](https://user-images.githubusercontent.com/62064078/158397659-6899c64a-09c2-4e00-92d7-dc006ff239f6.png)

Berikut confusion matrix dari model XGBoost

![coff-matrix-xgb](https://user-images.githubusercontent.com/62064078/158397680-bfe7e3cf-a3f0-4e96-ba14-9a51b9caa618.png)











