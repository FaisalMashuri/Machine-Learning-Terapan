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
1. ID: Unique Id of the row
2. CODE_GENDER: Gender of the applicant. M is male and F is female.
3. FLAG_OWN_CAR: Is an applicant with a car. Y is Yes and N is NO.
4. FLAG_OWN_REALTY: Is an applicant with realty. Y is Yes and N is No.
5. CNT_CHILDREN: Count of children.
6. AMT_INCOME_TOTAL: the amount of the income.
7. NAME_INCOME_TYPE: The type of income (5 types in total).
8. NAME_EDUCATION_TYPE: The type of education (5 types in total).
9. NAME_FAMILY_STATUS: The type of family status (6 types in total).
10. DAYS_BIRTH: The number of the days from birth (Negative values).
11. DAYS_EMPLOYED: The number of the days from employed (Negative values).  values.
12. FLAG_MOBIL: Is an applicant with a mobile. 1 is True and 0 is False.
13. FLAG_WORK_PHONE: Is an applicant with a work phone. 1 is True and 0 is False.
14. FLAG_PHONE: Is an applicant with a phone. 1 is True and 0 is False.
15. FLAG_EMAIL: Is an applicant with a email. 1 is True and 0 is False.
16. OCCUPATION_TYPE: The type of occupation (19 types in total). 
17. CNT_FAM_MEMBERS: The count of family members.


### Visualisasi
![output_1](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(1).png?raw=true)
![output_1](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(2).png?raw=true)
![output_1](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(3).png?raw=true)
![output_1](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(4).png?raw=true)
![output_1](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(5).png?raw=true)
![output_1](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(6).png?raw=true)
![output_1](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(7).png?raw=true)
![output_2](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(8).png?raw=true)
![output_2](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(9).png?raw=true)
![output_2](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(10).png?raw=true)
![output_2](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download.png?raw=true)

## Data Preparation
- Mengatasi pencilan atau outlier Pada tahapan ini berguna untuk menghapus data tidak normal pada dataset dengan menggunakan teknik IQR method. IQR merupakan interquartile yang dapat diformulasikan Q3 - Q1. Untuk mendeteksi pencilan langkah pertama mengalikan IQR dengan 1.5. Kemudian tentukan batas bawah dan batas atas dengan cara, batas bawah = Q1 - 1.5 * IQR dan batas atas = Q3 + 1.5 * IQR. Maka data yang bukan outlier adalah data yang berapa pada rentang batas bawah hingga batas atas. Untuk lebih jelasnya silahkan kunjungi tautan berikut.

- Membagi data menjadi data training dan testing
Tahapan ini bertujuan agar model yang dilatih dapat diuji dengan data yang berbeda dari data yang digunakan dalam pelatihan. Data dapat dibagi menjadi dua atau tiga, pada proyek ini data dibagi menjadi dua dengan persentase untuk training sebesar 80% dan sisanya 20% untuk testing. Fungsi train_test_split pada library sklearn yang akan digunakan untuk menangani tahapan ini.

- Transformasi pada kolom categorical
Mesin hanya dapat membaca angka berupa nol dan satu, oleh karenanya perlu adanya mekanisme untuk membuat mesin bisa menerima masukan berupa karakter, string maupun object. Proses transformasi pada fitur kategorikal sering kali disebut encoding. pada kali ini encoding yang digunakan adalah OneHotEncoder

- Melakukan teknik resample Pada kasus klasifikasi dataset yang tidak seimbang akan membuat bias pada model yang cenderung mengarah pada kategori yang lebih banyak datanya. Walau jika diukur dari segi akurasi bisa jadi model memiliki akurasi yang tinggi, namun salah dalam memprediksi semua data minoritas. Hal tersebut menjadikan model tidak sesuai harapan. Oleh karenanya tahapan resample ini dilakukan, teknik ini membuat data dummy atau data buatan. Banyak cara atau metode atau teknik dalam tahapan resample, namun pada proyek ini teknik SMOTE atau synthetic minority oversampling technique yang bekerja dengan memanfaatkan algoritma KNN atau K-nearest-neighbor dengan memilih secara acak data dari kelas minoritas, kemudian mencari tetangga terdekat lalu membentuk beberapa data buatan di antara data dengan tetangganya. Dalam implementasi, proyek ini menggukan library imblearn.

- Membuat Pipeline Pada tahapan sebelumnya telah dipaparkan mengapa penting membagi data menjadi data training dan testing. Data testing haruslah data yang benar-benar belum pernah dilihat oleh model. Oleh karenanya pada tahap pre-processing seperti transformasi, penyekalaan harus dilakukan pada data testing saja. Jika penyelakaan atau transformasi dilakukan terhadap semua data dalam tanda kutip dilakukan sebelum pembagian data train dan testing akan menyebabkan kebocoran data yang dapat menyebabkan model menjadi overfit kedepannya. Kebocoran data adalah suatu momen ketika model sudah pernah melihat data testing. Bisa dianalogikan seperti seorang yang sedang berlatih untuk ujian dan mendapatkan bocoran mengenai ujian pertama, maka pada ujian pertama dia akan mendapatkan nilai yang baik namun jika ada ujian kedua belum tentu akan mendapat nilai yang baik. Oleh karenanya untuk mengukur seseorang menguasai materi dengan baik haruslah diuji dengan soal atau ujian yang belum pernah dikerjakan sebelumnya. Apa hubungannya dengan pipeline ? jika dalam preprocessing tersebut dilakukan secara manual akan banyak sekali tahapan dan variabel yang perlu diingat. Sebagai contoh misal dalam menangani data kosong dengan rata-rata, nilai rata-rata tersebut harus diingat ketika ingin menyekalakan data testing. Library scikit-learn menyediakan pipeline untuk menangani masalah ini, namun belum terintegrasi dengan library imblearn, oleh karenanya pipeline yang digunakan adalah pipeline yang disegiakan oleh imblearn. Berikut ilustrasi dari salah satu pipeline dalam proyek ini.

## Modeling
Model – model yang saya pakai dalam projek ini adalah:


### SVM
- Pipeline SVM
![pipeline SVM](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/Screenshot%202022-03-07%20235735.jpg?raw=true)

### XGBoost
- Pipeline XGBoost
![](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/Screenshot%202022-03-07%20235826.jpg?raw=true)

## Evaluasi
Berikut adalah tabel evaluasi dari kedua model:
![](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/Screenshot%202022-03-08%20002532.jpg?raw=true)

dan berikut adalah confusion matrix dari model SVM
![](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(11).png?raw=true)

Berikut confusion matrix dari model XGBoost
![](https://github.com/FaisalMashuri/Machine-Learning-Terapan/blob/main/img/download%20(12).png?raw=true)










