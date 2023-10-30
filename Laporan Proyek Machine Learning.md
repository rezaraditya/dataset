Laporan Proyek Machine Learning
Nama : Reza Raditya
Nim : 231352004
Kelas : Pagi B
Domain Proyek
Mengestimasikan data klasifikasi dan harga Mobil, sehingga mendapat harga pasar yang relevan sesuai dengan kualitas dan kuantitas mobil
**

Business Understanding
Dalam dataset ini mencakup banyak data-data klarifikasi mobil oleh karena itu data harus dilakukan filtering sehingga memudahkan dalam pengisian input data Bagian laporan ini mencakup:

Problem Statements
Menjelaskan pernyataan masalah latar belakang:

klarifikasi data masih berupa acak dan tidak rapi (jika di heatmap)
Goals
mencari dan mengklasifikasikan data dengan secara detail dan rapi
Solution statements
- dapat mengklasifikan data dan mencari data yang sesuai dengan yang diinputkan costumer
Data Understanding
dataset : https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes/data

Variabel-variabel pada used-car-dataset adalah sebagai berikut:
1 year : Tahun Mobil
2 price : Harga Mobil 3 transmission : Transimi Mobil 4 mileage : Total jarak tempuh mobil yang telah dijalankan
5 fuelType : Tipe BBM 6 tax : Jumlah Pajak
7 mpg : Jumlah Bahan Bakar yang digunakan ketika mobil bergerak (perliter ) 8 engineSize : Ukuran Mesin

Data Preparation
Mengimport libary
import pandas as pd import numpy as np import matplotlib.pyplot as plt import seaborn as sns

Memanggil Dataset
df = pd.read_csv('toyota.csv')

DEKSRIPSI DATASET
df.head(), df.info(), sns.heatmap(df.isnull()), df.describe()

VISUALISAI DATA
plt.figure(figsize=(10,8)) sns.heatmap(df.corr(), annot=True) models = df.groupby('model').count()[['tax']].sort_values(by='tax' ,ascending=True).reset_index() models = models.rename(columns={'tax' : 'numberOfCars'})

fig = plt.figure(figsize=(15,5)) sns.barplot(x=models['model'], y=models['numberOfCars'], color='blue') plt.xticks(rotation=60)

Ukuran Mesin
engine = df.groupby('engineSize').count()[['tax']].sort_values(by='tax' ,ascending=True).reset_index() engine =engine.rename(columns={'tax' : 'count'})

fig = plt.figure(figsize=(15,5)) sns.barplot(x=engine['engineSize'], y=engine['count'], color='blue') plt.xticks(rotation=60)

distribusi mileage
fig = plt.figure(figsize=(15,5)) sns.distplot(df['mileage']) plt.xticks(rotation=60)
distribusi Price
fig = plt.figure(figsize=(15,5)) sns.distplot(df['price']) plt.xticks(rotation=60)
seleksi fitur
features = ['year', 'mileage', 'tax', 'mpg', 'engineSize'] x = df[features] y = df['price'] x.shape, y.shape
split data training dan testing
from sklearn.model_selection import train_test_split x_train, X_test, y_train, y_test = train_test_split(x,y,random_state=70) y_test.shape

**.

Modeling
membuat model regresi linier
-from sklearn.linear_model import LinearRegression lr = LinearRegression() lr.fit(x_train, y_train) pred = lr.predict(X_test)

score = lr.score(X_test, y_test) print('akurasi model regresi linear= ', score)
membuat input data model regresi linier
#years = 2019 , mileage = 5000, tax=145, mpg=30.2, enginesize = 2 input_data = np.array([[2019,5000,145,30.2,2]]) prediction = lr.predict(input_data) print('Estimasi Harga Mobil dalam EUR : ', prediction)
save model
import pickle filename = 'esti_mobil.sav' pickle.dump(lr,open(filename,'wb')) Jelaskan proses improvement yang dilakukan.
Proses Improvement ini menggunakan regresi linier karena dengan regresi linier proses lebih teratur
Evaluation
tidak ada evaluasi
Deployment
https://dataset-hzeytbrghafkrmxgjfqt6n.streamlit.app/

---Ini adalah bagian akhir laporan---

_
