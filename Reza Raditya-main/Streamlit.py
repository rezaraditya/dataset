import pickle
import streamlit as st

# Memuat model yang telah dilatih
model = pickle.load(open('Prediksi_Jenis_kanker_payudara.sav', 'rb'))

st.title('Prediksi Diagnosa pada kanker payudara')

keterangan = """kanker jinak: 

radius_mean = 0.1 - 14.0 

radius_worst = 0.1 - 16.00

texture_mean = 0.1 - 17.00 

texture_worst = 0.1 - 28.00 

perimeter_mean = 0.1 - 90.0 

perimeter_worst = 0.1 - 85.00

kanker ganas:

radius_mean = 14.0 - 90.00

radius_worst = 16.00 - 90.00

texture_mean = 17.00 - 90.00

texture_worst = 28.00 - 90.00

perimeter_mean = 90.0 - 150.00

perimeter_worst = 85.00- 185.00

jika selain data tersebut = tidak terdeteksi"""

st.markdown(keterangan)

# Menambahkan dropdown untuk memilih tipe kanker payudara
kanker_type = st.selectbox('Pilih Tipe Kanker Payudara:', ['Kanker Jinak', 'Kanker Ganas'])

radius_mean = st.number_input('Input radius_mean')
radius_worst = st.number_input('Input radius_worst')
texture_mean = st.number_input('Input texture_mean')
texture_worst = st.number_input('Input texture_worst')
perimeter_mean = st.number_input('Input perimeter_mean')
perimeter_worst = st.number_input('Input perimeter_worst')

predict = ''

if st.button('Estimasi Diagnosa'):
    # Melakukan prediksi berdasarkan tipe kanker yang dipilih
    if kanker_type == 'Kanker Jinak':
        if (0.1 <= radius_mean <= 14.0) and (0.1 <= radius_worst <= 16.00) and (0.1 <= texture_mean <= 17.00) and (0.1 <= texture_worst <= 28.00) and (0.1 <= perimeter_mean <= 90.0) and (0.1 <= perimeter_worst <= 85.00):
            predict = 'Kanker Jinak'
        else:
            predict = 'Tidak terdeteksi'
    else:
        if (14.0 <= radius_mean <= 90.00) and (16.00 <= radius_worst <= 90.00) and (17.00 <= texture_mean <= 90.00) and (28.00 <= texture_worst <= 90.00) and (90.0 <= perimeter_mean <= 150.00) and (85.00 <= perimeter_worst <= 185.00):
            predict = 'Kanker Ganas'
        else:
            predict = 'Tidak terdeteksi'
        
    st.write('Hasil Diagnosa:', predict)