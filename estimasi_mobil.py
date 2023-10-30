import pickle
import streamlit as st

model = pickle.load(open('estimasi_mobil.sav', 'rb'))

st.title('Estimasi Harga Mobil Bekas')

year = st.number_input('Input Tahun Mobil')
mileage = st.number_input('Input km Mobil')
tax = st.number_input('input Pajak Mobil')
mpg = st.number_input('Input Konsumsi BBM Mobil')
engineSize = st.number_input('Input Engine Size Mobil')

predict = ''

if st.button('Estimasti Harga') :
    predict = model.predict(
        [[year,mileage,tax,mpg, engineSize]]
    )
    st.write('Estimasi Harga mobil dalam ponds : ', predict)