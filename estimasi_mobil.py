import pickle
import streamlit as st

model = pickle.load(open('D:\machinelearning\estimasi_mobil.sav', 'rb'))

st.title ('Estimasi Harga Mobil')
year = st.number_input('input tahun mobil')
mileage = st.number_input('input km mobil')
tax = st.number_input('input pajak mobil')
mpg = st.number_input('input mpg mobil')
engineSize = st.number_input('input ukuran mesin mobil')

predict = ''


if st.button('Estimasti Harga') :
    predict = model.predict(
        [[year,mileage,tax,mpg, engineSize]]
    )
    st.write('Estimasi Harga mobil dalam ponds : ', predict)
