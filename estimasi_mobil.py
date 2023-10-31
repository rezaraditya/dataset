import pickle
import streamlit as st

model = pickle.load(open('estimasi_mobil.sav', 'rb'))

st.title ('Estimasi Harga Mobil')
model = st.number_input('input model mobil')
year = st.number_input('input Lama Tahun mobil')
mileage = st.number_input('input km mobil')
tax = st.number_input('input pajak mobil')
mpg = st.number_input('input mpg mobil')
engineSize = st.number_input('input ukuran mesin mobil')

predict = ''


if st.button('Estimasti Harga') :
    predict = model.predict(
        [[model,year,mileage,tax,mpg, engineSize]]
    )
    st.write('Estimasi Harga mobil dalam ponds : ', predict)
