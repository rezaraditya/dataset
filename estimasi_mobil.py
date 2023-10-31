import pickle
import streamlit as st

model = pickle.load(open('estimasi_mobil.sav', 'rb'))

st.title ('Estimasi Harga Mobil')

year = st.number_input('input Lama Tahun mobil')
price = st.number_input('input price mobil')
mileage = st.number_input('input km mobil')
tax = st.number_input('input pajak mobil')
mpg = st.number_input('input mpg mobil')
engineSize = st.number_input('input ukuran mesin mobil')

predict = ''


if st.button('Estimasti Harga') :
    predict = model.predict(
        [[year,price,mileage,tax,mpg, engineSize]]
    )
    st.write('Estimasi Harga mobil dalam ponds : ', predict)
