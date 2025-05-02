import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan label encoder
model = joblib.load("model_preferensi.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load data asli (misalnya untuk ditampilkan)
df = pd.read_csv("data_tempat_wisata_bali.csv")

st.title("🧭 NusaBali - Rekomendasi Tempat Wisata di Bali")

st.markdown("Masukkan preferensi kamu, dan kami akan merekomendasikan jenis wisata yang cocok.")

# Input dari user
kategori = st.selectbox("Kategori Tempat", df['kategori'].unique())
kabupaten = st.selectbox("Kabupaten/Kota", df['kabupaten_kota'].unique())
rating = st.slider("Rating", 1.0, 5.0, 4.5)
latitude = st.number_input("Latitude", value=-8.5)
longitude = st.number_input("Longitude", value=115.2)

# Prediksi ketika tombol ditekan
if st.button("Rekomendasikan"):
    input_data = pd.DataFrame({
        'kategori': [kategori],
        'kabupaten_kota': [kabupaten],
        'rating': [rating],
        'latitude': [latitude],
        'longitude': [longitude]
    })

    prediction = model.predict(input_data)
    predicted_label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🎯 Rekomendasi wisata kamu adalah: **{predicted_label}**")
