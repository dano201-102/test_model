import streamlit as st
import pandas as pd
import joblib

# Load model & encoder
model = joblib.load("model_preferensi.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load data wisata
df = pd.read_csv("data_tempat_wisata_bali.csv")
df_clean = df.dropna(subset=['rating'])

st.title("🧭 NusaBali - Rekomendasi Tempat Wisata di Bali")

st.markdown("Masukkan preferensimu, kami akan rekomendasikan tempat yang cocok!")

# Input pengguna
kategori = st.selectbox("Kategori", df_clean['kategori'].unique())
kabupaten = st.selectbox("Kabupaten/Kota", df_clean['kabupaten_kota'].unique())
rating = st.slider("Rating", 1.0, 5.0, 4.5)

if st.button("Rekomendasikan"):
    # Buat DataFrame input
    input_data = pd.DataFrame({
        'kategori': [kategori],
        'kabupaten_kota': [kabupaten],
        'rating': [rating]
    })

    # Prediksi preferensi
    predicted_pref = model.predict(input_data)
    pref_label = label_encoder.inverse_transform(predicted_pref)[0]

    # Cari tempat wisata dengan preferensi yang sama
    rekomendasi = df_clean[df_clean['preferensi'] == pref_label]

    if not rekomendasi.empty:
        st.success(f"Rekomendasi wisata berdasarkan preferensi **{pref_label}**:")
        st.write(rekomendasi[['nama', 'kategori', 'kabupaten_kota', 'rating', 'link']].reset_index(drop=True))
    else:
        st.warning("Maaf, belum ada tempat wisata yang cocok ditemukan.")
