import streamlit as st
import pandas as pd
import joblib

# Load model & label encoder
model = joblib.load("model_preferensi.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load data tempat wisata
df = pd.read_csv("data_tempat_wisata_bali.csv")
df_clean = df.dropna(subset=['rating'])

st.set_page_config(page_title="NusaBali - Rekomendasi Wisata", page_icon="🧭")
st.title("🧭 NusaBali - Rekomendasi Tempat Wisata di Bali")
st.markdown("Masukkan preferensimu, dan kami akan merekomendasikan tempat yang cocok!")

# Input dari user
kategori = st.selectbox("Pilih Kategori Tempat Wisata", df_clean['kategori'].unique())
kabupaten = st.selectbox("Pilih Kabupaten/Kota", df_clean['kabupaten_kota'].unique())
rating = st.slider("Berapa minimal rating yang kamu inginkan?", 1.0, 5.0, 4.5)

# Tombol rekomendasi
if st.button("🎯 Rekomendasikan"):
    # Siapkan input untuk model
    input_df = pd.DataFrame({
        'kategori': [kategori],
        'kabupaten_kota': [kabupaten],
        'rating': [rating]
    })

    # Prediksi preferensi
    predicted = model.predict(input_df)
    preferensi_pred = label_encoder.inverse_transform(predicted)[0]

    st.subheader(f"Hasil Prediksi Preferensi: **{preferensi_pred}**")

    # Filter rekomendasi berdasarkan preferensi & kabupaten
    rekomendasi = df_clean[
        (df_clean['preferensi'] == preferensi_pred) &
        (df_clean['kabupaten_kota'] == kabupaten)
    ]

    # Fallback jika tidak ada hasil spesifik
    if rekomendasi.empty:
        st.info("🙁 Tidak ditemukan tempat di kabupaten ini. Menampilkan semua wisata dengan preferensi yang sama.")
        rekomendasi = df_clean[df_clean['preferensi'] == preferensi_pred]

    # Tampilkan hasil rekomendasi
    st.success(f"✅ Ditemukan {len(rekomendasi)} tempat wisata:")
    st.dataframe(
        rekomendasi[['nama', 'kategori', 'kabupaten_kota', 'rating', 'link']].reset_index(drop=True),
        use_container_width=True
    )
