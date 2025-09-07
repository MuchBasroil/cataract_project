import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cataract_mobilenetv2.h5")
    return model

model = load_model()

# Preprocessing function
def preprocess_image(img):
    img = img.resize((224, 224))   # Sesuai input model
    img = np.array(img) / 255.0    # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambah batch dimensi
    return img

# Streamlit App
st.title("Deteksi Kataraktos")
st.write("Upload foto muatamu untuk dicek apakah **Normal** atau **Cataract Yek**")

uploaded_file = st.file_uploader("Uploaden Gambar Matamu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_column_width=True)

    if st.button("🔍 Deteksi"):
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0][0]

        if prediction > 0.5:
            st.error(f"Hasil: **Cataract** (probabilitas {prediction:.2f})")
        else:
            st.success(f"Hasil: **Normal** (probabilitas {1-prediction:.2f})")
