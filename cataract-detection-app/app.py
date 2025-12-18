# ==================== CATARACT DETECTION APP ====================

import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2
import time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Cataract Detection",
    page_icon="",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    """Load trained model (cached)"""
    try:
        model = keras.models.load_model('best_cataract_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==================== PREPROCESSING FUNCTION ====================
def preprocess_image(image):
    """
    Preprocess image untuk prediksi
    - Resize ke 224x224
    - Normalize ke 0-1
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Resize
    img_resized = cv2.resize(img_array, (224, 224))
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch

# ==================== PREDICTION FUNCTION ====================
def predict(model, image):
    """
    Predict apakah gambar Cataract atau Normal
    """
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Predict
    prediction = model.predict(processed_img, verbose=0)
    confidence = float(prediction[0][0])
    
    # Interpretation
    # Model output: 0 = Cataract, 1 = Normal
    if confidence > 0.5:
        label = "Normal"
        confidence_score = confidence * 100
    else:
        label = "Cataract"
        confidence_score = (1 - confidence) * 100
    
    return label, confidence_score

# ==================== MAIN APP ====================
def main():
    # Header
    st.title("Cataract Detection System")
    st.markdown("""
    Upload gambar mata untuk deteksi katarak menggunakan Deep Learning (MobileNetV2).
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Model Info:**
        - Architecture: MobileNetV2
        - Accuracy: 95.58%
        - Sensitivity: 100%
        - Specificity: 91%
        
        **Cara Pakai:**
        1. Upload gambar mata
        2. Klik 'Analyze'
        3. Lihat hasil prediksi
        """)
        
        st.markdown("---")
        st.markdown("**⚠️ Disclaimer:**")
        st.caption("Aplikasi ini hanya untuk edukasi. Bukan pengganti diagnosis medis profesional.")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ Model gagal dimuat. Pastikan file 'best_cataract_model.keras' ada di folder yang sama.")
        return
    
    st.success("✅ Model berhasil dimuat!")
    
    # File uploader
    st.markdown("---")
    st.subheader("📤 Upload Gambar Mata")
    
    uploaded_file = st.file_uploader(
        "Pilih gambar (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar fundus mata untuk dianalisis"
    )
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original Image:**")
            st.image(image, use_column_width=True)
        
        # Analyze button
        if st.button(" Analyze", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                # Simulate processing time
                time.sleep(0.5)
                
                # Predict
                label, confidence = predict(model, image)
                
                # Display results
                with col2:
                    st.markdown("**Analysis Result:**")
                    
                    # Result card
                    if label == "Normal":
                        st.success(f"### ✅ {label}")
                        st.progress(confidence / 100)
                        st.metric("Confidence", f"{confidence:.2f}%")
                    else:
                        st.error(f"### ⚠️ {label} Detected")
                        st.progress(confidence / 100)
                        st.metric("Confidence", f"{confidence:.2f}%")
                
                # Additional info
                st.markdown("---")
                st.info("""
                **🩺 Rekomendasi:**
                - **Normal**: Mata sehat, lakukan pemeriksaan rutin tahunan
                - **Cataract**: Konsultasi dengan dokter mata untuk pemeriksaan lebih lanjut
                """)
    
    else:
        # Instructions
        st.info("Upload gambar mata untuk memulai analisis!")
        
        # Sample images (optional)
        st.markdown("---")
        st.markdown("**💡 Tips:**")
        st.markdown("""
        - Gunakan gambar fundus mata yang jelas
        - Pastikan pencahayaan cukup
        - Hindari gambar yang blur
        """)

# ==================== RUN APP ====================
if __name__ == "__main__":
    main()