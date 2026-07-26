import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load model (di-cache biar tidak reload tiap kali user berinteraksi)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_optimasi_final.h5")

model = load_model()

# Urutan label HARUS sama dengan class_indices saat training: {'Cat': 0, 'Dog': 1}
labels = ["Cat", "Dog"]

IMG_SIZE = (150, 150)  # harus sama persis dengan IMG_HEIGHT/IMG_WIDTH saat training

st.title("Klasifikasi Kucing vs Anjing")
st.write("Upload gambar kucing atau anjing, model CNN akan memprediksi kelasnya.")

uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diupload", use_container_width=True)

    if st.button("Prediksi"):
        img_resized = image.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_idx = np.argmax(prediction)
        predicted_class = labels[predicted_idx]
        confidence = float(prediction[0][predicted_idx]) * 100

        st.success(f"Prediksi: **{predicted_class}**")
        st.write(f"Tingkat keyakinan: {confidence:.2f}%")

        st.write("Detail probabilitas:")
        for lbl, prob in zip(labels, prediction[0]):
            st.write(f"- {lbl}: {prob*100:.2f}%")