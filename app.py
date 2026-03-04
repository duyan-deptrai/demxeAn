
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vehicle Detector", page_icon="🚗")

st.title("🚗 Vehicle Detection App")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("vehicle_counter_model.h5")

model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((128,128))

    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.write(f"Confidence: {prediction:.4f}")

    if prediction > 0.5:
        st.success("🚗 Vehicle Detected")
    else:
        st.error("❌ No Vehicle Detected")
