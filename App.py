import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("tomato_disease_model.h5")
class_names = ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold',
               'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Mosaic Virus',
               'Yellow Leaf Curl Virus', 'Healthy']

st.title("🍅 Tomato Leaf Disease Detector")

uploaded_file = st.file_uploader("Upload a tomato leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.success(f"🌿 Detected: {predicted_class}")
