import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Animal Classifier", page_icon="🐾", layout="centered")

MODEL_PATH = Path("CAT.keras")
CLASSES_PATH = Path("classes.json")
IMAGE_SIZE = (64, 64)

st.title("🐾 Animal Image Classifier")
st.write("Upload an image and the trained CNN model will predict whether it contains a dog, cat, rabbit, or duck.")

@st.cache_resource
def load_classifier():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "CAT.keras was not found. Train the model in your notebook first "
            "and save it in the same folder as app.py."
        )
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            "classes.json was not found. Run the notebook cell that saves "
            "data.class_names to classes.json."
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with CLASSES_PATH.open("r", encoding="utf-8") as f:
        class_names = json.load(f)
    return model, class_names

uploaded_file = st.file_uploader(
    "Choose an animal image",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify image", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing image..."):
                model, class_names = load_classifier()
                resized = image.resize(IMAGE_SIZE)
                array = tf.keras.utils.img_to_array(resized)
                batch = tf.expand_dims(array, axis=0)
                predictions = model.predict(batch, verbose=0)[0]

                predicted_index = int(np.argmax(predictions))
                predicted_class = class_names[predicted_index]
                confidence = float(predictions[predicted_index]) * 100

            st.success(f"Prediction: **{predicted_class}**")
            st.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Prediction scores")
            for label, score in zip(class_names, predictions):
                st.write(f"**{label}** — {score * 100:.2f}%")
                st.progress(float(score))
        except Exception as error:
            st.error(str(error))

st.caption("Model architecture and image size follow the supplied Jupyter Notebook.")
