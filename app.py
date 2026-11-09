
import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

st.set_page_config(
    page_title="Smart Vision AI",
    page_icon="👁️",
    layout="wide"
)

st.title("Smart Vision AI")
st.subheader("AI-Based Real-Time Object Detection System")

st.write(
    "Upload an image to detect objects using YOLO."
)

@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    results = model.predict(image, conf=0.25)

    detected_image = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original Image")

    with col2:
        st.image(
            detected_image[:, :, ::-1],
            caption="Detected Objects"
        )

    st.success(
        f"Total Objects Detected: {len(results[0].boxes)}"
    )
