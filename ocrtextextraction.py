import streamlit as st
import easyocr
from PIL import Image
import numpy as np

def run_ocr():
    st.title("📝 OCR: Extract Text from Image")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        reader = easyocr.Reader(['en'])  # Initialize once; add more languages if needed
        results = reader.readtext(np.array(image))

        st.subheader("Extracted Text:")
        if results:
            for bbox, text, conf in results:
                st.write(f"**{text}** (Confidence: {conf:.2f})")
        else:
            st.write("No text found.")