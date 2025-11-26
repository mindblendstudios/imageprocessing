import streamlit as st
from skimage import color, io
from PIL import Image
import numpy as np

st.title("🖤 Image Grayscale Converter")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Open the uploaded image using PIL
    pil_image = Image.open(uploaded_file)

    # Convert PIL image to NumPy array
    img_array = np.array(pil_image)

    # Handle RGBA images by removing alpha channel
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]  # Drop the alpha channel

    # Convert to grayscale
    gray = color.rgb2gray(img_array)

    # Show original and grayscale images
    st.subheader("Original Image")
    st.image(pil_image, use_container_width=True)

    st.subheader("Grayscale Image")
    st.image(gray, caption="Grayscale", clamp=True, use_container_width=True)

    # Save to in-memory buffer
    buf = io.BytesIO()
    gray_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    # Add download button
    st.download_button(
        label="📥 Download Grayscale Image",
        data=byte_im,
        file_name="grayscale.png",
        mime="image/png"
    )