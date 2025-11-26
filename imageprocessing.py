import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
from skimage import color, filters
import io

def run_imageprocessing():
    st.title("🎨 Photo Editor")
    st.write("Apply filters to your image and download the result.")

    # Upload image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # Filter options
    filter_option = st.selectbox(
        "Choose a filter to apply:",
        ["Grayscale", "Sepia", "Blur", "Edge Detection (Sobel)"]
    )

    if uploaded_file:
        # Load and convert to RGB to handle all formats
        original_image = Image.open(uploaded_file).convert("RGB")
        result_image = None

        # Convert to NumPy array for some filters
        img_array = np.array(original_image)

        # Apply selected filter
        if filter_option == "Grayscale":
            gray = color.rgb2gray(img_array)
            gray_uint8 = (gray * 255).astype(np.uint8)
            result_image = Image.fromarray(gray_uint8)

        elif filter_option == "Sepia":
            sepia = np.array(original_image).astype(np.float64)
            sepia = sepia @ [[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]]
            sepia = np.clip(sepia, 0, 255).astype(np.uint8)
            result_image = Image.fromarray(sepia)

        elif filter_option == "Blur":
            result_image = original_image.filter(ImageFilter.GaussianBlur(radius=3))

        elif filter_option == "Edge Detection (Sobel)":
            gray = color.rgb2gray(img_array)
            edges = filters.sobel(gray)
            edges_uint8 = (edges * 255).astype(np.uint8)
            result_image = Image.fromarray(edges_uint8)

        # Display images
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

        st.subheader(f"{filter_option} Image")
        st.image(result_image, use_container_width=True)

        # Convert result to bytes
        buf = io.BytesIO()
        result_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Download button
        st.download_button(
            label="📥 Download Edited Image",
            data=byte_im,
            file_name=f"{filter_option.lower().replace(' ', '_')}.png",
            mime="image/png"
        )


