import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import io

def run_imageoperations():
    st.set_page_config(page_title="🛠️ Basic Image Operations", layout="centered")
    st.title("🛠️ Basic Image Processing in Python")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        pil_img = Image.open(uploaded_file).convert("RGB")
        st.image(pil_img, caption="Original Image", use_container_width=True)

        st.subheader("Choose operations to apply:")

        # Resize
        resize = st.checkbox("Resize Image")
        if resize:
            width = st.number_input("Width", value=200, min_value=1)
            height = st.number_input("Height", value=200, min_value=1)
            pil_img = pil_img.resize((int(width), int(height)))
            st.image(pil_img, caption="Resized Image")

        # Crop
        crop = st.checkbox("Crop Image")
        if crop:
            left = st.number_input("Left", value=0)
            top = st.number_input("Top", value=0)
            right = st.number_input("Right", value=pil_img.width)
            bottom = st.number_input("Bottom", value=pil_img.height)
            pil_img = pil_img.crop((left, top, right, bottom))
            st.image(pil_img, caption="Cropped Image")

        # Rotate
        rotate = st.checkbox("Rotate Image")
        if rotate:
            angle = st.slider("Rotation Angle", min_value=0, max_value=360, value=45)
            pil_img = pil_img.rotate(angle)
            st.image(pil_img, caption=f"Rotated by {angle}°")

        # Flip
        flip = st.checkbox("Flip Horizontally")
        if flip:
            pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
            st.image(pil_img, caption="Flipped Image")

        # Grayscale
        grayscale = st.checkbox("Convert to Grayscale")
        if grayscale:
            gray_img = pil_img.convert("L")
            st.image(gray_img, caption="Grayscale Image")

        # Draw text
        draw_text = st.checkbox("Add Text")
        if draw_text:
            text = st.text_input("Enter text to draw", "Hello, World!")
            draw_img = pil_img.copy()
            draw = ImageDraw.Draw(draw_img)
            draw.text((10, 10), text, fill="white")
            pil_img = draw_img
            st.image(pil_img, caption="Image with Text")

        # Filters
        filter_option = st.selectbox("Apply Filter", ["None", "Blur", "Sharpen", "Emboss"])
        if filter_option == "Blur":
            pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=3))
            st.image(pil_img, caption="Blurred Image")
        elif filter_option == "Sharpen":
            pil_img = pil_img.filter(ImageFilter.SHARPEN)
            st.image(pil_img, caption="Sharpened Image")
        elif filter_option == "Emboss":
            pil_img = pil_img.filter(ImageFilter.EMBOSS)
            st.image(pil_img, caption="Embossed Image")

        # Histogram Equalization (OpenCV)
        equalize = st.checkbox("Apply Histogram Equalization")
        if equalize:
            cv_img = np.array(pil_img)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            equalized = cv2.equalizeHist(gray)
            st.image(equalized, caption="Histogram Equalized", clamp=True)

        # Download processed image
        st.subheader("Download Processed Image")
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button("📥 Download Image", data=byte_im, file_name="processed_image.png", mime="image/png")
