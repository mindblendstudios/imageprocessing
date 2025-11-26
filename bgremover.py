import streamlit as st
from rembg import remove
from PIL import Image
import io

def run_bgremover():
    st.title("🧼 Background Remover")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

    if uploaded_file:
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Image", use_container_width=True)

        # Remove background
        result = remove(input_image)
        st.image(result, caption="Background Removed", use_container_width=True)

        # Convert to bytes and create download button
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="📥 Download Image",
            data=byte_im,
            file_name="no_bg.png",
            mime="image/png"
        )
