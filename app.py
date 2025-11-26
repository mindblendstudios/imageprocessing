import streamlit as st
from imageprocessing import run_imageprocessing
from bgremover import run_bgremover
from image_identifier2 import run_image_identifier
from imageoperations import run_imageoperations
from ocrtextextraction import run_ocr
from dashboard import run_excel_dashboard
from styleimage import run_style_transfer
from bankstatementanalysis import run_statement_analysis_app

st.set_page_config(page_title="My Image Processing Website", layout="wide")

# Inject navy blue background color
st.markdown(
    """
    <style>
        .stApp {
            background-color: #66788b;  /* Navy blue */
        }

        .stSidebar {
            background-color: #001a35 !important;  /* Slightly darker navy for sidebar */
        }

        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .css-1cpxqw2 {
            color: white; /* Text color for visibility on navy */
        }

        .stDownloadButton button, .stButton button {
            background-color: #0074D9;
            color: white;
            border: none;
        }

        .stDownloadButton button:hover, .stButton button:hover {
            background-color: #005fa3;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Add Logo (top of the page) ---
st.image("iplogo.jpg", width=70)

st.title("My Image Processor")

# Sidebar menu
tool = st.sidebar.selectbox(
    "Choose Options to Process Image:",
    ("Bank Statement Analysis", "Style Image", "Dashboard", "Photo Editor", "Background Remover", "Image Identifier", "Image Operations", "OCR Text Extraction")
)

# Display content based on selection
if tool == "Photo Editor":
    st.header("Photo Editor")
    run_imageprocessing()

elif tool == "Background Remover":
    st.header("Background Remover")
    run_bgremover()

elif tool == "Image Identifier":
    st.header("Image Identifier")
    run_image_identifier()

elif tool == "Image Operations":
    st.header("Image Operations")
    run_imageoperations()

elif tool == "OCR Text Extraction":
    st.header("OCR Text Extraction")
    run_ocr()

elif tool == "Dashboard":
    st.header("Dashboard")
    run_excel_dashboard()

elif tool == "Style Image":
    st.header("Style Image")
    run_style_transfer()

elif tool == "Bank Statement Analysis":
    st.header("Bank Statement Analysis")
    run_statement_analysis_app()

