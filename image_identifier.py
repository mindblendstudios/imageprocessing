import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests

def run_bgremover():

    # Load pre-trained model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval()

    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load labels
    @st.cache_data
    def load_labels():
        labels_url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        return requests.get(labels_url).text.splitlines()

    labels = load_labels()

    # Streamlit UI
    st.title("🖼️ Image Identifier")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image")

        # Preprocess and predict
        img_t = preprocess(image)
        batch_t = torch.unsqueeze(img_t, 0)

        with torch.no_grad():
            out = model(batch_t)
            _, predicted_idx = torch.max(out, 1)
            label = labels[predicted_idx.item()]

        st.success(f"Identified object: **{label}**")
