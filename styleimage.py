# style_transfer.py

import streamlit as st
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
import copy

# --- Set page config at the very top ---
st.set_page_config(page_title="🎨 Neural Style Transfer", layout="centered")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

# Modified to support resizing to the same target size
def image_loader(image_file, target_size=None):
    image = Image.open(image_file).convert('RGB')
    if target_size:
        image = image.resize(target_size, Image.LANCZOS)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            name = str(layer)

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], ContentLoss) or isinstance(model[j], StyleLoss):
            break
    model = model[:(j + 1)]
    return model, style_losses, content_losses

def run_transfer(content_file, style_file, num_steps=300):
    # Resize style image to match content image dimensions
    content_img_pil = Image.open(content_file).convert("RGB")
    target_size = content_img_pil.size  # (width, height)

    content_img = image_loader(content_file, target_size=target_size)
    style_img = image_loader(style_file, target_size=target_size)

    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )

    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    run = [0]

    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_score * 1e6 + content_score
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    output_image = input_img.cpu().clone().squeeze(0)
    return transforms.ToPILImage()(output_image)

# ✅ This function is called from app.py
def run_style_transfer():
    st.title("🎨 Neural Style Transfer - Upload Your Own Images")

    content_file = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
    style_file = st.file_uploader("Upload Style Image", type=["jpg", "jpeg", "png"])

    if content_file and style_file:
        st.image([Image.open(content_file), Image.open(style_file)],
                 caption=["Content Image", "Style Image"], width=300)

        if st.button("✨ Run Style Transfer"):
            with st.spinner("Running style transfer..."):
                output_image = run_transfer(content_file, style_file)

            st.image(output_image, caption="🖼️ Styled Image", use_container_width=True)

            buf = io.BytesIO()
            output_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                "📥 Download Styled Image",
                data=byte_im,
                file_name="styled_image.png",
                mime="image/png"
            )

