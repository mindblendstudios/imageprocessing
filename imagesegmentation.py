import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

def run_imagesegmentation():
# Load pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Load and preprocess image
img = Image.open("input.jpg").convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(img).unsqueeze(0)

# Get segmentation
with torch.no_grad():
    output = model(input_tensor)["out"][0]
segmentation = output.argmax(0).byte().cpu().numpy()

plt.imshow(segmentation)
plt.title("Segmented Image")
plt.axis("off")
plt.show()
