import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os 

# App setup
st.set_page_config(page_title="Fashion Classifier", layout="centered")
st.title("👕 Fashion-MNIST Image Classifier")

# Class names
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "fashion_cnn_resnet18.pth")

# Load model
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Upload image
uploaded = st.file_uploader("Upload a fashion image (28x28 grayscale or color):", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("L")  # convert to grayscale
    st.image(img, caption="Uploaded Image", width=200)

    # Preprocess
    img_t = transform(img).unsqueeze(0).repeat(1, 3, 1, 1)

    # Predict
    with torch.no_grad():
        outputs = model(img_t)
        _, predicted = torch.max(outputs, 1)
        pred_class = classes[predicted.item()]

    st.success(f"Predicted: {pred_class}")