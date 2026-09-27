import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

# Load class names
with open("weather_labels.json", "r") as f:
    class_names = json.load(f)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model architecture if needed
from torchvision.models import efficientnet_b3
def load_model():
    model = efficientnet_b3(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image transforms
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("🌦️ Weather Image Classification")
st.write("Upload an image, and the model will predict the weather condition.")

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict button
    if st.button("Predict Weather"):
        with st.spinner("Analyzing..."):
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[str(predicted.item())]

        st.success(f"☁️ Predicted Weather: **{prediction.capitalize()}**")
