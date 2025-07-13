import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from io import BytesIO
import requests
import json
import os
import sys
import pickle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.model import FashionModel
from collections import OrderedDict

# Load class mappings
@st.cache_resource
def load_class_mappings():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "..", "model", "class_mappings.json")
    with open(path, 'r') as f:
        return json.load(f)

# Load model
@st.cache_resource
def load_model(device='cpu'):
    class_mappings = load_class_mappings()
    model = FashionModel(
        num_genders=len(class_mappings['gender']),
        num_article_types=len(class_mappings['articleType']),
        num_base_colours=len(class_mappings['baseColour']),
        num_seasons=len(class_mappings['season'])
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "..", "model", "codemonk_model.pth")
    state_dict = torch.load(model_path, map_location=device)
    
    # Remove 'module.' if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, class_mappings

# Transform image
@st.cache_resource
def get_transform():
    # Load the exact transforms used during training
    script_dir = os.path.dirname(os.path.abspath(__file__))
    transform_path = os.path.join(script_dir, "..", "model", "transforms.pkl")
    with open(transform_path, "rb") as f:
        return pickle.load(f)

# Decode predictions
def decode_predictions(outputs, class_mappings):
    gender_out, article_out, base_colour_out, season_out = outputs

    def get_label(output, mapping):
        idx = torch.argmax(output).item()
        inv_map = {v: k for k, v in mapping.items()}
        return inv_map.get(idx, "Unknown")

    return {
        "Gender": get_label(gender_out, class_mappings["gender"]),
        "Article Type": get_label(article_out, class_mappings["articleType"]),
        "Base Colour": get_label(base_colour_out, class_mappings["baseColour"]),
        "Season": get_label(season_out, class_mappings["season"]),
    }

# Main App
st.set_page_config(page_title="Fashion Classifier", layout="centered")
st.title("👗 Fashion Product Classifier")
st.write("Upload an image or provide an image URL to classify:")

model, class_mappings = load_model()
device = 'cpu'
transform = get_transform()

# Upload or URL
option = st.radio("Choose input method:", ("Upload Image", "Image URL"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload a fashion image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
elif option == "Image URL":
    url = st.text_input("Enter image URL")
    if url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except:
            st.error("Failed to load image. Check the URL.")

# Predict
if 'image' in locals():
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("Classifying..."):
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
        predictions = decode_predictions(outputs, class_mappings)

    st.success("Prediction Results:")
    for key, value in predictions.items():
        st.markdown(f"**{key}:** {value}")
