import streamlit as st
import requests
from PIL import Image
import io

API_URL = "https://<your-railway-api-endpoint>/predict/"  

st.set_page_config(page_title="Fashion Attribute Classifier", layout="centered")
st.title("👕 Fashion Attribute Classifier")
st.markdown("Upload an image and get predictions for **Gender**, **Article Type**, **Base Colour**, and **Season**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Sending to model..."):
            # Prepare file for request
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            try:
                response = requests.post(API_URL, files=files)
                response.raise_for_status()
                prediction = response.json()
                
                st.success("Prediction successful!")
                st.write("### 🧠 Predicted Attributes:")
                st.markdown(f"- **Gender:** {prediction['gender']}")
                st.markdown(f"- **Article Type:** {prediction['articleType']}")
                st.markdown(f"- **Base Colour:** {prediction['baseColour']}")
                st.markdown(f"- **Season:** {prediction['season']}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
