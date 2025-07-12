# api/inference.py

import torch
from .utils import load_model, preprocess_image, decode_predictions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and class mappings once at startup
model, class_mappings = load_model(device=device)


def predict(image_file):
    image_tensor = preprocess_image(image_file).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
    predictions = decode_predictions(outputs, class_mappings)
    return predictions
