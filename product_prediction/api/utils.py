import torch
from torchvision import transforms
from PIL import Image
import json
import os
from model.model import FashionModel


# Define transformation consistent with training
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # Match training input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# Load class mappings from JSON file
def load_class_mappings(path="model/class_mappings.json"):
    with open(path, 'r') as f:
        return json.load(f)


# Load trained model with weights
def load_model(weights_path="model/codemonk_model.pth", mappings_path="model/class_mappings.json", device='cpu'):
    # Load class mappings to determine output sizes
    class_mappings = load_class_mappings(mappings_path)
    model = FashionModel(
        num_genders=len(class_mappings['gender']),
        num_article_types=len(class_mappings['articleType']),
        num_base_colours=len(class_mappings['baseColour']),
        num_seasons=len(class_mappings['season'])
    )
    state_dict = torch.load(weights_path, map_location=device)

    # Handle "module." prefix if model was trained with DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model, class_mappings


# Convert model output to class names
def decode_predictions(outputs, class_mappings):
    gender_out, article_out, base_colour_out, season_out = outputs

    def get_class(output, mapping):
        idx = torch.argmax(output, dim=1).item()
        inv_map = {v: k for k, v in mapping.items()}
        return inv_map.get(idx, "Unknown")

    return {
        "gender": get_class(gender_out, class_mappings["gender"]),
        "articleType": get_class(article_out, class_mappings["articleType"]),
        "baseColour": get_class(base_colour_out, class_mappings["baseColour"]),
        "season": get_class(season_out, class_mappings["season"])
    }


# Preprocess uploaded image to tensor
def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB")
    transform = get_transform()
    return transform(image).unsqueeze(0)  # Add batch dimension
