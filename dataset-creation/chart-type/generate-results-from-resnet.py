import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import json
import os
from tqdm import tqdm

# Constants
MODEL_PATH = './best_resnet_model.pth'
NUM_CLASSES = 4  # Adjust based on actual classes
CLASS_NAMES = ['h_bar', 'line', 'pie', 'v_bar']  # Adjust as per trained model
IMG_DIR = '../../data/unichart-pretrain/UniChart-pretrain-images/content/images'

# Model loading function
def load_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and predict chart type for an image
def predict_chart_type(model, img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]

# Read and update JSON data
def process_json_data(json_file, model):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    for item in tqdm(data, desc="Processing items"):
        img_id = item['img_id']
        img_path = os.path.join(IMG_DIR, f"{img_id}.png")
        if os.path.exists(img_path):
            chart_type = predict_chart_type(model, img_path)
            item['chart_type'] = chart_type
        else:
            item['chart_type'] = 'Unknown'

    return data

# Save updated JSON file
def save_updated_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

# Main execution logic
def main():
    model = load_model(MODEL_PATH, NUM_CLASSES)
    
    json_files = [
        '../../data/unichart-pretrain/filtered_unichart_pretrain_datatable_train.json',
        '../../data/unichart-pretrain/filtered_unichart_pretrain_datatable_val.json',
        '../../data/unichart-pretrain/filtered_unichart_pretrain_datatable_test.json'
    ]
    
    for json_file in json_files:
        data = process_json_data(json_file, model)
        save_updated_json(data, json_file.replace('filtered', 'updated'))

if __name__ == "__main__":
    main()