from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.nn.functional import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load the pretrained ResNet model
model = resnet18(pretrained=True)
model.eval()
model = torch.nn.Sequential(*list(model.children())[:-1])

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Helper functions
def extract_resnet_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(image).squeeze()
    return features

def extract_color_features(image_path):
    image = Image.open(image_path).convert("RGB")
    hist_r = np.histogram(image.getdata(0), bins=256, range=(0, 256), density=True)[0]
    hist_g = np.histogram(image.getdata(1), bins=256, range=(0, 256), density=True)[0]
    hist_b = np.histogram(image.getdata(2), bins=256, range=(0, 256), density=True)[0]
    return np.concatenate([hist_r, hist_g, hist_b])

def normalize_vector(vec):
    return vec / np.linalg.norm(vec)

@app.route('/api/run_model', methods=['POST'])
def run_model():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Both images are required."}), 400

    # Save images
    image1_path = "image1.jfif"
    image2_path = "image2.jfif"
    request.files['image1'].save(image1_path)
    request.files['image2'].save(image2_path)

    # Extract features
    features1_resnet = extract_resnet_features(image1_path)
    features2_resnet = extract_resnet_features(image2_path)

    features1_color = normalize_vector(extract_color_features(image1_path))
    features2_color = normalize_vector(extract_color_features(image2_path))

    # Compute similarities
    similarity_resnet = cosine_similarity(features1_resnet.unsqueeze(0), features2_resnet.unsqueeze(0)).item()
    similarity_color = np.dot(features1_color, features2_color)

    # Determine result
    resnet_threshold = 0.75
    color_threshold = 0.90
    if similarity_resnet > resnet_threshold and similarity_color > color_threshold:
        result = "Return Processed successfully[same products]"
    elif similarity_resnet > resnet_threshold and similarity_color <= color_threshold:
        result = "Return Refused[Products are same but different colours]"
    else:
        result = "Return Refused [Different Products]"

    # Cleanup
    os.remove(image1_path)
    os.remove(image2_path)

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run
