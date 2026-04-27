import numpy as np
import cv2  # OpenCV for image loading and processing
import joblib
import tensorflow as tf
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import base64
from io import BytesIO
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Define the PyTorch model to check for fracture or no fracture
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# Load the PyTorch model for fracture detection
fracture_model = MobileNetModel(num_classes=2)
fracture_model.load_state_dict(torch.load("mobilenet.pt", map_location=device))
fracture_model = fracture_model.to(device)
fracture_model.eval()

# Image transformations for PyTorch model
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 2: Load the Keras MobileNet model and Random Forest model for fracture type classification
feature_extractor = tf.keras.models.load_model('mobilenet_feature_extractor.h5')
rf_model = joblib.load('random_forest_classifier.pkl')

# Fracture type class names
class_names = [
    'Avulsion fracture',
    'Comminuted fracture',
    'Fracture Dislocation',
    'Greenstick fracture',
    'Hairline Fracture',
    'Impacted fracture',
    'Longitudinal fracture',
    'Oblique fracture',
    'Pathological fracture',
    'Spiral Fracture'
]

# Helper function to map the prediction to a label
def map_prediction_to_label(prediction):
    label_mapping = {0: "No Fracture", 1: "Fracture"}
    return label_mapping.get(prediction, "Unknown")

# Step 3: Define function for PyTorch fracture prediction
def predict_fracture(image):
    # Perform the fracture/no-fracture prediction
    image = image_transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = fracture_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Step 4: Preprocess the image for Keras MobileNet feature extraction
def preprocess_image_for_keras(image):
    # Resize the image to (256, 256) for the Keras model
    img = cv2.resize(image, (256, 256))
    
    # Normalize the image
    img = img / 255.0
    
    # Expand dimensions to fit the model input shape
    img = np.expand_dims(img, axis=0)  # Shape: (1, 256, 256, 3)
    
    return img

# Step 5: Extract features and classify fracture type
def classify_fracture_type(image):
    new_image = preprocess_image_for_keras(image)

    # Extract features using Keras MobileNet
    features = feature_extractor.predict(new_image)

    # Reshape features for the Random Forest classifier
    features = features.reshape(features.shape[0], -1)

    # Predict the fracture type
    prediction_index = rf_model.predict(features)[0]
    return class_names[prediction_index]

# Function to decode base64 image and process it
def decode_base64_image(base64_str):
    # Decode the base64 string
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return img

# Main function to run the complete prediction pipeline
def main(base64_image):
    if base64_image.startswith('data:image'):
            base64_image+=  base64_image.split(',')[1]
    # Step 1: Decode base64 image
    image = decode_base64_image(base64_image)
    
    # Step 2: Predict if fracture or not using PyTorch model
    fracture_prediction = predict_fracture(image)
    predicted_label = map_prediction_to_label(fracture_prediction)

    # Step 3: If fracture is detected, classify the fracture type
    if predicted_label == "Fracture":
        fracture_type = classify_fracture_type(np.array(image))
        print(f"Detected Fracture Type: {fracture_type}")
    else:
        print("No Fracture Detected.")

# Example base64 image string (replace this with your base64 string)
base64_image_string = "your_base64_encoded_image_here"

# Run the pipeline
