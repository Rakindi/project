import base64
from io import BytesIO

# Function to decode a base64 string and load it as a PIL image
def load_base64_image(base64_string):
    # Decode the base64 string
    image_data = base64.b64decode(base64_string)
    
    # Convert the binary data to a PIL image
    image = Image.open(BytesIO(image_data)).convert('RGB')
    
    return image

# Update predict_fracture to accept a PIL image
def predict_fracture_from_image(image):
    # Apply transformations
    image = image_transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform the fracture/no-fracture prediction
    with torch.no_grad():
        output = fracture_model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Update preprocess_image_for_keras to accept a PIL image
def preprocess_image_for_keras_from_pil(image):
    # Convert PIL image to numpy array (OpenCV format)
    img = np.array(image)
    
    # Resize the image to (256, 256) for the Keras model
    img = cv2.resize(img, (256, 256))
    
    # Normalize the image
    img = img / 255.0
    
    # Expand dimensions to fit the model input shape
    img = np.expand_dims(img, axis=0)  # Shape: (1, 256, 256, 3)
    
    return img

# Updated classify_fracture_type to accept a PIL image
def classify_fracture_type_from_image(image):
    new_image = preprocess_image_for_keras_from_pil(image)

    # Extract features using Keras MobileNet
    features = feature_extractor.predict(new_image)

    # Reshape features for the Random Forest classifier
    features = features.reshape(features.shape[0], -1)

    # Predict the fracture type
    prediction_index = rf_model.predict(features)[0]
    return class_names[prediction_index]

# Main function updated for base64 input
def main_base64(base64_string):
    # Load the base64 image
    image = load_base64_image(base64_string)

    # Step 1: Predict if fracture or not using PyTorch model
    fracture_prediction = predict_fracture_from_image(image)
    predicted_label = map_prediction_to_label(fracture_prediction)

    # Step 2: If fracture is detected, classify the fracture type
    if predicted_label == "Fracture":
        fracture_type = classify_fracture_type_from_image(image)
        print(f"Detected Fracture Type: {fracture_type}")
    else:
        print("No Fracture Detected.")

# Example Base64 Image Input (Provide your base64 image string here)
base64_image_string = "your_base64_encoded_image_string_here"
main_base64(base64_image_string)
