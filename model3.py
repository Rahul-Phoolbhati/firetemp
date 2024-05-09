

import cv2  # For image loading
from tensorflow.keras.applications import VGG16  # Replace with a suitable pre-trained model
from tensorflow.keras.layers import Dense  # For regression layer
from tensorflow.keras.models import Model  # For model manipulation
from sklearn.model_selection import train_test_split  # For data splitting
from sklearn.svm import SVR  # Support Vector Regression
import numpy as np


# for conversion 
import tensorflow as tf

# for server
from flask import Flask, request, jsonify

# Load temperature data from txt file
temp_data = np.loadtxt("./temperatures.txt")

# Define function to extract features from fire images (using pre-trained CNN)
def extract_features(image_path):
    img = cv2.imread(image_path)  # Load image
    if img is None:  # Check if image is loaded successfully
        raise ValueError(f"Could not read image: {image_path}")

    # Preprocess the image (resize, normalize) as needed for VGG16
    img = cv2.resize(img, (224, 224))  # Example resize to 224x224
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed

    x = np.expand_dims(img, axis=0)  # Add batch dimension
    features = VGG16(weights="imagenet", include_top=False)(x)  # Extract features

    # Convert TensorFlow tensor to NumPy array before flattening
    features = features.numpy()  # Convert to NumPy array
    return features.flatten()  # Flatten for regression

# Extract features from all fire images (assuming fire detection has marked them)
import os

# Replace 'firenet/images' with the actual folder name containing your images
image_folder = './images'

# Get all image paths within the folder
image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder)
              if filename.endswith(('.jpg', '.png', '.jpeg'))]  # Filter for image extensions

image_features = []
for image_path in image_paths:
  features = extract_features(image_path)  # Assuming you have an 'extract_features' function
  image_features.append(features)

image_features = np.array(image_features)  # Convert to NumPy array

# Train-test split for temperature data
X_train, X_test, y_train, y_test = train_test_split(image_features, temp_data, test_size=0.2)

# Train SVR model (replace with Random Forest if desired)
model = SVR()
model.fit(X_train, y_train)

# Prediction function (example)
def predict_temperature(image_path):
    features = extract_features(image_path)
    predicted_temp = model.predict([features])[0]
    return predicted_temp

# tf.saved_model.save(model, 'fire_model')

# Example usage (assuming fire detection identified a fire in "new_image.jpg")
predicted_temp1 = predict_temperature("firec.jpg")
# print(f"Predicted temperature for new fire image: {predicted_temp1:.2f} degrees Celsius")

from flask_cors import CORS
import base64
# from io import BytesIO

from werkzeug.utils import secure_filename
import uuid

# UPLOAD_FOLDER = 'temp_images'
UPLOAD_FOLDER = './'

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
cors = CORS(app)
# CORS(app, origins=['http://127.0.0.1:5500'], methods=['POST']) 



if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure the upload folder for Flask app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Check if the file has allowed extension
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file extension'}), 400

    # Save the file to the upload folder
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    if detect_fire_from_path(image_path) == False:
        return jsonify({'error': 'No fire detected'}), 400

    print(detect_fire_from_path(image_path))

    

    try:
        predicted_temp = predict_temperature(image_path)  # Pass base64 string
    except Exception as e:
        print(f"Error predicting temperature: {e}")
        return jsonify({'error': 'Failed to predict temperature'}), 500




    return jsonify({'message': 'File uploaded successfully', 'filename': filename, 'temperature': predicted_temp}), 200

# if __name__ == '__main__':
#     app.run(debug=True)















# @app.route('/capture-fire-image', methods=['POST'])
# def capture_fire_image():

#     photo = request.get_json()['photo']

#     print(photo)

#     image_data = base64.b64decode(photo)

#     print(image_data)

#     # print(request.files)
#     # print(request.data)
#     # if 'image' not in request.files:
#     #     return jsonify({'error': 'No image file found'}), 400

#     # image_file = request.files['image']
#     # if image_file.filename == '':
#     #     return jsonify({'error': 'No selected image'}), 400

#     # Generate a unique filename with a secure extension
#     # data = request.get_json()
#     # image_data = base64.b64decode(data['image'])
#     # print(image_data)
#     filename = secure_filename(uuid.uuid4().hex + '.jpg' )
#     image_path = os.path.join(UPLOAD_FOLDER, filename)

#     image_mime_type = image_file.content_type
#     print(f"Received image with MIME type: {image_mime_type}")

#     # print the image_path in string format
#     print(f"Image saved to: {image_path}")

#     # Save the image
#     # image_file.save(image_path)

#     with open(image_path, 'wb') as f:
#         f.write(image_data)

#     try:
#         predicted_temp = predict_temperature(image_path)  # Pass base64 string
#     except Exception as e:
#         print(f"Error predicting temperature: {e}")
#         return jsonify({'error': 'Failed to predict temperature'}), 500

    
#     # Delete the temporary image
#     # os.remove(image_path)

#     # Return predicted temperature
#     return jsonify({'predicted_temperature': predicted_temp}), 200



def detect_fire(image_bytes):
    """
    Detects fire in an image using a combination of color space conversion,
    thresholding, and morphological operations.

    Args:
        image_bytes (bytes): The image data in bytes format.

    Returns:
        bool: True if fire is detected, False otherwise.
    """

    if not image_bytes:  # Check if image data is empty
        print("Error: Empty image data!")
        return False

    try:
        # Fix deprecation warning
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        # ... (rest of the function remains the same)
    except cv2.error as e:
        print(f"Error decoding image: {e}")
        return False

    # Resize image for efficiency (adjust as needed)
    image = cv2.resize(image, (480, 360))

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define ranges for fire color detection (may need adjustment)
    lower_red = np.array([0, 100, 100], dtype=np.uint8)
    upper_red = np.array([10, 255, 255], dtype=np.uint8)
    lower_orange = np.array([15, 100, 100], dtype=np.uint8)
    upper_orange = np.array([25, 255, 255], dtype=np.uint8)

    # Create masks for red and orange hues
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # Combine masks for a broader fire color range
    mask = cv2.bitwise_or(mask_red, mask_orange)

    # Apply Gaussian filtering for noise reduction (optional)
    # kernel_size = (5, 5)
    # mask = cv2.GaussianBlur(mask, kernel_size, 0)

    # Apply morphological operations to remove small noise and enhance fire regions
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any significant contours (potential fire regions) are present
    fire_detected = False
    if len(contours) > 0:
        # Calculate the total area of all contours
        total_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            total_area += area

        # Set a threshold for minimum fire area (adjust as needed)
        min_fire_area = 1000  # Adjust based on image size and fire intensity

        if total_area > min_fire_area:
            fire_detected = True

    return fire_detected


def detect_fire_from_path(image_path):
    """
    Detects fire in an image from a specified path.

    Args:
        image_path (str): The path to the image file on the server.

    Returns:
        bool: True if fire is detected, False otherwise.
    """

    try:
        # Read image from the specified path
        image_bytes = cv2.imencode('.jpg', cv2.imread(image_path))[1].tobytes()

        # Call the fire detection function with the image bytes
        fire_detected = detect_fire(image_bytes)
        return fire_detected
    except Exception as e:
        print(f"Error reading or processing image: {e}")
        return False  # Indicate error






if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

