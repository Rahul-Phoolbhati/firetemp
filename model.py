import cv2  # For image loading
from tensorflow.keras.applications import VGG16  # Replace with a suitable pre-trained model
from tensorflow.keras.layers import Dense  # For regression layer
from tensorflow.keras.models import Model  # For model manipulation
from sklearn.model_selection import train_test_split  # For data splitting
from sklearn.svm import SVR  # Support Vector Regression
import numpy as np

# Load temperature data from txt file
temp_data = np.loadtxt("/content/firenet/temperatures.txt")

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
image_features = []
for fire_image_path in ["/content/firenet/images/1-detected.jpg", "/content/firenet/images/3-detected.jpg"]:  # Replace with actual paths
    features = extract_features(fire_image_path)
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

# Example usage (assuming fire detection identified a fire in "new_image.jpg")
predicted_temp = predict_temperature("1.jpg")
print(f"Predicted temperature for new fire image: {predicted_temp:.2f} degrees Celsius")
