import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (100, 100))  # Resize for consistency
                    images.append(img.flatten())
                    labels.append(label)
    return np.array(images), np.array(labels)

# Paths
dataset_path = "dataset"  # Replace with your dataset folder path
model_path = "person_recognition_model.pkl"

# Load data
print("Loading dataset...")
X, y = load_images_from_folder(dataset_path)

# Split data
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate model
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save model
print("Saving model...")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")