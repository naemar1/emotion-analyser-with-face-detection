import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
import numpy as np

# 1. Preprocessing Data
def load_facial_data(data_path):
    images, labels = [], []
    # These must match your folder names exactly as seen in your VS Code sidebar
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    for idx, label in enumerate(classes):
        path = os.path.join(data_path, label)
        if not os.path.exists(path):
            print(f"Warning: Path {path} not found. Skipping...")
            continue
            
        print(f"Loading images for: {label}")
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (48, 48))
                    images.append(img)
                    labels.append(idx)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
                
    return np.array(images).reshape(-1, 48, 48, 1) / 255.0, np.array(labels)

# Use the specific path from your error log
TRAIN_DIR = 'dataset/facial/train'
X_train, y_train = load_facial_data(TRAIN_DIR)

# 2. Build CNN Model (Designed for 48x48 Grayscale)
model = models.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D((2, 2)),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Helps prevent overfitting with manual datasets
    layers.Dense(7, activation='softmax') # Changed to 7 for your 7 categories
])

# 3. Compile and Train
model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

print("\nStarting training...")
# Adjust epochs based on how much data you have (try 30-50 for manual data)
model.fit(X_train, y_train, epochs=30, batch_size=32)

# 4. Save the model
model.save('facial_model.h5')
print("\nModel saved as facial_model.h5")