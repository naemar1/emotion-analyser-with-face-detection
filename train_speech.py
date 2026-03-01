import os
import pandas as pd
import librosa
import numpy as np
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder

def extract_mfcc(file_path):
    """Extracts MFCC features from an audio file."""
    audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=3, sr=22050, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def load_speech_data(base_path):
    file_paths = []
    labels = []
    
    # 1. Process RAVDESS: Filename '03-01-01-...' (3rd part is emotion)
    rav_path = os.path.join(base_path, 'Ravdess')
    if os.path.exists(rav_path):
        for root, _, files in os.walk(rav_path):
            for file in files:
                if file.endswith('.wav'):
                    parts = file.split('-')
                    # Ravdess 01=neutral, 03=happy, 04=sad, 05=angry, 06=fear, 07=disgust, 08=surprise
                    emotion_map = {'01':'neutral', '03':'happy', '04':'sad', '05':'angry', '06':'fear', '07':'disgust', '08':'surprise'}
                    if parts[2] in emotion_map:
                        file_paths.append(os.path.join(root, file))
                        labels.append(emotion_map[parts[2]])

    # 2. Process CREMA: Filename '1001_DFA_ANG_XX' (3rd part is emotion)
    crema_path = os.path.join(base_path, 'Crema')
    if os.path.exists(crema_path):
        for file in os.listdir(crema_path):
            if file.endswith('.wav'):
                part = file.split('_')[2]
                crema_map = {'ANG':'angry', 'DIS':'disgust', 'FEA':'fear', 'HAP':'happy', 'NEU':'neutral', 'SAD':'sad'}
                if part in crema_map:
                    file_paths.append(os.path.join(crema_path, file))
                    labels.append(crema_map[part])

    # 3. Process TESS: Folder name contains the emotion
    tess_path = os.path.join(base_path, 'Tess')
    if os.path.exists(tess_path):
        for folder in os.listdir(tess_path):
            emotion = folder.split('_')[-1].lower()
            if emotion == 'ps': emotion = 'surprise' # TESS uses 'ps' for pleasant surprise
            folder_full = os.path.join(tess_path, folder)
            for file in os.listdir(folder_full):
                file_paths.append(os.path.join(folder_full, file))
                labels.append(emotion)

    # Extract Features
    print(f"Extracting features from {len(file_paths)} files...")
    features = [extract_mfcc(p) for p in file_paths]
    return np.array(features), np.array(labels)

# --- Main Execution ---
DATASET_PATH = 'dataset/speech'
X, y = load_speech_data(DATASET_PATH)

# Encode labels to numbers (0-6)
lb = LabelEncoder()
y_encoded = lb.fit_transform(y)

# Build Model
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(40,)),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y_encoded, epochs=50, batch_size=32, validation_split=0.2)

model.save('speech_model.h5')
print("Speech model saved!")