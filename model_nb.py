# Music Genre Classifier using GTZAN Dataset from Kaggle
# Run this in Google Colab

import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import kagglehub
import warnings
warnings.filterwarnings('ignore')

# Download GTZAN dataset from Kaggle
print("Downloading GTZAN dataset from Kaggle...")
path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
print(f"Dataset downloaded to: {path}")

# Feature extraction function
def extract_features(file_path, duration=30):
    """
    Extract audio features from a music file
    """
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, duration=duration)

        # Extract features
        # 1. MFCCs (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # 2. Spectral Centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroids)
        spectral_centroid_std = np.std(spectral_centroids)

        # 3. Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        spectral_rolloff_std = np.std(spectral_rolloff)

        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # 5. Chroma Features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # 6. Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        # Combine all features - ensure all are 1D arrays
        features = np.concatenate([
            mfccs_mean.flatten(),
            mfccs_std.flatten(),
            np.array([spectral_centroid_mean, spectral_centroid_std]).flatten(),
            np.array([spectral_rolloff_mean, spectral_rolloff_std]).flatten(),
            np.array([zcr_mean, zcr_std]).flatten(),
            chroma_mean.flatten(),
            chroma_std.flatten(),
            np.array([tempo]).flatten()
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load dataset and extract features
print("\nExtracting features from audio files...")
# Find the Data/genres_original directory
data_path = os.path.join(path, 'Data', 'genres_original')
if not os.path.exists(data_path):
    # Alternative path structure
    data_path = os.path.join(path, 'genres_original')
    if not os.path.exists(data_path):
        # List what's available
        print(f"Contents of {path}:")
        for item in os.listdir(path):
            print(f"  - {item}")
        raise ValueError("Could not find genres_original directory")

print(f"Using data from: {data_path}")
genres_dir = data_path
genres = os.listdir(genres_dir)
genres = [g for g in genres if os.path.isdir(os.path.join(genres_dir, g))]
print(f"Found genres: {genres}")

features_list = []
labels_list = []

for genre in genres:
    genre_path = os.path.join(genres_dir, genre)
    files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]

    print(f"Processing {genre}...")
    for file in files:
        file_path = os.path.join(genre_path, file)
        features = extract_features(file_path)

        if features is not None:
            features_list.append(features)
            labels_list.append(genre)

# Create DataFrame
print("\nCreating dataset...")
X = np.array(features_list)
y = np.array(labels_list)

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {len(y)}")
print(f"Genres: {np.unique(y)}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\n{'='*50}")
print(f"Model Accuracy: {accuracy:.2%}")
print(f"{'='*50}\n")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix - Music Genre Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# Feature importance
feature_names = (
    [f'mfcc_mean_{i}' for i in range(13)] +
    [f'mfcc_std_{i}' for i in range(13)] +
    ['spectral_centroid_mean', 'spectral_centroid_std'] +
    ['spectral_rolloff_mean', 'spectral_rolloff_std'] +
    ['zcr_mean', 'zcr_std'] +
    [f'chroma_mean_{i}' for i in range(12)] +
    [f'chroma_std_{i}' for i in range(12)] +
    ['tempo']
)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Top 20 Most Important Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Function to predict genre of a new audio file
def predict_genre(file_path):
    """Predict the genre of a music file"""
    features = extract_features(file_path)
    if features is not None:
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction = rf_model.predict(features_scaled)
        genre = le.inverse_transform(prediction)[0]

        # Get probabilities
        probabilities = rf_model.predict_proba(features_scaled)[0]
        prob_dict = dict(zip(le.classes_, probabilities))

        print(f"\nPredicted Genre: {genre}")
        print("\nProbabilities:")
        for g, prob in sorted(prob_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"  {g}: {prob:.2%}")

        return genre
    return None

print("\n" + "="*50)
print("Model training complete!")
print("="*50)
print("\nTo predict a genre for a new file, use:")
print("predict_genre('path/to/your/audio.wav')")