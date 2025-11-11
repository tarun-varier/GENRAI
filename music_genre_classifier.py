import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')


class MusicGenreClassifier:
    """Music Genre Classifier using audio features"""
    
    def __init__(self, n_estimators=200, max_depth=20, random_state=42):
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
        self.is_trained = False
        
    def extract_features(self, file_path, duration=30):
        """Extract audio features from a music file"""
        try:
            audio, sr = librosa.load(file_path, duration=duration)
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)
            mfccs_std = np.std(mfccs, axis=1)
            
            # Spectral Centroid
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroids)
            spectral_centroid_std = np.std(spectral_centroids)
            
            # Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            spectral_rolloff_mean = np.mean(spectral_rolloff)
            spectral_rolloff_std = np.std(spectral_rolloff)
            
            # Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            
            # Chroma Features
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            
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
    
    def load_dataset(self, data_path):
        """Load and extract features from dataset"""
        print("Extracting features from audio files...")
        
        genres = [g for g in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, g))]
        print(f"Found genres: {genres}")
        
        features_list = []
        labels_list = []
        
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            
            print(f"Processing {genre}...")
            for file in files:
                file_path = os.path.join(genre_path, file)
                features = self.extract_features(file_path)
                
                if features is not None:
                    features_list.append(features)
                    labels_list.append(genre)
        
        X = np.array(features_list)
        y = np.array(labels_list)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Number of samples: {len(y)}")
        
        return X, y
    
    def train(self, X, y, test_size=0.2):
        """Train the classifier"""
        print("Training model...")
        
        # Encode labels
        y_encoded = self.le.fit_transform(y)
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.rf_model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.2%}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.le.classes_))
        
        return accuracy, confusion_matrix(y_test, y_pred)
    
    def predict(self, file_path):
        """Predict genre of a music file"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        features = self.extract_features(file_path)
        if features is None:
            return None, None
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.rf_model.predict(features_scaled)
        genre = self.le.inverse_transform(prediction)[0]
        
        # Get probabilities
        probabilities = self.rf_model.predict_proba(features_scaled)[0]
        prob_dict = dict(zip(self.le.classes_, probabilities))
        
        return genre, prob_dict
    
    def save_model(self, filepath='model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet!")
        
        model_data = {
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'le': self.le
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='model.pkl'):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.rf_model = model_data['rf_model']
        self.scaler = model_data['scaler']
        self.le = model_data['le']
        self.is_trained = True
        print(f"Model loaded from {filepath}")
