"""
Script to train the music genre classifier
Run this before using the Streamlit app if you don't have a pre-trained model
"""

import kagglehub
from music_genre_classifier import MusicGenreClassifier
import os

def main():
    # Download GTZAN dataset
    print("Downloading GTZAN dataset from Kaggle...")
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    print(f"Dataset downloaded to: {path}")
    
    # Find data path
    data_path = os.path.join(path, 'Data', 'genres_original')
    if not os.path.exists(data_path):
        data_path = os.path.join(path, 'genres_original')
    
    print(f"Using data from: {data_path}")
    
    # Initialize classifier
    classifier = MusicGenreClassifier()
    
    # Load dataset
    X, y = classifier.load_dataset(data_path)
    
    # Train model
    accuracy, cm = classifier.train(X, y)
    
    # Save model
    classifier.save_model('model.pkl')
    
    print("\n" + "="*50)
    print("Training complete! Model saved as 'model.pkl'")
    print("You can now run the Streamlit app with: streamlit run app.py")
    print("="*50)

if __name__ == "__main__":
    main()
