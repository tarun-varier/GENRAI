# Music Genre Classifier

A machine learning application that classifies music genres using audio features extracted from WAV files.

## Features

- Extract audio features (MFCCs, spectral features, chroma, tempo)
- Train Random Forest classifier on GTZAN dataset
- Interactive Streamlit web interface
- Real-time genre prediction with probability scores
- Live microphone recording for instant genre classification
- Upload audio files or record directly from your microphone

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model (First Time)

```bash
python train_model.py
```

This will:
- Download the GTZAN dataset from Kaggle
- Extract features from audio files
- Train the Random Forest classifier
- Save the model as `model.pkl`

### 2. Run the Streamlit App

```bash
streamlit run app.py
```

Then:
1. Click "Load Pre-trained Model" in the sidebar
2. Choose your input method:
   - **Upload Audio File**: Upload a WAV file from your computer
   - **Live Microphone**: Record audio directly from your microphone
3. Click "Predict Genre" to see results with probability scores

## Project Structure

- `music_genre_classifier.py` - Main classifier class
- `train_model.py` - Script to train the model
- `app.py` - Streamlit web interface
- `model_nb.py` - Original notebook code
- `requirements.txt` - Python dependencies

## Supported Genres

The GTZAN dataset includes 10 genres:
- Blues
- Classical
- Country
- Disco
- Hip-hop
- Jazz
- Metal
- Pop
- Reggae
- Rock
