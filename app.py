import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from music_genre_classifier import MusicGenreClassifier
import tempfile

# Page config
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

# Helper function to display prediction results
def display_prediction_results(genre, probabilities):
    """Display prediction results with visualizations"""
    st.markdown("---")
    st.header("Prediction Results")
    
    # Main prediction
    st.markdown(f"### 🎸 Predicted Genre: **{genre.upper()}**")
    
    # Probabilities
    st.subheader("Genre Probabilities")
    
    # Sort probabilities
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    # Create DataFrame for display
    prob_df = pd.DataFrame(sorted_probs, columns=['Genre', 'Probability'])
    prob_df['Probability'] = prob_df['Probability'] * 100
    
    # Display as bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4' if g == genre else '#7f7f7f' for g, _ in sorted_probs]
    ax.barh(prob_df['Genre'], prob_df['Probability'], color=colors)
    ax.set_xlabel('Probability (%)', fontsize=12)
    ax.set_title('Genre Classification Probabilities', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    for i, (g, p) in enumerate(sorted_probs):
        ax.text(p * 100 + 1, i, f'{p*100:.1f}%', va='center', fontsize=10)
    
    st.pyplot(fig)
    
    # Display table
    st.dataframe(
        prob_df.style.format({'Probability': '{:.2f}%'})
        .background_gradient(subset=['Probability'], cmap='Blues'),
        use_container_width=True
    )

# Title
st.title("🎵 Music Genre Classifier")
st.markdown("Upload an audio file to predict its genre using machine learning")

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = MusicGenreClassifier()
    st.session_state.model_loaded = False

# Sidebar
with st.sidebar:
    st.header("Model Management")
    
    # Load pre-trained model
    if st.button("Load Pre-trained Model"):
        try:
            st.session_state.classifier.load_model('model.pkl')
            st.session_state.model_loaded = True
            st.success("Model loaded successfully!")
        except FileNotFoundError:
            st.error("No pre-trained model found. Please train a model first.")
        except Exception as e:
            st.error(f"Error loading model: {e}")
    
    st.markdown("---")
    
    # Training section
    with st.expander("Train New Model"):
        st.write("Train a new model using GTZAN dataset")
        dataset_path = st.text_input("Dataset Path", "Data/genres_original")
        
        if st.button("Start Training"):
            if os.path.exists(dataset_path):
                with st.spinner("Training model... This may take a while."):
                    try:
                        X, y = st.session_state.classifier.load_dataset(dataset_path)
                        accuracy, cm = st.session_state.classifier.train(X, y)
                        st.session_state.classifier.save_model('model.pkl')
                        st.session_state.model_loaded = True
                        st.success(f"Training complete! Accuracy: {accuracy:.2%}")
                    except Exception as e:
                        st.error(f"Training error: {e}")
            else:
                st.error("Dataset path not found!")

# Main content
if not st.session_state.model_loaded:
    st.warning("⚠️ Please load a pre-trained model or train a new one from the sidebar.")
    st.info("💡 If you have a trained model, click 'Load Pre-trained Model' in the sidebar.")
else:
    st.success("✅ Model is ready for predictions!")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎤 Live Microphone"])
    
    # Tab 1: File Upload
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload an audio file (.wav format)",
            type=['wav'],
            help="Upload a WAV audio file to classify its genre"
        )
        
        if uploaded_file is not None:
            # Display file info
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.audio(uploaded_file, format='audio/wav')
            
            with col2:
                st.info(f"**Filename:** {uploaded_file.name}")
                st.info(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            
            # Predict button
            if st.button("🎯 Predict Genre", type="primary", key="predict_upload"):
                with st.spinner("Analyzing audio..."):
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        # Make prediction
                        genre, probabilities = st.session_state.classifier.predict(tmp_path)
                        
                        if genre is not None:
                            display_prediction_results(genre, probabilities)
                        else:
                            st.error("Failed to process the audio file.")
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {e}")
                    
                    finally:
                        # Clean up temp file
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
    
    # Tab 2: Live Microphone
    with tab2:
        st.markdown("### 🎤 Record Audio from Microphone")
        st.info("Click the microphone button to start recording. Play music near your microphone, then click stop to predict the genre.")
        
        # Audio recorder
        try:
            from audio_recorder_streamlit import audio_recorder
            
            st.markdown("**Click the microphone button below to start/stop recording:**")
            audio_bytes = audio_recorder(
                text="",
                recording_color="#e74c3c",
                neutral_color="#3498db",
                icon_name="microphone",
                icon_size="3x",
            )
            
            if audio_bytes:
                st.success("✅ Recording captured!")
                st.audio(audio_bytes, format='audio/wav')
                
                if st.button("🎯 Predict Genre from Recording", type="primary", key="predict_mic"):
                    with st.spinner("Analyzing recorded audio..."):
                        # Save recorded audio temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                            tmp_file.write(audio_bytes)
                            tmp_path = tmp_file.name
                        
                        try:
                            # Make prediction
                            genre, probabilities = st.session_state.classifier.predict(tmp_path)
                            
                            if genre is not None:
                                display_prediction_results(genre, probabilities)
                            else:
                                st.error("Failed to process the recorded audio.")
                        
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                        
                        finally:
                            # Clean up temp file
                            if os.path.exists(tmp_path):
                                os.unlink(tmp_path)
        
        except ImportError:
            st.error("⚠️ Audio recording library not installed.")
            st.code("pip install audio-recorder-streamlit", language="bash")
            st.markdown("---")
            st.markdown("**Alternative: Upload a recorded file in the 'Upload Audio File' tab**")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with Streamlit | Music Genre Classification using Random Forest</p>
    </div>
    """,
    unsafe_allow_html=True
)
