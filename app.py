# Import necessary libraries
import streamlit as st
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load Hugging Face's pre-trained model and processor
@st.cache_resource  # Cache to speed up subsequent loads
def load_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

# Function to convert audio to text using wav2vec2 model
def transcribe_audio(audio_file):
    # Load the processor and model
    processor, model = load_model()

    # Read the audio file with librosa
    audio, sr = librosa.load(audio_file, sr=16000)
    
    # Prepare the audio for the model
    input_values = processor(audio, return_tensors="pt", padding="longest").input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode the prediction to get the transcription
    transcription = processor.batch_decode(predicted_ids)[0]
    
    return transcription

# Streamlit app UI setup
st.title("Speech-to-Text Converter")

# Option to upload an audio file
uploaded_file = st.file_uploader("Upload your audio file (.wav or .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")  # Play the uploaded audio file
    st.write("Transcribing audio...")

    # Transcribe the uploaded audio file
    transcription = transcribe_audio(uploaded_file)
    st.write("Transcription:")
    st.success(transcription)
else:
    st.write("Please upload a .wav or .mp3 audio file for transcription.")

