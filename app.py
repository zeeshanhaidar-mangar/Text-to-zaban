import streamlit as st
from transformers import VitsModel, AutoTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import torch
import scipy.io.wavfile
import numpy as np
import requests
import tempfile
import os
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="TTS & STT App",
    page_icon="🎤",
    layout="centered"
)

@st.cache_resource
def load_tts_model():
    """
    Load the Text-to-Speech model and tokenizer.
    This function is cached to avoid reloading on every interaction.
    """
    try:
        model_name = "facebook/mms-tts-eng"
        model = VitsModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading TTS model: {str(e)}")
        return None, None

@st.cache_resource
def load_stt_model():
    """
    Load the Speech-to-Text model and processor.
    This function is cached to avoid reloading on every interaction.
    """
    try:
        model_name = "openai/whisper-small"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        model.config.forced_decoder_ids = None
        return model, processor
    except Exception as e:
        st.error(f"Error loading STT model: {str(e)}")
        return None, None

def generate_speech(text, model, tokenizer):
    """
    Generate speech audio from text using TTS model.
    
    Args:
        text: Input text string
        model: TTS model
        tokenizer: Tokenizer for TTS
    
    Returns:
        tuple: (audio_array, sample_rate) or (None, None) if error
    """
    try:
        inputs = tokenizer(text, return_tensors="pt")
        
        with torch.no_grad():
            output = model(**inputs).waveform
        
        audio_array = output.squeeze().cpu().numpy()
        sample_rate = model.config.sampling_rate
        
        return audio_array, sample_rate
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None, None

def save_audio_to_bytes(audio_array, sample_rate):
    """
    Save audio array to bytes buffer in WAV format.
    
    Args:
        audio_array: Numpy array of audio samples
        sample_rate: Sample rate of audio
    
    Returns:
        BytesIO: Audio data in WAV format
    """
    audio_buffer = BytesIO()
    scipy.io.wavfile.write(audio_buffer, rate=sample_rate, data=audio_array)
    audio_buffer.seek(0)
    return audio_buffer

def download_audio_from_url(url):
    """
    Download audio file from URL and save to temporary file.
    
    Args:
        url: Audio file URL
    
    Returns:
        str: Path to temporary file or None if error
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Create temporary file
        suffix = os.path.splitext(url)[1] or ".wav"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(response.content)
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        st.error(f"Error downloading audio from URL: {str(e)}")
        return None

def transcribe_audio(audio_path, model, processor):
    """
    Transcribe audio file to text using STT model.
    
    Args:
        audio_path: Path to audio file
        model: STT model
        processor: Audio processor
    
    Returns:
        str: Transcribed text or None if error
    """
    try:
        import librosa
        
        # Load audio file
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
        
        # Process audio
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        
        # Decode transcription
        transcription = processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        return transcription
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return None

def main():
    st.title("🎤 Text-to-Speech & Speech-to-Text App")
    st.write("Convert text to speech and speech to text using AI models!")
    
    # Create tabs
    tab1, tab2 = st.tabs(["🔊 Text to Speech", "📝 Speech to Text"])
    
    # Tab 1: Text to Speech
    with tab1:
        st.header("Text to Speech")
        st.write("Enter text below and generate spoken audio.")
        
        # Load TTS model
        with st.spinner("Loading TTS model..."):
            tts_model, tts_tokenizer = load_tts_model()
        
        if tts_model is not None and tts_tokenizer is not None:
            # Text input
            default_text = "Hello! This is a text to speech demonstration using artificial intelligence."
            text_input = st.text_area(
                "Enter text to convert to speech:",
                value=default_text,
                height=100,
                help="Type or paste the text you want to convert to speech"
            )
            
            # Generate button
            if st.button("🎵 Generate Speech", type="primary", use_container_width=True):
                if text_input.strip():
                    with st.spinner("Generating speech..."):
                        audio_array, sample_rate = generate_speech(
                            text_input,
                            tts_model,
                            tts_tokenizer
                        )
                        
                        if audio_array is not None:
                            # Save audio to buffer
                            audio_buffer = save_audio_to_bytes(audio_array, sample_rate)
                            
                            st.success("Speech generated successfully!")
                            
                            # Display audio player
                            st.audio(audio_buffer, format="audio/wav")
                            
                            # Download button
                            audio_buffer.seek(0)
                            st.download_button(
                                label="⬇️ Download Audio",
                                data=audio_buffer,
                                file_name="generated_speech.wav",
                                mime="audio/wav",
                                use_container_width=True
                            )
                else:
                    st.warning("⚠️ Please enter some text to generate speech.")
    
    # Tab 2: Speech to Text
    with tab2:
        st.header("Speech to Text")
        st.write("Upload an audio file or provide a URL to transcribe.")
        
        # Load STT model
        with st.spinner("Loading STT model..."):
            stt_model, stt_processor = load_stt_model()
        
        if stt_model is not None and stt_processor is not None:
            # Input methods
            input_method = st.radio(
                "Choose input method:",
                ["Upload Audio File", "Audio URL"],
                horizontal=True
            )
            
            audio_path = None
            temp_file_path = None
            
            if input_method == "Upload Audio File":
                uploaded_file = st.file_uploader(
                    "Choose an audio file",
                    type=["mp3", "wav", "m4a", "ogg", "flac"],
                    help="Upload an audio file to transcribe"
                )
                
                if uploaded_file is not None:
                    # Save uploaded file to temporary location
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    audio_path = temp_file.name
                    temp_file_path = temp_file.name
                    
                    st.audio(uploaded_file, format=f"audio/{uploaded_file.type.split('/')[-1]}")
            
            else:  # Audio URL
                audio_url = st.text_input(
                    "Enter audio URL:",
                    placeholder="https://example.com/audio.mp3",
                    help="Paste a direct link to an audio file"
                )
                
                if audio_url:
                    with st.spinner("Downloading audio..."):
                        audio_path = download_audio_from_url(audio_url)
                        temp_file_path = audio_path
                        
                        if audio_path:
                            st.audio(audio_url)
            
            st.write("---")
            
            # Transcribe button
            if st.button("🎯 Transcribe", type="primary", use_container_width=True):
                if audio_path:
                    with st.spinner("Transcribing audio..."):
                        transcription = transcribe_audio(
                            audio_path,
                            stt_model,
                            stt_processor
                        )
                        
                        # Clean up temporary file
                        if temp_file_path and os.path.exists(temp_file_path):
                            try:
                                os.unlink(temp_file_path)
                            except:
                                pass
                        
                        if transcription:
                            st.success("Transcription completed!")
                            st.write("### 📝 Transcribed Text:")
                            st.info(transcription)
                else:
                    st.warning("⚠️ Please upload an audio file or provide a URL.")
    
    # Footer
    st.write("---")
    st.caption("Powered by Hugging Face Transformers | TTS: facebook/mms-tts-eng | STT: openai/whisper-small")

if __name__ == "__main__":
    main()
