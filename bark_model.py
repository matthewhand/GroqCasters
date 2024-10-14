from tts_model import TTSModel
from bark import generate_audio, preload_models
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform
import os
import numpy as np

# Bark-specific voice presets and paths
MALE_VOICE_PRESET = "v2/en_speaker_6"
FEMALE_VOICE_PRESET = "v2/en_speaker_9"

CUSTOM_MALE_VOICE_PATH = "null"
CUSTOM_FEMALE_VOICE_PATH = "bark_voice_samples/female.wav"

class BarkTTSModel(TTSModel):
    def __init__(self):
        preload_models()  # Load Bark model
        self.custom_male_voice = self._create_voice_prompt(CUSTOM_MALE_VOICE_PATH)
        self.custom_female_voice = self._create_voice_prompt(CUSTOM_FEMALE_VOICE_PATH)

    def load_model(self, model_path=None):
        # Bark does not need specific model paths to be loaded here
        pass

    def infer(self, text: str, temperature: float = 0.7) -> np.ndarray:
        if isinstance(self.custom_male_voice, str):
            voice = self.custom_male_voice
        else:
            voice = self.custom_female_voice

        try:
            semantic_tokens = generate_text_semantic(text, history_prompt=voice, temp=temperature)
            audio_array = semantic_to_waveform(semantic_tokens)
            return audio_array
        except Exception as e:
            raise RuntimeError(f"Error during Bark inference: {e}")

    def _create_voice_prompt(self, audio_file_path):
        if audio_file_path == "null" or not os.path.exists(audio_file_path):
            return None
        try:
            sample_rate, audio_data = read_wav(audio_file_path)
            audio_data = audio_data.astype(np.float32) / 32767.0  # Normalize to float32
            if sample_rate != SAMPLE_RATE:
                print(f"Warning: Sample rate mismatch ({sample_rate} vs {SAMPLE_RATE}).")
            semantic_tokens = generate_text_semantic("Hello, this is a voice prompt.", history_prompt=audio_data, temp=0.7)
            return semantic_tokens
        except Exception as e:
            print(f"Error processing voice file {audio_file_path}: {e}")
            return None
