import argparse
import os
import sys
import torch
import numpy as np
from pocketgroq import GroqProvider, GroqAPIKeyMissingError, GroqAPIError
from bark_model import BarkTTSModel
from f5_tts_model import F5TTSModel
from tts_model import TTSModel
from scipy.io.wavfile import write as write_wav, read as read_wav
from bark import SAMPLE_RATE, generate_audio, preload_models
from bark.generation import generate_text_semantic
from bark.api import semantic_to_waveform
from config import (
    DEFAULT_MODEL,
    MAX_TOKENS,
    HOST_PROFILES,
    OUTLINE_PROMPT_TEMPLATE,
    EXPAND_PROMPT_TEMPLATE,
    DIALOGUE_PROMPT_TEMPLATE
)

# Set environment variables
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # Try GPU first

MALE_VOICE_PRESET = "v2/en_speaker_6"
FEMALE_VOICE_PRESET = "v2/en_speaker_9"

CUSTOM_MALE_VOICE_PATH = "null"
CUSTOM_FEMALE_VOICE_PATH = "bark_voice_samples/female.wav"

# Main GroqCastersApp with TTS model integration
class GroqCastersApp:
    def __init__(self, tts_model: TTSModel):
        self.tts_model = tts_model
        self.groq = GroqProvider()  # Initialize the GroqProvider here

    def load_voice_model(self, model_path):
        self.tts_model.load_model(model_path)

    def generate_audio(self, text: str, temp: float = 0.7, output_dir=""):
        # Check the model type and handle the infer call accordingly
        if isinstance(self.tts_model, F5TTSModel):
            return self.tts_model.infer(text)  # No temperature for F5TTSModel
        else:
            return self.tts_model.infer(text, temperature=temp)  # BarkTTSModel accepts temperature

    # Move script generation methods here
    def generate_podcast_script(self, input_text):
        outline = self._generate_outline(input_text)
        if not outline:
            return None
        full_script = self._expand_outline(outline)
        if not full_script:
            return None
        dialogue_script = self._convert_to_dialogue(full_script)
        return dialogue_script

    def _generate_outline(self, input_text):
        prompt = OUTLINE_PROMPT_TEMPLATE.format(input_text=input_text)
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["outline"])
        except GroqAPIError as e:
            print(f"Error generating outline: {e}")
            return None

    def _expand_outline(self, outline):
        prompt = EXPAND_PROMPT_TEMPLATE.format(outline=outline, host_profiles=HOST_PROFILES)
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["full_script"])
        except GroqAPIError as e:
            print(f"Error expanding outline: {e}")
            return None

    def _convert_to_dialogue(self, full_script):
        prompt = DIALOGUE_PROMPT_TEMPLATE.format(full_script=full_script, host_profiles=HOST_PROFILES)
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["dialogue"])
        except GroqAPIError as e:
            print(f"Error converting to dialogue: {e}")
            return None

# GroqCasters class handling script generation and audio
class GroqCasters:
    def __init__(self):
        try:
            self.groq = GroqProvider()
            self._setup_gpu()
            preload_models()
            self.custom_male_voice = self._create_voice_prompt(CUSTOM_MALE_VOICE_PATH)
            self.custom_female_voice = self._create_voice_prompt(CUSTOM_FEMALE_VOICE_PATH)
        except GroqAPIKeyMissingError:
            print("Error: GROQ_API_KEY not found. Please set it in your environment variables.")
            sys.exit(1)
        except Exception as e:
            print(f"Error during initialization: {e}")
            sys.exit(1)

    def _setup_gpu(self):
        if torch.cuda.is_available():
            print(f"GPU available: {torch.cuda.get_device_name(0)}")
            torch.cuda.set_device(0)
        else:
            print("No GPU available. Using CPU.")
            os.environ["SUNO_OFFLOAD_CPU"] = "True"

    def _create_voice_prompt(self, audio_file_path):
        if audio_file_path == "null" or not os.path.exists(audio_file_path):
            print(f"Warning: Custom voice file not found at {audio_file_path}. Using default voice.")
            return None
        try:
            sample_rate, audio_data = read_wav(audio_file_path)
            if audio_data.dtype != np.int16:
                audio_data = (audio_data * 32767).astype(np.int16)
            audio_data = audio_data.astype(np.float32) / 32767.0
            audio_data = audio_data[:, 0] if len(audio_data.shape) > 1 else audio_data
            if sample_rate != SAMPLE_RATE:
                print(f"Warning: Audio file sample rate ({sample_rate} Hz) does not match required rate ({SAMPLE_RATE} Hz).")
            semantic_tokens = generate_text_semantic(
                "Hello, this is a voice prompt.", 
                history_prompt=audio_data, temp=0.7, 
                min_eos_p=0.05, d_vector=None
            )
            return semantic_tokens
        except Exception as e:
            print(f"Error processing custom voice file: {e}")
            return None

    def generate_audio_from_script(self, script, output_dir):
        lines = script.split("\n")
        audio_segments = []
        for line in lines:
            if line.strip():
                speaker, text = line.split(":", 1)
                speaker = speaker.strip().lower()
                text = text.strip()
                if speaker == "mike":
                    voice = self.custom_male_voice if self.custom_male_voice is not None else MALE_VOICE_PRESET
                else:
                    voice = self.custom_female_voice if self.custom_female_voice is not None else FEMALE_VOICE_PRESET
                try:
                    if isinstance(voice, str):
                        audio_array = generate_audio(text, history_prompt=voice)
                    else:
                        semantic_tokens = generate_text_semantic(
                            text,
                            history_prompt=voice,
                            temp=0.7,
                            min_eos_p=0.05,
                        )
                        audio_array = semantic_to_waveform(semantic_tokens)
                    audio_segments.append(audio_array)
                except Exception as e:
                    print(f"Error generating audio for line: {line}")
                    print(f"Error details: {e}")
        full_audio = np.concatenate(audio_segments)
        output_file = os.path.join(output_dir, "full_podcast.wav")
        write_wav(output_file, SAMPLE_RATE, full_audio)
        print(f"Full podcast audio saved to: {output_file}")

def process_input_text(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except IOError:
        print(f"Error: Unable to read file at {file_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description="GroqCasters Podcast Generation")
    
    # Existing arguments
    parser.add_argument("input_file_path", type=str, help="Path to the input file")
    parser.add_argument("output_directory", type=str, help="Directory to save the output")
    parser.add_argument("--use-script", action="store_true", help="Use a pre-written script")
    
    # New arguments for voice model and temperature
    parser.add_argument("--voice-model", type=str, choices=["f5_tts", "bark"], default="bark",
                        help="Specify the TTS model to use (e.g., 'f5_tts' or 'bark')")
    parser.add_argument("--voice-temperature", type=float, default=0.7,
                        help="Temperature for TTS generation (default: 0.7 for creativity vs. stability)")

    args = parser.parse_args()

    input_file_path = args.input_file_path
    output_directory = args.output_directory
    use_existing_script = args.use_script
    voice_model = args.voice_model
    voice_temperature = args.voice_temperature

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load the correct model based on the user-specified voice model
    if voice_model == "f5_tts":
        tts_model = F5TTSModel()
    else:
        tts_model = BarkTTSModel()  # Default

    # Initialize the GroqCastersApp with the specified TTS model
    casters_app = GroqCastersApp(tts_model)

    if use_existing_script:
        print("Using pre-written script...")
        script = process_input_text(input_file_path)
        if not script:
            print("Failed to read the script file.")
            return
    else:
        print("Generating new podcast script...")
        input_text = process_input_text(input_file_path)
        if not input_text:
            print("Failed to process input text.")
            return
        script = casters_app.generate_podcast_script(input_text)
        if not script:
            print("Failed to generate podcast script.")
            return

    print("Generated/Loaded podcast script:")
    print(script)

    # Use the specified temperature for audio generation
    print(f"Generating audio with voice model '{voice_model}' and temperature {voice_temperature}")
    casters_app.generate_audio(script, voice_temperature, output_directory)
    print("Done!")

if __name__ == "__main__":
    main()

