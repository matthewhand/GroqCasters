# groqcasters.py

import argparse
import os
import sys
import torch
import numpy as np
import logging
from pocketgroq import GroqProvider, GroqAPIKeyMissingError, GroqAPIError
from bark_model import BarkTTSModel
from f5_tts_model import F5TTSModel
from tts_model import TTSModel
from scipy.io.wavfile import write as write_wav
from config import (
    DEFAULT_MODEL,
    MAX_TOKENS,
    HOST_PROFILES,
    OUTLINE_PROMPT_TEMPLATE,
    EXPAND_PROMPT_TEMPLATE,
    DIALOGUE_PROMPT_TEMPLATE
)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variables
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # Try GPU first

class GroqCastersApp:
    def __init__(self, tts_model: TTSModel, model_path=None, vocab_char_map=None):
        self.tts_model = tts_model
        logging.debug(f"Initialized GroqCastersApp with TTS model: {self.tts_model}")
        # Always call load_model, even if model_path is None
        self.tts_model.load_model(model_path=model_path)

    def load_voice_model(self, model_path, vocab_char_map=None):
        logging.debug(f"Loading voice model from: {model_path}")
        self.tts_model.load_model(model_path)

    def generate_audio(self, text: str, speaker=None, ref_audio_path: str = None, output_dir: str = None, output_filename: str = "test_single.wav"):
        """
        Generate audio for a single text input.

        Args:
            text (str): The text to generate audio for.
            speaker (str): The speaker identifier.
            ref_audio_path (str): Path to the reference audio file (required for F5-TTS).
            output_dir (str): Directory to save the output audio.
            output_filename (str): Name of the output audio file.

        Returns:
            np.ndarray or None: The generated audio waveform as a numpy array (for Bark) or None (for F5-TTS).
        """
        # Generate audio based on the model type
        logging.info(f"Generating audio with text: '{text[:50]}...'")
        if isinstance(self.tts_model, F5TTSModel):
            if not ref_audio_path or not output_dir:
                logging.error("ref_audio_path and output_dir must be provided for F5-TTS.")
                return None
            logging.debug(f"Using F5TTSModel for inference with speaker: {speaker}")
            # F5TTSModel's infer expects (text, speaker, ref_audio_path, output_dir, output_filename)
            self.tts_model.infer(text, speaker, ref_audio_path, output_dir, output_filename)
            # Assuming F5TTSModel.infer saves the audio directly
            # Return None as audio is saved separately
            return None
        elif isinstance(self.tts_model, BarkTTSModel):
            logging.debug("Using BarkTTSModel for inference.")
            # BarkTTSModel's infer expects preprocessed data
            preprocessed = self.tts_model.preprocess_text(text)
            audio = self.tts_model.infer(preprocessed, speaker=speaker)
            # Return the audio segment for concatenation
            return audio
        else:
            raise NotImplementedError("Unsupported TTS model.")
    
class GroqCasters:
    def __init__(self, voice_model='bark', model_path=None, vocab_char_map_path=None):
        try:
            self.groq = GroqProvider()
            self._setup_gpu()
            # Initialize vocab_char_map to None
            vocab_char_map = None

            # Load vocabulary character map if provided
            if vocab_char_map_path and os.path.exists(vocab_char_map_path):
                vocab_char_map = self._load_vocab_char_map(vocab_char_map_path)
                logging.info(f"Loaded vocabulary character map from: {vocab_char_map_path}")
            elif voice_model == "f5_tts":
                logging.error("vocab_char_map_path must be provided for F5-TTS.")
                sys.exit(1)
            else:
                logging.warning("Vocabulary character map not provided. Proceeding without it.")

            # Initialize the correct TTS model based on the argument
            if voice_model == "f5_tts":
                logging.info("Using F5-TTS Model")
                self.tts_model = F5TTSModel()
            else:
                logging.info("Using Bark Model")
                self.tts_model = BarkTTSModel()
            
            # Automatically load the model
            self.app = GroqCastersApp(self.tts_model, model_path, vocab_char_map)
        except GroqAPIKeyMissingError:
            logging.error("GROQ_API_KEY not found. Please set it in your environment variables.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error during initialization: {e}")
            sys.exit(1)

    def _setup_gpu(self):
        if torch.cuda.is_available():
            logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            torch.cuda.set_device(0)
        else:
            logging.warning("No GPU available. Using CPU.")
            os.environ["SUNO_OFFLOAD_CPU"] = "True"

    def _load_vocab_char_map(self, vocab_char_map_path):
        """
        Load the vocabulary character map from a text file.
        Each line in the file should contain a single token.

        Args:
            vocab_char_map_path (str): Path to the vocabulary character map text file.

        Returns:
            dict: A dictionary mapping tokens to unique indices.
        """
        vocab_char_map = {}
        try:
            with open(vocab_char_map_path, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f, start=1):
                    token = line.strip()
                    if token:  # Skip empty lines
                        vocab_char_map[token] = idx
            return vocab_char_map
        except Exception as e:
            logging.exception(f"Failed to load vocab_char_map from {vocab_char_map_path}")
            raise e

    def generate_podcast_script(self, input_text):
        logging.info(f"Generating podcast script from input: '{input_text[:50]}...'")
        outline = self._generate_outline(input_text)
        if not outline:
            logging.error("Failed to generate outline")
            return None
        full_script = self._expand_outline(outline)
        if not full_script:
            logging.error("Failed to expand outline")
            return None
        dialogue_script = self._convert_to_dialogue(full_script)
        return dialogue_script

    def _generate_outline(self, input_text):
        prompt = OUTLINE_PROMPT_TEMPLATE.format(input_text=input_text)
        logging.debug(f"Outline prompt: {prompt}")
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["outline"])
        except GroqAPIError as e:
            logging.error(f"Error generating outline: {e}")
            return None

    def _expand_outline(self, outline):
        prompt = EXPAND_PROMPT_TEMPLATE.format(outline=outline, host_profiles=HOST_PROFILES)
        logging.debug(f"Expand prompt: {prompt}")
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["full_script"])
        except GroqAPIError as e:
            logging.error(f"Error expanding outline: {e}")
            return None

    def _convert_to_dialogue(self, full_script):
        prompt = DIALOGUE_PROMPT_TEMPLATE.format(full_script=full_script, host_profiles=HOST_PROFILES)
        logging.debug(f"Dialogue prompt: {prompt}")
        try:
            return self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["dialogue"])
        except GroqAPIError as e:
            logging.error(f"Error converting to dialogue: {e}")
            return None

    def generate_audio_from_script(self, script, output_dir, output_filename="full_podcast.wav", ref_audio_path: str = None):
        """
        Generate audio from the podcast script.

        Args:
            script (str): The podcast script in dialogue format.
            output_dir (str): Directory to save the output audio.
            output_filename (str): Name of the output audio file.
            ref_audio_path (str): Path to the reference audio file (required for F5-TTS).

        Returns:
            None
        """
        # Preprocess the script to replace speaker names with "Speaker:"
        #processed_script = preprocess_script(script)
        processed_script = script

        lines = processed_script.split("\n")
        audio_segments = []

        for line in lines:
            if line.strip():
                try:
                    speaker, text = line.split(":", 1)
                    speaker = speaker.strip().lower()  # Normalize the speaker label
                    text = text.strip()

                    # Debug logging
                    logging.debug(f"Processing line - Speaker: {speaker}, Text: '{text[:50]}...'")

                    # Generate audio using the appropriate model (Bark or F5-TTS)
                    audio_segment = self.app.generate_audio(text, speaker=speaker, ref_audio_path=ref_audio_path, output_dir=output_dir, output_filename=output_filename)
                    if audio_segment is not None:
                        audio_segments.append(audio_segment)
                    else:
                        logging.info(f"F5-TTS audio segment for line saved to {output_filename}")
                
                except ValueError as e:
                    logging.error(f"Error processing line: '{line}'. Expected format: 'Speaker: Text'. Details: {e}")
        
        # Concatenate all audio segments (only for Bark)
        if isinstance(self.tts_model, BarkTTSModel):
            if audio_segments:
                full_audio = np.concatenate(audio_segments)
                output_file = os.path.join(output_dir, output_filename)
                write_wav(output_file, 22050, full_audio)
                logging.info(f"Full podcast audio saved to: {output_file}")
            else:
                logging.error("No audio segments were generated.")
        else:
            logging.info("F5-TTS audio segments saved individually.")

def process_input_text(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        return None
    except IOError:
        logging.error(f"Unable to read file at {file_path}")
        return None

def main():
    parser = argparse.ArgumentParser(description="GroqCasters Podcast Generation")
    parser.add_argument("input_file_path", type=str, help="Path to the input file")
    parser.add_argument("output_directory", type=str, help="Directory to save the output")
    parser.add_argument("--use-script", action="store_true", help="Use a pre-written script")
    parser.add_argument("--output-filename", type=str, default="full_podcast.wav", help="Name of the output audio file (default: full_podcast.wav)")
    parser.add_argument("--voice-model", type=str, choices=["bark", "f5_tts"], default="bark", help="Choose between 'bark' and 'f5_tts' voice models")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model (optional)")
    parser.add_argument("--vocab-char-map", type=str, default=None, help="Path to the vocabulary character map text file (required for F5-TTS)")
    parser.add_argument("--ref-audio-path", type=str, default=None, help="Path to the reference audio file (required for F5-TTS)")
    args = parser.parse_args()

    input_file_path = args.input_file_path
    output_directory = args.output_directory
    output_filename = args.output_filename
    use_existing_script = args.use_script
    voice_model = args.voice_model
    model_path = args.model_path
    vocab_char_map_path = args.vocab_char_map
    ref_audio_path = args.ref_audio_path

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    casters = GroqCasters(voice_model, model_path, vocab_char_map_path)

    if use_existing_script:
        logging.info("Using pre-written script...")
        script = process_input_text(input_file_path)
        if not script:
            logging.error("Failed to read the script file.")
            return
    else:
        logging.info("Generating new podcast script...")
        input_text = process_input_text(input_file_path)
        if not input_text:
            logging.error("Failed to process input text.")
            return
        script = casters.generate_podcast_script(input_text)
        if not script:
            logging.error("Failed to generate podcast script.")
            return

    logging.info("Generated/Loaded podcast script:")
    logging.info(script)
    logging.info("Generating audio")
    casters.generate_audio_from_script(script, output_directory, output_filename, ref_audio_path)
    logging.info("Done!")

if __name__ == "__main__":
    main()
