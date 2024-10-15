# groqcasters.py

import shlex
import argparse
import os
import sys
import torch
import numpy as np
import logging
import re
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

import torchaudio  # Added import for audio processing

# Ensure LOG_LEVEL is fetched and used, default to INFO if not provided
log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

# Apply the log level configuration globally
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),  # This dynamically sets the log level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# Test logging
logging.debug("This is a DEBUG message.")
logging.info("This is an INFO message.")
logging.error("This is an ERROR message.")

# Set environment variables
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
os.environ["SUNO_OFFLOAD_CPU"] = "False"  # Try GPU first

class GroqCastersApp:
    def __init__(self, tts_model: TTSModel, speaker_ref_map=None):
        """
        Initialize the GroqCastersApp with the TTS model and speaker-reference audio mapping.

        Args:
            tts_model (TTSModel): The TTS model instance (F5TTSModel or BarkTTSModel).
            speaker_ref_map (dict, optional):
                A dictionary mapping speaker names to their reference audio paths.
                Example: {"mike": "path/to/zuckerborg.wav", "rachel": "path/to/scarjo.wav"}
        """
        self.tts_model = tts_model
        self.speaker_ref_map = speaker_ref_map if speaker_ref_map else {}
        logging.debug(f"Initialized GroqCastersApp with TTS model: {self.tts_model} and speaker_ref_map: {self.speaker_ref_map}")

    def generate_audio(self, text: str, speaker: str):
        """
        Generate audio for a single text input.

        Args:
            text (str): The text to generate audio for.
            speaker (str): The speaker identifier (e.g., "mike", "rachel").

        Returns:
            np.ndarray or None: The generated audio waveform as a NumPy array (for F5-TTS) or None (for Bark).
        """
        logging.info(f"Generating audio for speaker '{speaker}' with text: '{text[:50]}...'")
        try:
            if isinstance(self.tts_model, F5TTSModel):
                # Retrieve reference audio path for the speaker
                ref_audio_path = self.speaker_ref_map.get(speaker)
                if not ref_audio_path:
                    logging.warning(f"No reference audio provided for speaker '{speaker}'. Using default.")
                # Generate audio waveform
                audio_waveform = self.tts_model.infer(
                    text=text,
                    speaker=speaker,
                    ref_audio_path=ref_audio_path
                )
                return audio_waveform
            elif isinstance(self.tts_model, BarkTTSModel):
                # Bark handles labels internally
                audio_waveform = self.tts_model.infer(text)
                return audio_waveform
            else:
                logging.error("Unsupported TTS model.")
                return None
        except Exception as e:
            logging.error(f"Error during audio generation: {e}", exc_info=True)
            return None

class GroqCasters:
    def __init__(self, voice_model='bark', model_path=None, vocab_char_map_path=None, ref_audio_paths=None):
        """
        Initialize the GroqCasters application.

        Args:
            voice_model (str): The voice model to use ('bark' or 'f5_tts').
            model_path (str, optional): Path to the model (optional).
            vocab_char_map_path (str, optional): Path to the vocabulary character map text file (required for F5-TTS).
            ref_audio_paths (str, optional): Comma-separated list of reference audio paths for each speaker.
        """
        try:
            self.groq = GroqProvider()
            self._setup_gpu()
            # Initialize speaker-reference mapping
            speaker_ref_map = {}

            if voice_model == "f5_tts":
                if not vocab_char_map_path:
                    logging.error("vocab_char_map_path must be provided for F5-TTS.")
                    sys.exit(1)
                # Parse reference audio paths
                if ref_audio_paths:
                    ref_paths = [path.strip() for path in ref_audio_paths.split(',')]
                else:
                    # Default reference audios
                    ref_paths = [
                        "f5_tts/tests/ref_audio/zuckerborg.wav",  # Mike
                        "f5_tts/tests/ref_audio/scarjo.wav"       # Rachel
                    ]
                # Define speakers in the same order as ref_paths
                speakers = ["mike", "rachel"]
                if len(ref_paths) != len(speakers):
                    logging.error(f"Number of reference audio paths ({len(ref_paths)}) does not match number of speakers ({len(speakers)}).")
                    sys.exit(1)
                # Convert reference audios to mono and map
                speaker_ref_map = self.parse_ref_audio_paths(ref_paths, speakers)
                logging.info(f"Speaker to Reference Audio Mapping: {speaker_ref_map}")

                # Initialize F5-TTS Model with speaker to model mapping and vocab_char_map_path
                speaker_model_map = {
                    "mike": "F5TTS_Base",
                    "rachel": "E2TTS_Base"
                }
                self.tts_model = F5TTSModel(
                    speaker_model_map=speaker_model_map, 
                    vocab_char_map_path=vocab_char_map_path
                )
            else:
                # Initialize Bark Model
                self.tts_model = BarkTTSModel()
                logging.info("Initialized BarkTTSModel.")

            # Initialize the application with the TTS model and speaker-reference mapping
            self.app = GroqCastersApp(self.tts_model, speaker_ref_map)
            logging.debug(f"GroqCastersApp initialized with TTS model: {self.tts_model} and speaker_ref_map: {speaker_ref_map}")

        except GroqAPIKeyMissingError:
            logging.error("GROQ_API_KEY not found. Please set it in your environment variables.")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error during initialization: {e}", exc_info=True)
            sys.exit(1)

    def _setup_gpu(self):
        if torch.cuda.is_available():
            logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            torch.cuda.set_device(0)
        else:
            logging.warning("No GPU available. Using CPU.")
            os.environ["SUNO_OFFLOAD_CPU"] = "True"

    def parse_ref_audio_paths(self, ref_audio_paths_str: str, speakers: list):
        """
        Parse a comma-separated string of reference audio paths and map them to speakers.

        Args:
            ref_audio_paths_str (str): Comma-separated reference audio paths.
            speakers (list): List of speaker identifiers.

        Returns:
            dict: Mapping from speaker to their reference audio path.
        """
        # No need to split again if ref_audio_paths_str is already a list
        if isinstance(ref_audio_paths_str, str):
            ref_audio_paths = [path.strip() for path in ref_audio_paths_str.split(",")]
        else:
            ref_audio_paths = ref_audio_paths_str  # Already a list

        if len(ref_audio_paths) != len(speakers):
            logging.error(f"Number of reference audio paths ({len(ref_audio_paths)}) does not match number of speakers ({len(speakers)}).")
            raise ValueError(f"Number of reference audio paths ({len(ref_audio_paths)}) does not match number of speakers ({len(speakers)}).")

        ref_audio_map = {}
        for speaker, ref_path in zip(speakers, ref_audio_paths):
            if not os.path.exists(ref_path):
                logging.error(f"Reference audio path for speaker '{speaker}' does not exist: {ref_path}")
                raise FileNotFoundError(f"Reference audio path for speaker '{speaker}' does not exist: {ref_path}")
            ref_audio_map[speaker] = ref_path

        logging.debug(f"Parsed reference audio paths: {ref_audio_map}")
        return ref_audio_map

    def generate_podcast_script(self, input_text, max_lines=None):
        """
        Generate the podcast script by interacting with the GroqProvider.

        Args:
            input_text (str): The input text to generate the podcast script from.
            max_lines (int, optional): Maximum number of lines to generate per speaker.

        Returns:
            str or None: The generated dialogue script or None if generation fails.
        """
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

        if max_lines is not None:
            # Split the script into lines and truncate to max_lines
            lines = dialogue_script.strip().split('\n')
            truncated_lines = lines[:max_lines]
            dialogue_script = '\n'.join(truncated_lines)
            logging.info(f"Truncated script to {max_lines} lines.")

        # Enforce speaker alternation
        speakers = list(self.tts_model.speaker_model_map.keys())
        dialogue_script = self.enforce_speaker_alternation(dialogue_script, speakers)
        logging.info("Enforced speaker alternation.")

        return dialogue_script

    def enforce_speaker_alternation(self, dialogue_script, speakers):
        """
        Ensure that the dialogue alternates between the specified speakers.

        Args:
            dialogue_script (str): The generated dialogue script.
            speakers (list): List of speaker identifiers (e.g., ["mike", "rachel"]).

        Returns:
            str: The dialogue script with enforced speaker alternation.
        """
        lines = dialogue_script.strip().split('\n')
        alternated_lines = []
        last_speaker = None

        for line in lines:
            if not line.strip():
                continue  # Skip empty lines
            try:
                speaker_label, text = line.split(":", 1)
                speaker = speaker_label.strip().lower()

                if speaker not in speakers:
                    logging.warning(f"Speaker '{speaker}' not recognized. Skipping line.")
                    continue

                if speaker == last_speaker:
                    # Switch to the other speaker
                    speaker = speakers[1] if speakers[0] == speaker else speakers[0]
                    line = f"{speaker.capitalize()}: {text.strip()}"
                    logging.debug(f"Switched speaker to '{speaker}' to enforce alternation.")

                alternated_lines.append(line)
                last_speaker = speaker

            except ValueError as e:
                logging.error(f"Error processing line for alternation: '{line}'. Expected format: 'Speaker: Text'. Details: {e}")

        alternated_script = '\n'.join(alternated_lines)
        return alternated_script

    def _generate_outline(self, input_text):
        prompt = OUTLINE_PROMPT_TEMPLATE.format(input_text=input_text)
        logging.debug(f"Outline prompt: {prompt}")
        try:
            response = self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["outline"])
            logging.debug(f"Outline response: {response}")
            return response
        except GroqAPIError as e:
            logging.error(f"Error generating outline: {e}", exc_info=True)
            return None

    def _expand_outline(self, outline):
        prompt = EXPAND_PROMPT_TEMPLATE.format(outline=outline, host_profiles=HOST_PROFILES)
        logging.debug(f"Expand prompt: {prompt}")
        try:
            response = self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["full_script"])
            logging.debug(f"Full script response: {response}")
            return response
        except GroqAPIError as e:
            logging.error(f"Error expanding outline: {e}", exc_info=True)
            return None

    def _convert_to_dialogue(self, full_script):
        prompt = DIALOGUE_PROMPT_TEMPLATE.format(full_script=full_script, host_profiles=HOST_PROFILES)
        logging.debug(f"Dialogue prompt: {prompt}")
        try:
            response = self.groq.generate(prompt, model=DEFAULT_MODEL, max_tokens=MAX_TOKENS["dialogue"])
            logging.debug(f"Dialogue script response: {response}")
            return response
        except GroqAPIError as e:
            logging.error(f"Error converting to dialogue: {e}", exc_info=True)
            return None

    def generate_audio_from_script(self, script, output_dir, output_filename="full_podcast.wav"):
        """
        Generate audio from the podcast script.

        Args:
            script (str): The podcast script in dialogue format.
            output_dir (str): Directory to save the output audio.
            output_filename (str): Name of the output audio file.

        Returns:
            None
        """
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Created output directory: {output_dir}")

        # Preprocess the script
        processed_script = script

        lines = processed_script.split("\n")
        audio_segments = []

        for idx, line in enumerate(lines):
            if line.strip():
                try:
                    speaker_label, text = line.split(":", 1)
                    speaker = speaker_label.strip().lower()  # Normalize the speaker label
                    text = text.strip()

                    # Debug logging
                    logging.debug(f"Processing line {idx+1} - Speaker: '{speaker}', Text: '{text[:50]}...'")

                    # Generate audio using the appropriate model (F5-TTS or Bark)
                    audio_segment = self.app.generate_audio(
                        text=text,
                        speaker=speaker
                    )
                    if audio_segment is not None:
                        audio_segments.append(audio_segment)
                        logging.info(f"Generated audio segment for line {idx+1}.")
                    else:
                        logging.info(f"Audio segment for line {idx+1} was not generated.")

                except ValueError as e:
                    logging.error(f"Error processing line {idx+1}: '{line}'. Expected format: 'Speaker: Text'. Details: {e}")

        # Concatenate all audio segments and save as a single file
        if isinstance(self.tts_model, F5TTSModel) or isinstance(self.tts_model, BarkTTSModel):
            if audio_segments:
                # Determine sample rate based on model's vocoder (assuming 24000 Hz)
                sample_rate = 24000  # Adjust if different

                # Define output file path
                output_file = os.path.join(output_dir, output_filename)  # Ensure output_file is defined here

                if len(audio_segments) == 1:
                    # Single segment: Transpose and process directly
                    full_audio = audio_segments[0].T  # Shape: (n_samples, channels)
                    full_audio = full_audio.astype(np.float32)
                    full_audio = np.clip(full_audio, -1.0, 1.0)
                    logging.debug(f"Single audio segment shape after transpose: {full_audio.shape}, dtype: {full_audio.dtype}")
                    logging.debug(f"Single audio segment max value: {full_audio.max()}, min value: {full_audio.min()}")
                else:
                    # Multiple segments: Pad and concatenate
                    max_length = max(segment.shape[1] for segment in audio_segments)
                    padded_segments = []
                    for segment in audio_segments:
                        pad_length = max_length - segment.shape[1]
                        if pad_length > 0:
                            silence = np.zeros((segment.shape[0], pad_length), dtype=segment.dtype)
                            padded_segment = np.concatenate((segment, silence), axis=1)
                        else:
                            padded_segment = segment
                        padded_segments.append(padded_segment)

                    # Concatenate along the time axis (axis=1)
                    full_audio = np.concatenate(padded_segments, axis=1)  # Shape: (channels, n_samples)

                    # Transpose to (n_samples, channels)
                    full_audio = full_audio.T  # Shape: (n_samples, channels)

                    # Ensure data type is float32
                    full_audio = full_audio.astype(np.float32)

                    # Clip values to [-1.0, 1.0] to prevent clipping issues
                    full_audio = np.clip(full_audio, -1.0, 1.0)

                    # Debugging information
                    logging.debug(f"Full audio shape after transpose: {full_audio.shape}, dtype: {full_audio.dtype}")
                    logging.debug(f"Full audio max value: {full_audio.max()}, min value: {full_audio.min()}")

                # Write WAV file
                write_wav(output_file, sample_rate, full_audio)
                logging.info(f"Full podcast audio saved to: {output_file}")
            else:
                logging.error("No audio segments were generated.")
        else:
            logging.warning("Unsupported TTS model type for concatenation.")

def process_input_text(file_path):
    try:
        with open(file_path, "r", encoding='utf-8') as file:
            content = file.read().strip()
            logging.debug(f"Input text loaded from {file_path}: {content[:50]}...")
            return content
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
    parser.add_argument("--ref-audio-paths", type=str, default=None, help="Comma-separated list of reference audio paths for each speaker (required for F5-TTS)")
    # New Argument Added Here
    parser.add_argument("--max-lines", type=int, default=None, help="Maximum number of lines to generate for each voice (default: all)")
    args = parser.parse_args()

    input_file_path = args.input_file_path
    output_directory = args.output_directory
    output_filename = args.output_filename
    use_existing_script = args.use_script
    voice_model = args.voice_model
    model_path = args.model_path
    vocab_char_map_path = args.vocab_char_map
    ref_audio_paths_str = args.ref_audio_paths
    max_lines = args.max_lines  # New Argument Extracted Here

    logging.debug(f"Parsed arguments: {args}")

    # Initialize GroqCasters
    casters = GroqCasters(
        voice_model=voice_model, 
        model_path=model_path, 
        vocab_char_map_path=vocab_char_map_path, 
        ref_audio_paths=ref_audio_paths_str
    )

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
        # Pass max_lines to generate_podcast_script
        script = casters.generate_podcast_script(input_text, max_lines=max_lines)
        if not script:
            logging.error("Failed to generate podcast script.")
            return

    logging.info("Generated/Loaded podcast script:")
    # Log the first five lines of the script
    for line in script.split('\n')[:5]:
        logging.info(line)

    # If using F5-TTS, parse reference audio paths
    if voice_model == "f5_tts":
        # Extract speakers from the speaker_model_map
        speakers = list(casters.tts_model.speaker_model_map.keys())
        if ref_audio_paths_str:
            try:
                ref_audio_map = casters.parse_ref_audio_paths(ref_audio_paths_str, speakers)
                logging.info(f"Reference audio paths mapped to speakers: {ref_audio_map}")
                # Update speaker_ref_map in GroqCastersApp
                casters.app.speaker_ref_map = ref_audio_map
            except Exception as e:
                logging.error(f"Error parsing reference audio paths: {e}", exc_info=True)
                sys.exit(1)
        else:
            # Default reference audios are already set in GroqCasters during initialization
            ref_audio_map = casters.app.speaker_ref_map
            logging.info(f"Using default reference audios: {ref_audio_map}")
    else:
        ref_audio_map = None

    logging.info("Generating audio")
    casters.generate_audio_from_script(
        script=script, 
        output_dir=output_directory, 
        output_filename=output_filename
    )
    logging.info("Done!")

if __name__ == "__main__":
    main()
