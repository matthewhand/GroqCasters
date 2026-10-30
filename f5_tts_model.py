# f5_tts_model.py

import os
import logging

import numpy as np
import torch

from tts_model import TTSModel  # Abstract base class
from f5_tts.api import F5TTS


class F5TTSModel(TTSModel):
    def __init__(self, speaker_model_map=None, vocab_char_map_path=None):
        """
        Initialize the F5TTSModel with a mapping of speakers to their respective models.

        Args:
            speaker_model_map (dict, optional):
                A dictionary mapping speaker names to their experiment/model names.
                Example: {"mike": "F5TTS_Base", "rachel": "E2TTS_Base"}
            vocab_char_map_path (str, optional):
                Path to the vocabulary character map text file.
        """
        super().__init__()
        self.tts_models = {}  # Dictionary to hold multiple official F5TTS APIs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Define default speaker to model mapping if not provided
        if speaker_model_map is None:
            self.speaker_model_map = {
                "mike": "F5TTS_Base",
                "rachel": "F5TTS_Base"
            }
        else:
            self.speaker_model_map = speaker_model_map

        if not vocab_char_map_path:
            logging.error("vocab_char_map_path must be provided for F5TTSModel.")
            raise ValueError("vocab_char_map_path must be provided for F5TTSModel.")
        self.vocab_char_map_path = vocab_char_map_path

        # Set up base model path
        base_model_path = os.getenv("F5_TTS_MODEL_PATH", "./f5_tts/ckpts/")

        # Define per-speaker checkpoint paths (local files only, no downloads)
        self.model_paths = {
            speaker: os.path.join(base_model_path, f"{model_name}/model_1200000.pt")
            for speaker, model_name in self.speaker_model_map.items()
        }

        # Require a pre-existing local vocoder directory (config.yaml + pytorch_model.bin).
        # Fail fast BEFORE any F5TTS construction; the official F5TTS constructor would
        # otherwise attempt to load/download the vocoder from HuggingFace by default.
        vocoder_local_path = os.getenv("F5_TTS_VOCODER_PATH")
        if not vocoder_local_path:
            logging.error("F5_TTS_VOCODER_PATH environment variable must be set to a local vocoder directory.")
            raise ValueError(
                "F5_TTS_VOCODER_PATH environment variable must be set to a local "
                "directory containing config.yaml and pytorch_model.bin. "
                "Automatic vocoder downloads are disabled."
            )
        vocoder_config = os.path.join(vocoder_local_path, "config.yaml")
        vocoder_weights = os.path.join(vocoder_local_path, "pytorch_model.bin")
        if not os.path.isdir(vocoder_local_path) or not os.path.isfile(vocoder_config) or not os.path.isfile(vocoder_weights):
            logging.error(
                f"Local vocoder directory required at F5_TTS_VOCODER_PATH: {vocoder_local_path}. "
                "It must contain config.yaml and pytorch_model.bin."
            )
            raise FileNotFoundError(
                f"Local vocoder directory required at F5_TTS_VOCODER_PATH: {vocoder_local_path}. "
                "It must contain config.yaml and pytorch_model.bin. "
                "Automatic downloads are disabled."
            )
        self.vocoder_local_path = vocoder_local_path

        for speaker in self.speaker_model_map:
            self.load_model(speaker)

    def load_model(self, speaker):
        """
        Load the official f5_tts.api.F5TTS model for a speaker from an
        existing local checkpoint. Remote downloads are never requested.

        Args:
            speaker (str): The speaker identifier (e.g. "mike", "rachel").
        """
        model_name = self.speaker_model_map[speaker]
        ckpt_file = self.model_paths.get(speaker)
        if not ckpt_file or not os.path.isfile(ckpt_file):
            logging.error(f"Model checkpoint for speaker '{speaker}' does not exist: {ckpt_file}")
            raise FileNotFoundError(
                f"Local F5-TTS checkpoint required for speaker '{speaker}'. "
                f"Expected file: {ckpt_file}. Automatic downloads are disabled; "
                "place the checkpoint at F5_TTS_MODEL_PATH or the default path."
            )
        if not os.path.isfile(self.vocab_char_map_path):
            logging.error(f"Vocabulary file does not exist: {self.vocab_char_map_path}")
            raise FileNotFoundError(f"Vocabulary file does not exist: {self.vocab_char_map_path}")

        logging.info(f"Loading F5-TTS '{model_name}' for speaker '{speaker}' from: {ckpt_file}")
        tts_model = F5TTS(
            model=model_name,
            ckpt_file=ckpt_file,
            vocab_file=self.vocab_char_map_path,
            vocoder_local_path=self.vocoder_local_path,
            device=self.device,
        )
        self.tts_models[speaker] = tts_model
        logging.info(f"F5-TTS model for speaker '{speaker}' loaded successfully.")

    def preprocess_text(self, text: str, speaker: str = None):
        """
        Passthrough preprocessing: the official F5TTS API performs its own
        text normalization internally, so no manual tokenization is needed.
        """
        return text

    def _ref_text_for(self, speaker: str) -> str:
        """Resolve the per-speaker reference transcript from the environment."""
        env_key = f"F5_TTS_REF_TEXT_{speaker.upper()}"
        ref_text = os.environ.get(env_key)
        if not ref_text:
            raise ValueError(
                f"Reference text for speaker '{speaker}' is required. Set the "
                f"{env_key} environment variable to the transcript of the "
                f"speaker's reference audio. Automatic transcription (ASR/Whisper "
                "downloads) is disabled."
            )
        return ref_text

    def infer(self, text: str, speaker: str, ref_audio_path: str = None) -> np.ndarray:
        """
        Generate audio from text input for a specified speaker.

        Args:
            text (str): The text to generate audio for.
            speaker (str): The speaker identifier (e.g. "mike", "rachel").
            ref_audio_path (str): Path to the reference audio file. Required.

        Returns:
            np.ndarray: The generated audio waveform as a NumPy array.
        """
        if speaker not in self.tts_models:
            logging.error(f"Speaker '{speaker}' not found. Available speakers: {list(self.tts_models.keys())}")
            raise ValueError(
                f"Speaker '{speaker}' not found. Available speakers: {list(self.tts_models.keys())}"
            )

        if ref_audio_path is None:
            logging.error(f"No reference audio path provided for speaker '{speaker}'.")
            raise ValueError(f"Reference audio path is required for speaker '{speaker}'.")
        if not os.path.isfile(ref_audio_path):
            logging.error(f"Reference audio file not found: {ref_audio_path}")
            raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")

        ref_text = self._ref_text_for(speaker)

        tts_model = self.tts_models[speaker]
        logging.info(f"Generating audio for speaker '{speaker}' with text: '{text[:50]}...'")

        try:
            with torch.inference_mode():
                waveform, sample_rate, _spectrogram = tts_model.infer(
                    ref_file=ref_audio_path,
                    ref_text=ref_text,
                    gen_text=text,
                )
        except Exception as e:
            logging.error(f"Error during F5-TTS inference for speaker '{speaker}': {e}", exc_info=True)
            raise RuntimeError(f"Error during F5-TTS inference for speaker '{speaker}': {e}") from e

        logging.info(f"Audio generation complete for speaker '{speaker}'.")
        return np.asarray(waveform)
