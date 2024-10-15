# f5_tts_model.py

import os
import logging
import torch
import torchaudio
from einops import rearrange
from ema_pytorch import EMA
from vocos import Vocos

from tts_model import TTSModel  # Abstract base class
from f5_tts.model.load_and_setup_model import load_and_setup_model  # Import the loader

from f5_tts.model import CFM, UNetT, DiT, MMDiT
from f5_tts.model.utils import (
    get_tokenizer, 
    convert_char_to_pinyin, 
    save_spectrogram,
)

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
        self.tts_models = {}  # Dictionary to hold multiple models
        self.vocab_char_map_per_speaker = {}
        self.vocab_size_per_speaker = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define default speaker to model mapping if not provided
        if speaker_model_map is None:
            self.speaker_model_map = {
                "mike": "F5TTS_Base",
                #"rachel": "E2TTS_Base"
                "rachel": "F5TTS_Base"
            }
        else:
            self.speaker_model_map = speaker_model_map

        # Set up base model path
        base_model_path = os.getenv("F5_TTS_MODEL_PATH", "./f5_tts/ckpts/")

        # Define model paths for each speaker
        self.model_paths = {
            speaker: os.path.join(base_model_path, f"{model_name}/model_1200000.pt") 
            for speaker, model_name in self.speaker_model_map.items()
        }

        # Check if vocab_char_map_path is provided
        if not vocab_char_map_path:
            logging.error("vocab_char_map_path must be provided for F5TTSModel.")
            raise ValueError("vocab_char_map_path must be provided for F5TTSModel.")
        
        # Load the vocabulary character map using get_tokenizer
        dataset_name = "Emilia_ZH_EN"
        tokenizer = "pinyin"
        self.vocab_char_map, self.vocab_size = get_tokenizer(dataset_name, tokenizer)
        logging.info(f"Initial vocab_char_map size: {self.vocab_size}")

        # Verify and adjust the vocabulary
        self.verify_and_adjust_vocab()

        # Initialize models for each speaker
        for speaker in self.speaker_model_map:
            self.load_model(speaker)

        # Initialize vocoder once
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.vocos.eval()
        logging.info("Vocoder loaded and ready.")

    def verify_and_adjust_vocab(self):
        """
        Verify that the vocabulary meets the expected size and contains necessary tokens.
        Adjust the vocabulary if required.
        """
        expected_vocab_size = 2546  # As per model's expectation
        actual_vocab_size = len(self.vocab_char_map)

        # Check if the space character is present and correctly indexed
        if " " not in self.vocab_char_map:
            logging.warning("Space character ' ' not found in vocab_char_map. Prepending it as [UNK].")
            # Prepend space with index 0 and shift existing indices
            updated_vocab_char_map = {" ": 0}
            for token, idx in self.vocab_char_map.items():
                updated_vocab_char_map[token] = idx + 1
            self.vocab_char_map = updated_vocab_char_map
            self.vocab_size = len(self.vocab_char_map)
            logging.info(f"Prepend space character. New vocab size: {self.vocab_size}")
        else:
            # Ensure space character is at index 0
            if self.vocab_char_map[" "] != 0:
                logging.warning("Space character ' ' is not assigned index 0. Reassigning.")
                # Reassign space to index 0 and adjust others
                updated_vocab_char_map = {" ": 0}
                for token, idx in self.vocab_char_map.items():
                    if token != " ":
                        updated_vocab_char_map[token] = idx
                self.vocab_char_map = updated_vocab_char_map
                logging.info("Space character ' ' reassigned to index 0.")
        
        actual_vocab_size = len(self.vocab_char_map)
        logging.debug(f"Final vocab_char_map: {self.vocab_char_map}")
        logging.debug(f"Final vocab_size: {self.vocab_size}")

    def load_model(self, speaker):
        """
        Load a specific F5-TTS model based on the speaker's experiment name.

        Args:
            speaker (str): The speaker identifier (e.g., "mike", "rachel").
        """
        model_name = self.speaker_model_map[speaker]
        model_path = self.model_paths.get(speaker)
        if not model_path or not os.path.exists(model_path):
            logging.error(f"Model path for speaker '{speaker}' does not exist: {model_path}")
            raise FileNotFoundError(f"Model path for speaker '{speaker}' does not exist: {model_path}")

        logging.info(f"Loading F5-TTS model '{model_name}' for speaker '{speaker}' from path: {model_path}")

        # Use load_and_setup_model to initialize and load the model
        tts_model = load_and_setup_model(
            model_name="F5-TTS",
            checkpoint=model_path,
            exp_name=model_name,
            vocab_char_map=self.vocab_char_map  # Pass the loaded vocab_char_map
        )

        if tts_model is None:
            logging.error(f"Failed to load the F5-TTS model for speaker '{speaker}'.")
            raise ValueError(f"Failed to load the F5-TTS model for speaker '{speaker}'.")

        # Handle EMA if applicable
        use_ema = True  # Set based on your requirements
        if use_ema:
            ema_model = EMA(tts_model, include_online_model=False).to(self.device)
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                if 'ema_model_state_dict' in checkpoint:
                    ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
                    ema_model.copy_params_from_ema_to_model()
                    logging.info(f"EMA model loaded and parameters copied for speaker '{speaker}'.")
                else:
                    logging.warning(f"No EMA state dict found in checkpoint for speaker '{speaker}'. Proceeding without EMA.")
            except Exception as e:
                logging.error(f"Failed to load EMA state dict for speaker '{speaker}': {e}")
                raise e

        # Move model to device
        tts_model.to(self.device)
        tts_model.eval()

        # Store the loaded model
        self.tts_models[speaker] = tts_model
        logging.info(f"F5-TTS model for speaker '{speaker}' loaded successfully.")

    def preprocess_text(self, text: str, speaker: str) -> torch.Tensor:
        """
        Preprocess the input text by converting characters to pinyin or tokens.

        Args:
            text (str): The input text to preprocess.
            speaker (str): The speaker identifier.

        Returns:
            torch.Tensor: The tensor of token indices.
        """
        # Using a common vocab_char_map for all speakers
        vocab_char_map = self.vocab_char_map

        tokenizer = "pinyin"  # Adjust based on your tokenizer
        text_list = [text]
        if tokenizer == "pinyin":
            final_text_list = convert_char_to_pinyin(text_list)
        else:
            final_text_list = text_list

        # Tokenizer mapping
        token_indices = []
        for word in final_text_list:
            tokens = [vocab_char_map.get(char, 0) for char in word]  # 0 as default for unknown chars
            token_indices.append(tokens)

        # Convert to tensor and pad
        token_tensor = torch.tensor(token_indices, dtype=torch.long).to(self.device)  # Shape: (batch, nt)
        logging.debug(f"Processed text tensor shape: {token_tensor.shape}")
        return token_tensor

    def infer(self, text: str, speaker: str, ref_audio_path: str = None):
        """
        Generate audio from text input for a specified speaker.

        Args:
            text (str): The text to generate audio for.
            speaker (str): The speaker identifier (e.g., "mike", "rachel").
            ref_audio_path (str, optional): Path to the reference audio file (required for conditioning).

        Returns:
            np.ndarray: The generated audio waveform as a NumPy array.
        """
        if speaker not in self.tts_models:
            logging.error(f"Speaker '{speaker}' not found. Available speakers: {list(self.tts_models.keys())}")
            raise ValueError(f"Speaker '{speaker}' not found. Available speakers: {list(self.tts_models.keys())}")

        tts_model = self.tts_models[speaker]
        logging.info(f"Generating audio for speaker '{speaker}' with text: '{text[:50]}...'")

        # Load and preprocess reference audio if required
        audio = None
        if ref_audio_path:
            logging.debug(f"Loading reference audio from: {ref_audio_path}")
            if not os.path.exists(ref_audio_path):
                logging.error(f"Reference audio file not found: {ref_audio_path}")
                raise FileNotFoundError(f"Reference audio file not found: {ref_audio_path}")
            audio, sr = torchaudio.load(ref_audio_path)
            target_sample_rate = 24000
            n_mel_channels = 100
            hop_length = 256
            target_rms = 0.1

            # Normalize RMS
            rms = torch.sqrt(torch.mean(torch.square(audio)))
            if rms < target_rms:
                audio = audio * target_rms / rms
                logging.debug(f"Normalized audio RMS from {rms.item()} to {target_rms}")

            # Resample if necessary
            if sr != target_sample_rate:
                logging.debug(f"Resampling audio from {sr} Hz to {target_sample_rate} Hz")
                resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
                audio = resampler(audio)

            audio = audio.to(self.device)
            logging.debug(f"Reference audio loaded and moved to device: {self.device}")
        else:
            logging.warning("No reference audio path provided. Proceeding without conditioning audio.")

        # Preprocess text
        processed_text = self.preprocess_text(text, speaker)
        logging.debug(f"Processed text tensor shape: {processed_text.shape}")

        # Duration calculation
        if ref_audio_path:
            ref_audio_len = audio.shape[-1] // hop_length
            fix_duration = 27  # Adjust or make dynamic if needed

            if fix_duration is not None:
                duration = int(fix_duration * target_sample_rate / hop_length)
                logging.debug(f"Fixed duration set to: {duration}")
            else:
                # Example duration estimation (can be more sophisticated)
                zh_pause_punc = r"。，、；：？！"
                ref_text_len = len(text) + len(re.findall(zh_pause_punc, text))
                duration = ref_audio_len + int(ref_audio_len / ref_text_len * len(text))
                logging.debug(f"Estimated duration based on reference audio and text: {duration}")
        else:
            # Default duration if no reference audio
            duration = 1000  # Adjust based on requirements
            logging.info(f"No reference audio provided. Using default duration: {duration} frames.")

        # Inference
        with torch.inference_mode():
            try:
                generated, trajectory = tts_model.sample(
                    cond=audio,
                    text=processed_text,
                    duration=duration,
                    steps=32,  # Adjust as needed
                    cfg_strength=2.0,  # Adjust as needed
                    sway_sampling_coef=-1.0,  # Adjust as needed
                    seed=None,  # Set seed if needed
                )
                logging.debug(f"Inference completed. Generated mel shape: {generated.shape}")
            except Exception as e:
                logging.error(f"Error during inference: {e}", exc_info=True)
                raise e

        # Post-processing
        if ref_audio_path:
            generated = generated[:, ref_audio_len:, :]
            logging.debug(f"Trimmed generated mel spectrogram to exclude reference audio: {generated.shape}")
        generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
        generated_wave = self.vocos.decode(generated_mel_spec.cpu())
        logging.debug(f"Decoded generated wave shape: {generated_wave.shape}")
        if ref_audio_path and rms < target_rms:
            generated_wave = generated_wave * rms / target_rms
            logging.debug(f"Adjusted wave amplitude based on RMS normalization.")

        # Convert to NumPy array
        generated_wave_np = generated_wave.numpy()

        logging.info(f"Audio generation complete for speaker '{speaker}'.")
        return generated_wave_np
