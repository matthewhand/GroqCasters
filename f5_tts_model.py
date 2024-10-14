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
    def __init__(self, exp_name="F5TTS_Base"):
        super().__init__()
        self.tts_model = None
        base_model_path = os.getenv("F5_TTS_MODEL_PATH", "./f5_tts/ckpts/")
        self.model_paths = {
            "default": os.path.join(base_model_path, "F5TTS_Base/model_1200000.pt"),
            "alternate": os.path.join(base_model_path, "E2TTS_Base/model_1200000.pt")
        }
        self.exp_name = exp_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.vocab_char_map = None  # Initialize the vocab_char_map as None
        tokenizer = "pinyin"
        dataset_name = "Emilia_ZH_EN"
        self.vocab_char_map, self.vocab_size = get_tokenizer(dataset_name, tokenizer)

    def load_model(self, model_path=None):
        if model_path:
            if model_path in self.model_paths:
                model_path = self.model_paths[model_path]
            logging.info(f"Loading F5-TTS model from provided path: {model_path}")
        else:
            model_path = self.model_paths["default"]
            logging.info(f"Loading F5-TTS model from default path: {model_path}")

        if self.vocab_char_map is None:
            raise ValueError("vocab_char_map must be provided to load the model.")

        # Use load_and_setup_model to initialize and load the model
        self.tts_model = load_and_setup_model(
            model_name="F5-TTS",
            checkpoint=model_path,
            exp_name=self.exp_name,
            vocab_char_map=self.vocab_char_map
        )

        if self.tts_model is None:
            raise ValueError("Failed to load the F5-TTS model.")

        logging.info("F5-TTS model loaded successfully.")

    def preprocess_text(self, text: str) -> torch.Tensor:
        """
        Preprocess the input text by converting characters to pinyin or tokens.

        Args:
            text (str): The input text to preprocess.

        Returns:
            torch.Tensor: The tensor of token indices.
        """
        if self.vocab_char_map is None:
            raise ValueError("Vocabulary character map is not set. Provide it during model loading.")

        # Convert to pinyin if tokenizer is pinyin
        tokenizer = "pinyin"  # Adjust based on your tokenizer
        text_list = [text]
        if tokenizer == "pinyin":
            final_text_list = convert_char_to_pinyin(text_list)
        else:
            final_text_list = [text_list]
        
        # Tokenizer mapping
        token_indices = []
        for word in final_text_list:
            tokens = [self.vocab_char_map.get(char, 0) for char in word]  # 0 as default for unknown chars
            token_indices.append(tokens)
        
        # Convert to tensor and pad
        token_tensor = torch.tensor(token_indices, dtype=torch.long).to(self.device)  # Shape: (batch, nt)
        return token_tensor

    def infer(self, text: str, speaker: str, ref_audio_path: str, output_dir: str, output_filename="test_single.wav"):
        """
        Generate audio from text input.

        Args:
            text (str): The text to generate audio for.
            speaker (str): The speaker identifier.
            ref_audio_path (str): Path to the reference audio file.
            output_dir (str): Directory to save the output audio.
            output_filename (str): Name of the output audio file.

        Returns:
            None
        """
        if not self.tts_model:
            raise ValueError("TTS model not loaded. Call 'load_model' before inference.")

        # Load and preprocess reference audio
        audio, sr = torchaudio.load(ref_audio_path)
        target_sample_rate = 24000
        n_mel_channels = 100
        hop_length = 256
        target_rms = 0.1

        # Normalize RMS
        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms

        # Resample if necessary
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)

        audio = audio.to(self.device)

        # Preprocess text
        processed_text = self.preprocess_text(text)

        # Duration calculation
        ref_audio_len = audio.shape[-1] // hop_length
        fix_duration = 27  # Adjust or make dynamic if needed

        if fix_duration is not None:
            duration = int(fix_duration * target_sample_rate / hop_length)
        else:
            # Example duration estimation (can be more sophisticated)
            ref_text_len = len(text)  # Adjust based on actual text length calculation
            gen_text_len = len(text)  # Adjust based on actual generation logic
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len)

        # Inference
        with torch.inference_mode():
            generated, trajectory = self.tts_model.sample(
                cond=audio,
                text=processed_text,
                duration=duration,
                steps=32,  # Adjust as needed
                cfg_strength=2.0,  # Adjust as needed
                sway_sampling_coef=-1.0,  # Adjust as needed
                seed=None,  # Set seed if needed
            )
        logging.debug(f"Generated mel: {generated.shape}")

        # Post-processing
        generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, '1 n d -> 1 d n')
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # Save spectrogram and audio
        save_spectrogram(generated_mel_spec[0].cpu().numpy(), f"{output_dir}/test_single.png")
        torchaudio.save(f"{output_dir}/{output_filename}", generated_wave, target_sample_rate)
        logging.info(f"Generated wav: {generated_wave.shape}")
