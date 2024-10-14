# tts_model.py

from abc import ABC, abstractmethod
import torch

class TTSModel(ABC, torch.nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_model(self, model_path=None):
        """Load the TTS model from the specified path."""
        pass

    @abstractmethod
    def preprocess_text(self, text: str) -> torch.Tensor:
        """Preprocess the input text and convert it to tensor indices."""
        pass

    @abstractmethod
    def infer(self, text: str, speaker: str) -> torch.Tensor:
        """Generate audio waveform from text and speaker information."""
        pass
