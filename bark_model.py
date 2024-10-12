from bark import preload_models, generate_text_semantic
from bark.api import semantic_to_waveform
from tts_model import TTSModel

class BarkTTSModel(TTSModel):
    def __init__(self):
        self.model = None
    def load_model(self, model_path: str):
        preload_models()
    def infer(self, text: str) -> bytes:
        semantic_tokens = generate_text_semantic(text)
        audio = semantic_to_waveform(semantic_tokens)
        return audio.numpy().tobytes()
