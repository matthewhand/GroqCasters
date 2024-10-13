class TTSModel:
    """
    Base class for Text-to-Speech models. 
    Both Bark and F5-TTS models will inherit from this class.
    """

    def __init__(self):
        raise NotImplementedError("TTSModel is an abstract base class and cannot be instantiated directly.")
    
    def generate_audio(self, text):
        """
        Method to generate audio from text.
        Each TTS model should implement this method.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def preload_models(self):
        """
        Preload model files and assets if required.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

