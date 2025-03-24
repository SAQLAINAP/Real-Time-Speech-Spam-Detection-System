import os
import logging
import time
import whisper
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WhisperTranscriber:
    """
    A class for handling audio transcription using OpenAI's Whisper model locally.
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the Whisper transcriber with a specific model size.
        
        Args:
            model_size (str): The size of the Whisper model to use. 
                              Options: 'tiny', 'base', 'small', 'medium', 'large'
        """
        self.model_size = model_size
        self.model = None
        self.last_loaded_time = None
    
    def load_model(self) -> bool:
        """
        Load the Whisper model if not already loaded or if it was loaded too long ago.
        
        Returns:
            bool: True if the model was loaded successfully, False otherwise.
        """
        try:
            # If model is not loaded or was loaded more than 15 minutes ago, load/reload it
            if self.model is None or self.last_loaded_time is None or (time.time() - self.last_loaded_time > 900):
                logger.info(f"Loading Whisper {self.model_size} model...")
                self.model = whisper.load_model(self.model_size)
                self.last_loaded_time = time.time()
                logger.info("Whisper model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            return False
    
    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio file using the loaded Whisper model.
        
        Args:
            audio_path (str): Path to the audio file to transcribe.
            
        Returns:
            Optional[str]: The transcribed text, or None if transcription failed.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file does not exist: {audio_path}")
            return None
        
        try:
            if not self.load_model():
                return None
            
            logger.info(f"Transcribing audio file: {audio_path}")
            result = self.model.transcribe(audio_path)
            return result["text"]
        
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Information about the model.
        """
        return {
            "model_size": self.model_size,
            "is_loaded": self.model is not None,
            "last_loaded": self.last_loaded_time
        }