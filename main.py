import logging
import threading
import time
from app import app, transcriber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preload_whisper_model():
    """
    Preload the Whisper model in a background thread to avoid blocking the app startup.
    """
    logger.info("Starting background thread to preload Whisper model...")
    def _load_model():
        try:
            start_time = time.time()
            logger.info("Preloading Whisper model...")
            result = transcriber.load_model()
            end_time = time.time()
            if result:
                logger.info(f"Whisper model preloaded successfully in {end_time - start_time:.2f} seconds")
            else:
                logger.error("Failed to preload Whisper model")
        except Exception as e:
            logger.error(f"Error preloading Whisper model: {e}")
    
    # Start the model loading in a background thread
    thread = threading.Thread(target=_load_model)
    thread.daemon = True
    thread.start()

if __name__ == "__main__":
    # Preload the Whisper model in the background
    preload_whisper_model()
    
    # Start the Flask application
    app.run(host="0.0.0.0", port=5000, debug=True)
