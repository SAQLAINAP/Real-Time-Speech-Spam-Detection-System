import whisper
import logging
import os
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_whisper_model(model_size="base"):
    """
    Download and cache the Whisper model of the specified size.
    
    Args:
        model_size (str): The size of the model to download.
                         Options: 'tiny', 'base', 'small', 'medium', 'large'
    
    Returns:
        bool: True if model was downloaded successfully, False otherwise.
    """
    try:
        logger.info(f"Downloading Whisper {model_size} model...")
        start_time = time.time()
        
        # This will download and cache the model
        model = whisper.load_model(model_size)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Whisper {model_size} model downloaded successfully in {duration:.2f} seconds")
        logger.info(f"Model is now cached and ready for use")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading Whisper model: {e}")
        return False

if __name__ == "__main__":
    # Sizes and approximate disk space requirements:
    # - tiny: ~75 MB
    # - base: ~142 MB
    # - small: ~466 MB
    # - medium: ~1.5 GB
    # - large: ~3 GB
    model_size = "base"  # Default to base model (~142 MB)
    
    # Check if model size was provided as an argument
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ["tiny", "base", "small", "medium", "large"]:
        model_size = sys.argv[1]
    
    logger.info(f"Selected model size: {model_size}")
    download_whisper_model(model_size)