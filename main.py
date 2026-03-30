import logging
import threading
import time
from app import app, transcriber, monitoring_transcriber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preload_whisper_models():
    """
    Preload both Whisper models (base + tiny) in background threads so they're
    ready before the first request arrives.
    """
    def _load(t, label):
        try:
            start = time.time()
            logger.info(f"Preloading Whisper {label} model...")
            if t.load_model():
                logger.info(f"Whisper {label} loaded in {time.time() - start:.1f}s")
            else:
                logger.error(f"Failed to load Whisper {label} model")
        except Exception as e:
            logger.error(f"Error loading Whisper {label}: {e}")

    for model, label in [(transcriber, "base"), (monitoring_transcriber, "tiny")]:
        t = threading.Thread(target=_load, args=(model, label), daemon=True)
        t.start()


if __name__ == "__main__":
    preload_whisper_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
