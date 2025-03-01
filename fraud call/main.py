import os
import time
import threading
import sys
from speech_capture import SpeechCapture
from spam_detector import SpamDetector
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spam_detection_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RealTimeSpamDetection")

class RealTimeSpamDetection:
    def __init__(self, save_dir="results"):
        """
        Initialize real-time spam detection system
        
        Args:
            save_dir: Directory to save results
        """
        logger.info("Initializing real-time spam detection system")
        
        # Create directories
        self.save_dir = save_dir
        self.transcript_dir = os.path.join(save_dir, "transcripts")
        self.results_dir = os.path.join(save_dir, "spam_results")
        
        for directory in [self.save_dir, self.transcript_dir, self.results_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                
        # Initialize spam detector
        logger.info("Initializing spam detector")
        try:
            self.detector = SpamDetector()
        except Exception as e:
            logger.error(f"Failed to initialize spam detector: {e}")
            sys.exit(1)
            
        # Initialize speech capture
        logger.info("Initializing speech capture")
        self.speech_capture = SpeechCapture(
            callback=self.process_speech,
            save_path=self.transcript_dir
        )
        
        # Create processing thread
        self.processing_queue = []
        self.processing_lock = threading.Lock()
        self.is_running = False
        self.processing_thread = None
        
        logger.info("System initialized")
    
    def process_speech(self, text):
        """
        Process speech text through spam detector
        
        Args:
            text: Transcribed speech text
        """
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text received, skipping")
            return
            
        logger.info(f"Processing text: {text[:50]}..." if len(text) > 50 else f"Processing text: {text}")
        
        # Add to processing queue
        with self.processing_lock:
            self.processing_queue.append(text)
    
    def process_queue(self):
        """Process items in the queue"""
        while self.is_running:
            # Get item from queue
            text = None
            with self.processing_lock:
                if self.processing_queue:
                    text = self.processing_queue.pop(0)
            
            # Process item if available
            if text:
                try:
                    logger.info(f"Detecting spam in: {text[:30]}..." if len(text) > 30 else f"Detecting spam in: {text}")
                    result = self.detector.predict(text)
                    
                    # Log result
                    logger.info(f"Prediction: {result['prediction']} " 
                               f"(Confidence: {result['confidence']:.4f})")
                    
                    # Save result
                    self.detector.save_result(result, self.results_dir)
                    
                    # Print alert for spam
                    if result['is_spam']:
                        logger.warning("⚠️ SPAM DETECTED ⚠️")
                        print("\n" + "=" * 60)
                        print("⚠️ SPAM DETECTED ⚠️")
                        print(f"Text: {text}")
                        print(f"Confidence: {result['confidence']:.4f}")
                        print("=" * 60 + "\n")
                    
                except Exception as e:
                    logger.error(f"Error processing text: {e}")
            
            # Sleep to prevent CPU usage
            time.sleep(0.1)
    
    def start(self):
        """Start the system"""
        if self.is_running:
            logger.warning("System already running")
            return
            
        # Start processing thread
        self.is_running = True
        self.processing_thread = threading.Thread(target=self.process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start speech capture
        self.speech_capture.start()
        
        logger.info("System started")
        print("\n" + "=" * 60)
        print("🎤 Real-time Speech Spam Detection System Started 🔍")
        print("Speak into your microphone to detect potential spam")
        print("Press Ctrl+C to stop")
        print("=" * 60 + "\n")
    
    def stop(self):
        """Stop the system"""
        logger.info("Stopping system")
        
        # Stop speech capture
        self.speech_capture.stop()
        
        # Stop processing thread
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            
        logger.info("System stopped")
        print("\nSystem stopped. Thank you for using the Speech Spam Detector!")


if __name__ == "__main__":
    # Run the system
    system = RealTimeSpamDetection()
    
    try:
        system.start()
        # Keep running until keyboard interrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        system.stop()