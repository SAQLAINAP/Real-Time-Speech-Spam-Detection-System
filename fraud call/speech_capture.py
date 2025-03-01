import pocketsphinx
import queue
import threading
import time
import os
import sys

class SpeechCapture:
    def __init__(self, callback=None, save_path=None):
        """
        Initialize speech capture system with Windows API
        
        Args:
            callback: Function to call with transcribed text
            save_path: Path to save transcripts
        """
        self.recognizer = pocketsphinx.Decoder()
        self.microphone = None  # Update this to use pocketsphinx microphone handling
        self.callback = callback
        self.save_path = save_path
        self.transcript_queue = queue.Queue()
        self.is_running = False
        self.thread = None
        
        # Create save directory if it doesn't exist
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path)
    
    def save_transcript(self, text):
        """Save transcript to file"""
        if not self.save_path:
            return
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.save_path, f"transcript_{timestamp}.txt")
        
        with open(filename, "w") as f:
            f.write(text)
        
        print(f"Transcript saved to {filename}")
        return filename
        
    def process_audio(self):
        """Process audio from microphone and convert to text"""
        # Update this to use pocketsphinx microphone handling
        while self.is_running:
            try:
                # Capture audio and recognize speech using pocketsphinx
                audio = None  # Update this to capture audio using pocketsphinx
                text = self.recognizer.decode(audio)
                print(f"Recognized: {text}")
                
                # Put in queue for processing
                self.transcript_queue.put(text)
                
                # Save transcript
                self.save_transcript(text)
                
                # Call callback if provided
                if self.callback:
                    self.callback(text)
                    
            except Exception as e:
                print(f"Error capturing audio: {e}")
                time.sleep(1)
    
    def start(self):
        """Start speech capture in background thread"""
        if self.is_running:
            print("Speech capture already running")
            return
            
        self.is_running = True
        self.thread = threading.Thread(target=self.process_audio)
        self.thread.daemon = True
        self.thread.start()
        print("Speech capture started")
    
    def stop(self):
        """Stop speech capture"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2)
        print("Speech capture stopped")
    
    def get_transcript(self, block=False, timeout=None):
        """Get latest transcript from queue"""
        try:
            return self.transcript_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None


if __name__ == "__main__":
    # Demo usage
    def print_callback(text):
        print(f"Callback received: {text}")
    
    capture = SpeechCapture(
        callback=print_callback,
        save_path="transcripts"
    )
    
    try:
        capture.start()
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        capture.stop()
        print("Stopped speech capture")