import os
import logging
import json
import base64
import tempfile
import uuid
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

# Upload folder configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'webm', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Initialize OpenAI client for API fallback
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize local Whisper model
# We'll lazy-load the model to save memory until it's needed
whisper_model = None
MODEL_SIZE = "base"  # ~140MB, good compromise between size and quality
model_loaded_time = None

def load_whisper_model():
    """Load the Whisper model if not already loaded"""
    global whisper_model, model_loaded_time
    
    try:
        # If model is not loaded or was loaded more than 15 minutes ago, load/reload it
        if whisper_model is None or model_loaded_time is None or (time.time() - model_loaded_time > 900):
            logging.info(f"Loading local Whisper {MODEL_SIZE} model...")
            import whisper
            whisper_model = whisper.load_model(MODEL_SIZE)
            model_loaded_time = time.time()
            logging.info("Local Whisper model loaded successfully")
        return True
    except Exception as e:
        logging.error(f"Error loading Whisper model: {e}")
        return False

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(audio_path):
    """Transcribe audio using local Whisper model with OpenAI API fallback"""
    # Try local Whisper first
    try:
        # Load the model if not already loaded
        if load_whisper_model():
            logging.info("Transcribing with local Whisper model...")
            result = whisper_model.transcribe(audio_path)
            return result["text"]
    except Exception as e:
        logging.error(f"Error with local Whisper transcription: {e}")
    
    # Fall back to OpenAI API if local transcription fails or model can't be loaded
    if openai_client:
        try:
            logging.info("Falling back to OpenAI API for transcription...")
            with open(audio_path, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return response.text
        except Exception as e:
            logging.error(f"Error transcribing with OpenAI API: {e}")
    
    return "Error transcribing audio. Please try again."

def local_spam_detection(text):
    """
    Perform basic rule-based spam detection as a fallback method
    This is much simpler than ML-based detection but can catch obvious scams
    """
    text = text.lower()
    
    # Common scam indicators and their weights
    scam_indicators = {
        "urgent": 0.4,
        "emergency": 0.4,
        "account.+suspend": 0.6,
        "verify.+identity": 0.5,
        "verify.+account": 0.5,
        "your.+password": 0.6,
        "credit card": 0.5,
        "social security": 0.7,
        "bank.+details": 0.6,
        "lottery": 0.5,
        "won.+prize": 0.5,
        "inheritance": 0.5,
        "prince": 0.4,
        "nigerian": 0.4,
        "transfer.+money": 0.5,
        "offer.+expire": 0.4,
        "limited.+time": 0.4,
        "investment opportunity": 0.6,
        "act now": 0.4,
        "claim your": 0.3,
        "one time offer": 0.4,
        "do not ignore": 0.4,
        "click.+link": 0.5,
        "gift card": 0.5,
        "overdue payment": 0.5,
        "tax refund": 0.5,
        "contact us immediately": 0.5,
        "your account has been": 0.5,
        "security breach": 0.5
    }
    
    # Calculate a spam score based on the presence of indicators
    import re
    spam_score = 0.0
    found_indicators = 0
    
    for pattern, weight in scam_indicators.items():
        if re.search(pattern, text):
            spam_score += weight
            found_indicators += 1
    
    # Normalize the score
    if found_indicators > 0:
        # Increase the score if multiple indicators are found (pattern recognition)
        if found_indicators >= 3:
            spam_score = min(1.0, spam_score * 1.5)
        else:
            spam_score = min(1.0, spam_score)
    
    is_spam = spam_score > 0.6  # Threshold for spam classification
    
    # Format prediction text
    prediction_text = "🚨 Potential Scam Detected!" if is_spam else "✅ Safe"
    
    return {
        "is_spam": is_spam,
        "prediction": prediction_text,
        "confidence": spam_score
    }

def predict_spam(text):
    """Predict if text contains spam/scam content using OpenAI API with local fallback"""
    # Try OpenAI API first if available
    if openai_client:
        try:
            # Use GPT model to analyze the text for potential scam content
            response = openai_client.chat.completions.create(
                model="gpt-4o", # the newest OpenAI model is "gpt-4o" which was released May 13, 2024
                messages=[
                    {
                        "role": "system",
                        "content": "You are a scam detection expert. Analyze the text for potential scam indicators like urgency, requests for personal information, suspicious offers, etc. Respond with a JSON object with these fields: 'is_spam' (boolean), 'confidence' (number between 0 and 1), 'prediction' (string explaining your analysis)."
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Format the prediction text for display
            is_spam = result.get("is_spam", False)
            prediction_text = "🚨 Potential Scam Detected!" if is_spam else "✅ Safe"
            
            return {
                "is_spam": is_spam,
                "prediction": prediction_text,
                "confidence": result.get("confidence", 0.5)
            }
        except Exception as e:
            logging.error(f"Error predicting spam with OpenAI API: {e}")
            # Fall back to local detection
            logging.info("Falling back to local spam detection...")
    
    # If OpenAI client is not available or API call failed, use local detection
    return local_spam_detection(text)

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Process uploaded or recorded audio"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Create a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Transcribe the audio
            transcription = transcribe_audio(filepath)
            
            # Predict if it's spam
            prediction_result = predict_spam(transcription)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify({
                "transcription": transcription,
                "prediction": prediction_result["prediction"],
                "is_spam": prediction_result["is_spam"],
                "confidence": prediction_result["confidence"]
            })
        
        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            
            logging.error(f"Error processing audio: {e}")
            return jsonify({"error": f"Error processing audio: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid file format. Allowed formats: wav, mp3, ogg, flac, webm, m4a"}), 400

@app.route('/analyze-chunk', methods=['POST'])
def analyze_audio_chunk():
    """Process audio chunk for real-time monitoring"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio chunk provided"}), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({"error": "No audio chunk"}), 400
    
    if file:
        # Create a unique filename to avoid collisions
        unique_filename = f"{uuid.uuid4()}_chunk.webm"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        try:
            # Transcribe the audio chunk
            transcription = transcribe_audio(filepath)
            
            # Skip empty transcriptions
            if not transcription or transcription == "Error transcribing audio. Please try again.":
                os.remove(filepath)
                return jsonify({
                    "transcription": "",
                    "is_spam": False,
                    "empty": True
                })
            
            # Predict if it's spam
            prediction_result = predict_spam(transcription)
            
            # Clean up the temporary file
            os.remove(filepath)
            
            return jsonify({
                "transcription": transcription,
                "prediction": prediction_result["prediction"],
                "is_spam": prediction_result["is_spam"],
                "confidence": prediction_result["confidence"],
                "empty": False
            })
        
        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            
            logging.error(f"Error processing audio chunk: {e}")
            return jsonify({"error": f"Error processing audio chunk: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid audio format"}), 400
