import os
import logging
import json
import base64
import tempfile
import uuid
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI

# Import our utility classes
from utils.transcriber import WhisperTranscriber
from utils.rule_based_detector import RuleBasedScamDetector

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

# Initialize the transcriber and scam detector
transcriber = WhisperTranscriber(model_size="base")  # ~140MB model
scam_detector = RuleBasedScamDetector()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def transcribe_audio(audio_path):
    """
    Transcribe audio using local Whisper model with OpenAI API fallback
    """
    # Try local Whisper first
    try:
        logger.info("Attempting local transcription...")
        transcription = transcriber.transcribe(audio_path)
        if transcription:
            logger.info("Local transcription successful")
            return transcription
    except Exception as e:
        logger.error(f"Error with local transcription: {e}")
    
    # Fall back to OpenAI API if local transcription fails
    if openai_client:
        try:
            logger.info("Falling back to OpenAI API for transcription...")
            with open(audio_path, "rb") as audio_file:
                response = openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
            return response.text
        except Exception as e:
            logger.error(f"Error with OpenAI API transcription: {e}")
    
    return "Error transcribing audio. Please try again."

def predict_spam(text):
    """
    Predict if text contains spam/scam content using OpenAI API with local fallback
    """
    # Try OpenAI API first if available
    if openai_client:
        try:
            logger.info("Analyzing text with OpenAI API...")
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
            logger.error(f"Error analyzing with OpenAI API: {e}")
            logger.info("Falling back to rule-based detection...")
    
    # If OpenAI client is not available or API call failed, use rule-based detection
    result = scam_detector.detect(text)
    return {
        "is_spam": result["is_spam"],
        "prediction": result["prediction"],
        "confidence": result["confidence"]
    }

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
