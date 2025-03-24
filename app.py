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
from utils.hotword_detector import HotwordDetector
from utils.hotwords_data import hotwords_severity

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

# Initialize the transcriber and scam detectors
transcriber = WhisperTranscriber(model_size="base")  # ~140MB model
rule_based_detector = RuleBasedScamDetector()
hotword_detector = HotwordDetector(hotwords_severity)

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
    Predict if text contains spam/scam content using OpenAI API with enhanced local fallback
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
                        "content": "You are a scam detection expert. Analyze the text for potential scam indicators like urgency, requests for personal information, suspicious offers, etc. Respond with a JSON object with these fields: 'is_spam' (boolean), 'confidence' (number between 0 and 1), 'category' (one of: 'safe', 'neutral', 'suspicious', 'highly_suspicious'), 'prediction' (string explaining your analysis), 'severity' (number from 0-10 with 10 being most severe)."
                    },
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Format the prediction text based on category
            is_spam = result.get("is_spam", False)
            category = result.get("category", "safe")
            severity = result.get("severity", 0)
            
            if category == "highly_suspicious":
                prediction_text = "🚨 HIGH RISK: Potential Scam Detected!"
            elif category == "suspicious":
                prediction_text = "⚠️ SUSPICIOUS: Possible Scam Detected"
            elif category == "neutral":
                prediction_text = "🔍 NEUTRAL: Some Unusual Elements"
            else:
                prediction_text = "✅ SAFE: No Scam Detected"
            
            return {
                "is_spam": is_spam,
                "prediction": prediction_text,
                "confidence": result.get("confidence", 0.5),
                "category": category,
                "severity": severity
            }
        except Exception as e:
            logger.error(f"Error analyzing with OpenAI API: {e}")
            logger.info("Falling back to local detection methods...")
    
    # First try hotword-based detection (more precise)
    logger.info("Using hotword-based detection...")
    hotword_result = hotword_detector.detect(text)
    
    # If hotword detection found potential scam or has high confidence, use it
    if hotword_result["is_spam"] or hotword_result["confidence"] > 0.3:
        # Format prediction text based on category
        category = hotword_result["category"]
        severity = hotword_result["severity"]
        
        if category == "highly_suspicious":
            prediction_text = "🚨 HIGH RISK: Potential Scam Detected!"
        elif category == "suspicious":
            prediction_text = "⚠️ SUSPICIOUS: Possible Scam Detected"
        elif category == "neutral":
            prediction_text = "🔍 NEUTRAL: Some Unusual Elements"
        else:
            prediction_text = "✅ SAFE: No Scam Detected"
        
        return {
            "is_spam": hotword_result["is_spam"],
            "prediction": prediction_text,
            "confidence": hotword_result["confidence"],
            "category": category,
            "severity": severity,
            "matches": hotword_result["matches"],
            "detection_method": "hotword"
        }
    
    # Fall back to rule-based detection if hotword detection didn't find anything conclusive
    logger.info("Falling back to rule-based detection...")
    rule_result = rule_based_detector.detect(text)
    
    # Map rule-based results to categories
    confidence = rule_result["confidence"]
    if confidence > 0.7:
        category = "highly_suspicious"
        severity = min(10, int(confidence * 10))
    elif confidence > 0.5:
        category = "suspicious"
        severity = min(7, int(confidence * 8))
    elif confidence > 0.3:
        category = "neutral"
        severity = min(5, int(confidence * 6))
    else:
        category = "safe"
        severity = min(3, int(confidence * 4))
    
    # Format prediction text
    if category == "highly_suspicious":
        prediction_text = "🚨 HIGH RISK: Potential Scam Detected!"
    elif category == "suspicious":
        prediction_text = "⚠️ SUSPICIOUS: Possible Scam Detected"
    elif category == "neutral":
        prediction_text = "🔍 NEUTRAL: Some Unusual Elements"
    else:
        prediction_text = "✅ SAFE: No Scam Detected"
    
    return {
        "is_spam": rule_result["is_spam"],
        "prediction": prediction_text,
        "confidence": rule_result["confidence"],
        "category": category,
        "severity": severity,
        "matched_patterns": rule_result.get("matched_patterns", []),
        "detection_method": "rule_based"
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
            
            # Add category and severity for UI improvements
            response_data = {
                "transcription": transcription,
                "prediction": prediction_result["prediction"],
                "is_spam": prediction_result["is_spam"],
                "confidence": prediction_result["confidence"],
                "category": prediction_result.get("category", "safe"),
                "severity": prediction_result.get("severity", 0)
            }
            
            # Add matched patterns or hotwords if available
            if "matches" in prediction_result:
                response_data["matches"] = prediction_result["matches"]
            elif "matched_patterns" in prediction_result:
                response_data["matched_patterns"] = prediction_result["matched_patterns"]
                
            return jsonify(response_data)
        
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
            
            # Add category and severity for UI improvements
            response_data = {
                "transcription": transcription,
                "prediction": prediction_result["prediction"],
                "is_spam": prediction_result["is_spam"],
                "confidence": prediction_result["confidence"],
                "category": prediction_result.get("category", "safe"),
                "severity": prediction_result.get("severity", 0),
                "empty": False
            }
            
            # Add matched patterns or hotwords if available
            if "matches" in prediction_result:
                response_data["matches"] = prediction_result["matches"]
            elif "matched_patterns" in prediction_result:
                response_data["matched_patterns"] = prediction_result["matched_patterns"]
                
            return jsonify(response_data)
        
        except Exception as e:
            # Clean up the temporary file in case of error
            if os.path.exists(filepath):
                os.remove(filepath)
            
            logging.error(f"Error processing audio chunk: {e}")
            return jsonify({"error": f"Error processing audio chunk: {str(e)}"}), 500
    
    return jsonify({"error": "Invalid audio format"}), 400
