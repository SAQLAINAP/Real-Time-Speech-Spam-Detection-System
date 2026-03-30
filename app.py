import os
import logging
import json
import tempfile
import uuid
import requests as http_requests
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from openai import OpenAI

from utils.transcriber import WhisperTranscriber
from utils.rule_based_detector import RuleBasedScamDetector
from utils.hotword_detector import HotwordDetector
from utils.hotwords_data import hotwords_severity
from utils.embedding_detector import EmbeddingScamDetector

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default_secret_key")

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'webm', 'm4a'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# ── API clients ───────────────────────────────────────────────────────────────
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ── Transcribers ──────────────────────────────────────────────────────────────
# "base" (~140 MB) for single-file analysis — more accurate
transcriber = WhisperTranscriber(model_size="base")
# "tiny" (~75 MB) for real-time monitoring chunks — much faster (~1-2 s vs 5-8 s)
monitoring_transcriber = WhisperTranscriber(model_size="tiny")

# ── Detectors ─────────────────────────────────────────────────────────────────
rule_based_detector = RuleBasedScamDetector()
hotword_detector = HotwordDetector(hotwords_severity)
# Semantic embedding detector — initialises lazily on first call (needs OpenAI key)
embedding_detector = EmbeddingScamDetector(openai_client)

# ── GPT system prompt (2C: phone-call specific, false-positive aware) ─────────
SCAM_DETECTION_SYSTEM_PROMPT = """You are an expert at detecting phone call scams in real time.
You analyse spoken conversation transcripts for fraud indicators.

SCAM SIGNALS to look for:
- Impersonation of government agencies (IRS, SSA, Medicare, police)
- Artificial urgency or fear ("legal action", "arrest warrant", "suspended account")
- Payment via gift cards, wire transfer, cryptocurrency, or prepaid cards
- Requests for SSN, bank/card details, passwords, or remote device access
- Unsolicited prize/lottery winnings requiring upfront fees
- Tech-support scams claiming your computer/account is compromised

FALSE POSITIVES to AVOID:
- Normal mentions of "account", "security", "verify" without suspicious context
- Phrases like "your account is secure" or "we verified your identity successfully"
- Customer service follow-ups WITHOUT demands for payment or personal data
- The caller describing their own situation rather than demanding action

If a PREVIOUS CONVERSATION CONTEXT is provided, use it to judge continuity and escalating pressure patterns.

Respond ONLY with a JSON object (no markdown) with these exact keys:
{
  "is_spam": <boolean>,
  "confidence": <float 0-1>,
  "category": <"safe" | "neutral" | "suspicious" | "highly_suspicious">,
  "prediction": <short human-readable verdict>,
  "severity": <integer 0-10>,
  "reasoning": <one sentence explaining the key signal>
}"""


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── 2A: Deepgram fast transcription ──────────────────────────────────────────
def deepgram_transcribe(audio_path: str):
    """Transcribe via Deepgram REST API (nova-2 model, ~300 ms latency)."""
    if not DEEPGRAM_API_KEY:
        return None
    try:
        ext = os.path.splitext(audio_path)[1].lstrip('.').lower()
        mime_map = {'webm': 'audio/webm', 'wav': 'audio/wav',
                    'mp3': 'audio/mpeg', 'ogg': 'audio/ogg', 'm4a': 'audio/mp4'}
        mime = mime_map.get(ext, 'audio/webm')

        with open(audio_path, 'rb') as f:
            resp = http_requests.post(
                "https://api.deepgram.com/v1/listen?model=nova-2&language=en&smart_format=true",
                headers={"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": mime},
                data=f.read(),
                timeout=12
            )
        if resp.status_code == 200:
            transcript = resp.json()["results"]["channels"][0]["alternatives"][0]["transcript"]
            if transcript.strip():
                logger.info("Deepgram transcription successful")
                return transcript
    except Exception as e:
        logger.error(f"Deepgram transcription error: {e}")
    return None


def transcribe_audio(audio_path: str, fast: bool = False) -> str:
    """
    Transcribe audio with a priority chain:
      1. Deepgram (if DEEPGRAM_API_KEY set) — fastest
      2. Local Whisper tiny (if fast=True) or base — reliable offline
      3. OpenAI Whisper API — fallback
    """
    # 1. Deepgram (fast mode preferred)
    if fast or DEEPGRAM_API_KEY:
        result = deepgram_transcribe(audio_path)
        if result:
            return result

    # 2. Local Whisper
    local = monitoring_transcriber if fast else transcriber
    try:
        logger.info(f"Local Whisper ({'tiny' if fast else 'base'}) transcription...")
        result = local.transcribe(audio_path)
        if result:
            logger.info("Local transcription successful")
            return result
    except Exception as e:
        logger.error(f"Local Whisper error: {e}")

    # 3. OpenAI Whisper API fallback
    if openai_client:
        try:
            logger.info("Falling back to OpenAI Whisper API...")
            with open(audio_path, "rb") as f:
                resp = openai_client.audio.transcriptions.create(model="whisper-1", file=f)
            return resp.text
        except Exception as e:
            logger.error(f"OpenAI Whisper API error: {e}")

    return "Error transcribing audio. Please try again."


# ── 2C: Improved contextual spam detection ────────────────────────────────────
def predict_spam(text: str, context: str = None) -> dict:
    """
    Detect spam/scam with a cascading pipeline:
      1. OpenAI GPT-4o (context-aware, low false positives)
      2. Hotword detection (with negation awareness)
      3. Rule-based fallback
    """
    # Build the user message, optionally prepending conversation context
    user_message = text
    if context and context.strip():
        user_message = (
            f"[Previous conversation context]\n{context.strip()}\n\n"
            f"[New segment to analyze]\n{text}"
        )

    # 1. OpenAI GPT-4o
    if openai_client:
        try:
            logger.info("Analyzing with GPT-4o (contextual)...")
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SCAM_DETECTION_SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message}
                ],
                response_format={"type": "json_object"},
                temperature=0.1   # low temperature for consistent classification
            )
            result = json.loads(response.choices[0].message.content)

            category = result.get("category", "safe")
            severity = result.get("severity", 0)
            prediction_labels = {
                "highly_suspicious": "🚨 HIGH RISK: Potential Scam Detected!",
                "suspicious":        "⚠️ SUSPICIOUS: Possible Scam Detected",
                "neutral":           "🔍 NEUTRAL: Some Unusual Elements",
                "safe":              "✅ SAFE: No Scam Detected"
            }
            return {
                "is_spam":    result.get("is_spam", False),
                "prediction": prediction_labels.get(category, "✅ SAFE: No Scam Detected"),
                "confidence": result.get("confidence", 0.5),
                "category":   category,
                "severity":   severity,
                "reasoning":  result.get("reasoning", "")
            }
        except Exception as e:
            logger.error(f"GPT-4o error: {e}")

    # 2. Semantic embedding similarity (robust fallback — not keyword-based)
    logger.info("Using embedding-based semantic detection...")
    emb_result = embedding_detector.detect(text)
    if emb_result is not None:
        return emb_result

    # 3. Hotword detection (negation-aware)
    logger.info("Using hotword-based detection...")
    hotword_result = hotword_detector.detect(text)

    if hotword_result["is_spam"] or hotword_result["confidence"] > 0.3:
        category = hotword_result["category"]
        prediction_labels = {
            "highly_suspicious": "🚨 HIGH RISK: Potential Scam Detected!",
            "suspicious":        "⚠️ SUSPICIOUS: Possible Scam Detected",
            "neutral":           "🔍 NEUTRAL: Some Unusual Elements",
            "safe":              "✅ SAFE: No Scam Detected"
        }
        return {
            "is_spam":    hotword_result["is_spam"],
            "prediction": prediction_labels.get(category, "✅ SAFE: No Scam Detected"),
            "confidence": hotword_result["confidence"],
            "category":   category,
            "severity":   hotword_result["severity"],
            "matches":    hotword_result["matches"],
            "detection_method": "hotword"
        }

    # 4. Rule-based fallback (last resort)
    logger.info("Falling back to rule-based detection...")
    rule_result = rule_based_detector.detect(text)
    confidence = rule_result["confidence"]

    if confidence > 0.7:
        category, severity = "highly_suspicious", min(10, int(confidence * 10))
    elif confidence > 0.5:
        category, severity = "suspicious", min(7, int(confidence * 8))
    elif confidence > 0.3:
        category, severity = "neutral", min(5, int(confidence * 6))
    else:
        category, severity = "safe", min(3, int(confidence * 4))

    prediction_labels = {
        "highly_suspicious": "🚨 HIGH RISK: Potential Scam Detected!",
        "suspicious":        "⚠️ SUSPICIOUS: Possible Scam Detected",
        "neutral":           "🔍 NEUTRAL: Some Unusual Elements",
        "safe":              "✅ SAFE: No Scam Detected"
    }
    return {
        "is_spam":          rule_result["is_spam"],
        "prediction":       prediction_labels.get(category, "✅ SAFE: No Scam Detected"),
        "confidence":       confidence,
        "category":         category,
        "severity":         severity,
        "matched_patterns": rule_result.get("matched_patterns", []),
        "detection_method": "rule_based"
    }


# ── Optional: Plivo phone integration (2B) ───────────────────────────────────
# Registers /plivo/* routes only when PLIVO_AUTH_ID is set — zero impact otherwise.
if os.environ.get("PLIVO_AUTH_ID"):
    from utils.plivo_handler import plivo_bp
    app.register_blueprint(plivo_bp)
    logger.info("Plivo integration enabled — routes registered at /plivo/*")
else:
    logger.info("Plivo integration disabled (set PLIVO_AUTH_ID to enable)")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_audio():
    """Process uploaded or recorded audio (single-analysis mode)."""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not (file and allowed_file(file.filename)):
        return jsonify({"error": "Invalid file format. Allowed: wav, mp3, ogg, flac, webm, m4a"}), 400

    unique_filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        transcription = transcribe_audio(filepath, fast=False)
        prediction_result = predict_spam(transcription)
        os.remove(filepath)

        response_data = {
            "transcription": transcription,
            "prediction":    prediction_result["prediction"],
            "is_spam":       prediction_result["is_spam"],
            "confidence":    prediction_result["confidence"],
            "category":      prediction_result.get("category", "safe"),
            "severity":      prediction_result.get("severity", 0)
        }
        if "matches" in prediction_result:
            response_data["matches"] = prediction_result["matches"]
        elif "matched_patterns" in prediction_result:
            response_data["matched_patterns"] = prediction_result["matched_patterns"]

        return jsonify(response_data)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Error processing audio: {e}")
        return jsonify({"error": f"Error processing audio: {str(e)}"}), 500


@app.route('/analyze-chunk', methods=['POST'])
def analyze_audio_chunk():
    """
    Process a real-time monitoring chunk.
    Accepts optional 'context' field (previous transcript text) for context-aware detection.
    Uses the fast (tiny Whisper / Deepgram) transcriber.
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio chunk provided"}), 400

    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No audio chunk"}), 400

    # Optional conversation context from previous chunks
    context = request.form.get('context', '')

    unique_filename = f"{uuid.uuid4()}_chunk.webm"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    try:
        # 2A: use fast=True → Deepgram or tiny Whisper
        transcription = transcribe_audio(filepath, fast=True)

        if not transcription or transcription == "Error transcribing audio. Please try again.":
            os.remove(filepath)
            return jsonify({"transcription": "", "is_spam": False, "empty": True})

        # 2C: pass accumulated context for smarter detection
        prediction_result = predict_spam(transcription, context=context)
        os.remove(filepath)

        response_data = {
            "transcription": transcription,
            "prediction":    prediction_result["prediction"],
            "is_spam":       prediction_result["is_spam"],
            "confidence":    prediction_result["confidence"],
            "category":      prediction_result.get("category", "safe"),
            "severity":      prediction_result.get("severity", 0),
            "empty":         False
        }
        if "matches" in prediction_result:
            response_data["matches"] = prediction_result["matches"]
        elif "matched_patterns" in prediction_result:
            response_data["matched_patterns"] = prediction_result["matched_patterns"]

        return jsonify(response_data)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Error processing audio chunk: {e}")
        return jsonify({"error": f"Error processing audio chunk: {str(e)}"}), 500


@app.route('/analyze-text', methods=['POST'])
def analyze_text():
    """Analyze text directly — used by demo sample buttons (no audio needed)."""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    prediction_result = predict_spam(text)

    response_data = {
        "transcription": text,
        "prediction":    prediction_result["prediction"],
        "is_spam":       prediction_result["is_spam"],
        "confidence":    prediction_result["confidence"],
        "category":      prediction_result.get("category", "safe"),
        "severity":      prediction_result.get("severity", 0)
    }
    if "matches" in prediction_result:
        response_data["matches"] = prediction_result["matches"]
    elif "matched_patterns" in prediction_result:
        response_data["matched_patterns"] = prediction_result["matched_patterns"]

    return jsonify(response_data)
