"""
REST API Blueprint (Phase 3B)
------------------------------
Exposes clean /api/v1/* endpoints for B2B integration.

Authentication:
  Pass your API key in the X-API-Key header.
  Keys are set via the API_KEYS environment variable (comma-separated).
  Example: export API_KEYS="key-abc123,key-xyz456"
  If API_KEYS is not set, auth is disabled (dev mode).

Endpoints:
  GET  /api/v1/health          — health check, no auth required
  GET  /api/v1/docs            — Swagger UI
  GET  /api/v1/openapi.json    — raw OpenAPI spec
  POST /api/v1/analyze         — analyze an audio file
  POST /api/v1/analyze-text    — analyze text directly
"""

import os
import json
import logging
import tempfile
import uuid
from functools import wraps
from flask import Blueprint, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")

ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'webm', 'm4a'}

# ── API Key auth ──────────────────────────────────────────────────────────────
_raw_keys = os.environ.get("API_KEYS", "")
VALID_KEYS = set(k.strip() for k in _raw_keys.split(",") if k.strip())
AUTH_ENABLED = bool(VALID_KEYS)

if not AUTH_ENABLED:
    logger.warning("API_KEYS not set — API auth disabled (dev mode)")


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if not key or key not in VALID_KEYS:
            return jsonify({"error": "Invalid or missing API key", "code": 401}), 401
        return f(*args, **kwargs)
    return decorated


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Routes ────────────────────────────────────────────────────────────────────
@api_bp.route("/health", methods=["GET"])
def health():
    """Health check — no auth required."""
    from app import transcriber, openai_client, DEEPGRAM_API_KEY
    return jsonify({
        "status":      "ok",
        "version":     "1.0.0",
        "auth_enabled": AUTH_ENABLED,
        "capabilities": {
            "openai":   bool(openai_client),
            "deepgram": bool(DEEPGRAM_API_KEY),
            "whisper":  transcriber.model_size
        }
    })


@api_bp.route("/analyze", methods=["POST"])
@require_api_key
def api_analyze_audio():
    """
    Analyze an audio file for scam content.

    Request (multipart/form-data):
      audio  — audio file (wav, mp3, ogg, flac, webm, m4a)

    Response (application/json):
      transcription  string   — transcribed text
      is_spam        boolean  — whether a scam was detected
      category       string   — safe | neutral | suspicious | highly_suspicious
      confidence     float    — 0.0–1.0
      severity       integer  — 0–10
      prediction     string   — human-readable verdict
      matches        array    — matched hotwords (if applicable)
    """
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided. Send file in 'audio' field.", "code": 400}), 400

    file = request.files['audio']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "code": 400}), 400

    filepath = os.path.join(tempfile.gettempdir(), f"api_{uuid.uuid4()}_{secure_filename(file.filename)}")
    file.save(filepath)

    try:
        from app import transcribe_audio, predict_spam
        transcription = transcribe_audio(filepath, fast=False)
        result = predict_spam(transcription)
        os.remove(filepath)

        response = {
            "transcription": transcription,
            "is_spam":       result["is_spam"],
            "category":      result.get("category", "safe"),
            "confidence":    result["confidence"],
            "severity":      result.get("severity", 0),
            "prediction":    result["prediction"],
        }
        if "matches" in result:
            response["matches"] = result["matches"]
        elif "matched_patterns" in result:
            response["matched_patterns"] = result["matched_patterns"]

        return jsonify(response)

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"[API] analyze error: {e}")
        return jsonify({"error": str(e), "code": 500}), 500


@api_bp.route("/analyze-text", methods=["POST"])
@require_api_key
def api_analyze_text():
    """
    Analyze text directly for scam content (no audio needed).

    Request (application/json):
      text     string   — text to analyze
      context  string   — optional prior conversation context

    Response: same as /analyze (without transcription field)
    """
    data = request.get_json(silent=True)
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Request body must be JSON with a 'text' field.", "code": 400}), 400

    text    = data["text"].strip()
    context = data.get("context", "")

    try:
        from app import predict_spam
        result = predict_spam(text, context=context or None)

        response = {
            "transcription": text,
            "is_spam":       result["is_spam"],
            "category":      result.get("category", "safe"),
            "confidence":    result["confidence"],
            "severity":      result.get("severity", 0),
            "prediction":    result["prediction"],
        }
        if "matches" in result:
            response["matches"] = result["matches"]
        elif "matched_patterns" in result:
            response["matched_patterns"] = result["matched_patterns"]

        return jsonify(response)

    except Exception as e:
        logger.error(f"[API] analyze-text error: {e}")
        return jsonify({"error": str(e), "code": 500}), 500


# ── OpenAPI spec ──────────────────────────────────────────────────────────────
OPENAPI_SPEC = {
    "openapi": "3.0.3",
    "info": {
        "title":       "ScamShield API",
        "version":     "1.0.0",
        "description": "Real-time speech scam detection API. Analyze audio files or text for fraud indicators."
    },
    "servers": [{"url": "/api/v1"}],
    "security": [{"ApiKeyAuth": []}],
    "components": {
        "securitySchemes": {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in":   "header",
                "name": "X-API-Key"
            }
        },
        "schemas": {
            "DetectionResult": {
                "type": "object",
                "properties": {
                    "transcription": {"type": "string"},
                    "is_spam":       {"type": "boolean"},
                    "category":      {"type": "string", "enum": ["safe", "neutral", "suspicious", "highly_suspicious"]},
                    "confidence":    {"type": "number", "minimum": 0, "maximum": 1},
                    "severity":      {"type": "integer", "minimum": 0, "maximum": 10},
                    "prediction":    {"type": "string"}
                }
            }
        }
    },
    "paths": {
        "/health": {
            "get": {
                "summary":     "Health check",
                "description": "Returns API status and available capabilities. No auth required.",
                "security":    [],
                "responses":   {"200": {"description": "API is healthy"}}
            }
        },
        "/analyze": {
            "post": {
                "summary":     "Analyze audio file",
                "description": "Upload an audio file. Returns transcription and scam detection result.",
                "requestBody": {
                    "required": True,
                    "content": {
                        "multipart/form-data": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "audio": {"type": "string", "format": "binary"}
                                },
                                "required": ["audio"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Detection result",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DetectionResult"}
                            }
                        }
                    },
                    "400": {"description": "Bad request"},
                    "401": {"description": "Invalid API key"}
                }
            }
        },
        "/analyze-text": {
            "post": {
                "summary":     "Analyze text",
                "description": "Send text directly for scam detection (no audio required).",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "text":    {"type": "string", "description": "Text to analyze"},
                                    "context": {"type": "string", "description": "Optional prior conversation context"}
                                },
                                "required": ["text"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Detection result",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/DetectionResult"}
                            }
                        }
                    },
                    "400": {"description": "Bad request"},
                    "401": {"description": "Invalid API key"}
                }
            }
        }
    }
}


@api_bp.route("/openapi.json", methods=["GET"])
def openapi_spec():
    """Serve the raw OpenAPI 3.0 spec."""
    return jsonify(OPENAPI_SPEC)


# ── Swagger UI ────────────────────────────────────────────────────────────────
SWAGGER_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>ScamShield API Docs</title>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
<div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>
  SwaggerUIBundle({
    url: '/api/v1/openapi.json',
    dom_id: '#swagger-ui',
    presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
    layout: 'BaseLayout',
    deepLinking: true
  });
</script>
</body>
</html>"""


@api_bp.route("/docs", methods=["GET"])
def swagger_ui():
    """Interactive Swagger UI for the API."""
    return render_template_string(SWAGGER_HTML)
