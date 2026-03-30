"""
Plivo Phone Call Integration (Phase 2B)
-----------------------------------------
Optional Blueprint — registers only when PLIVO_AUTH_ID is set.
Does NOT touch any existing routes or logic.

Environment variables required:
  PLIVO_AUTH_ID       — from your Plivo console
  PLIVO_AUTH_TOKEN    — from your Plivo console
  PLIVO_FROM_NUMBER   — your Plivo phone number (E.164, e.g. +12025551234)
  PLIVO_ALERT_NUMBER  — number to SMS when a scam is detected

Plivo console setup:
  1. Create an application → set Answer URL to https://yourhost/plivo/answer
  2. Assign your Plivo number to that application
  3. (Optional) set Event URL to https://yourhost/plivo/status

Flow:
  Inbound call  →  /plivo/answer  →  PHML: Record + callback
  Recording done →  /plivo/recording  →  transcribe → detect → SMS alert
  Call events   →  /plivo/status  (logged only)

For real-time streaming (lower latency), see the STREAMING NOTE at the
bottom of this file — requires installing flask-sock and an SSL-enabled host.
"""

import os
import logging
import tempfile
import threading
import uuid

from flask import Blueprint, request, jsonify, current_app

logger = logging.getLogger(__name__)

# ── Env vars ──────────────────────────────────────────────────────────────────
PLIVO_AUTH_ID      = os.environ.get("PLIVO_AUTH_ID")
PLIVO_AUTH_TOKEN   = os.environ.get("PLIVO_AUTH_TOKEN")
PLIVO_FROM_NUMBER  = os.environ.get("PLIVO_FROM_NUMBER")   # your Plivo number
PLIVO_ALERT_NUMBER = os.environ.get("PLIVO_ALERT_NUMBER")  # SMS alert recipient

# ── Blueprint ─────────────────────────────────────────────────────────────────
plivo_bp = Blueprint("plivo", __name__, url_prefix="/plivo")


# ── PHML helpers ──────────────────────────────────────────────────────────────
def _phml_record(callback_url: str) -> str:
    """
    PHML that plays a brief notice then records the caller.
    When recording finishes Plivo POSTs to `callback_url`.
    """
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak voice="WOMAN">
        This call is being analyzed for your protection.
        Please speak after the tone.
    </Speak>
    <Record
        action="{callback_url}"
        maxLength="120"
        finishOnKey="#"
        callbackUrl="{callback_url}"
        callbackMethod="POST"
        recordSession="true"
        redirect="false"
    />
</Response>"""


def _phml_hangup() -> str:
    return """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Speak>Thank you. Goodbye.</Speak>
    <Hangup/>
</Response>"""


# ── Routes ────────────────────────────────────────────────────────────────────
@plivo_bp.route("/answer", methods=["GET", "POST"])
def answer():
    """
    Plivo Answer URL — called when an inbound call connects.
    Returns PHML instructing Plivo to record the caller's audio.
    """
    call_uuid   = request.values.get("CallUUID", "unknown")
    from_number = request.values.get("From",     "unknown")
    to_number   = request.values.get("To",       "unknown")
    logger.info(f"[Plivo] Inbound call {call_uuid}: {from_number} → {to_number}")

    # Build absolute callback URL
    base_url      = request.host_url.rstrip("/")
    callback_url  = f"{base_url}/plivo/recording"

    xml = _phml_record(callback_url)
    return current_app.response_class(xml, mimetype="application/xml")


@plivo_bp.route("/recording", methods=["GET", "POST"])
def recording_callback():
    """
    Plivo calls this when a recording is complete.
    Spawns a background thread to download → transcribe → detect → alert.
    Returns 200 immediately so Plivo doesn't time out.
    """
    recording_url = (
        request.values.get("RecordUrl") or
        request.values.get("RecordingUrl") or
        request.values.get("record_url")
    )
    call_uuid   = request.values.get("CallUUID",   "unknown")
    from_number = request.values.get("From",        "unknown")
    duration    = request.values.get("RecordingDuration", "?")

    logger.info(f"[Plivo] Recording ready — call {call_uuid}, {duration}s, {from_number}")

    if not recording_url:
        logger.warning("[Plivo] No recording URL in callback — check Plivo app settings")
        return jsonify({"error": "No recording URL provided"}), 400

    thread = threading.Thread(
        target=_process_recording,
        args=(recording_url, call_uuid, from_number),
        daemon=True
    )
    thread.start()

    return jsonify({"status": "processing", "call_uuid": call_uuid}), 200


@plivo_bp.route("/status", methods=["GET", "POST"])
def call_status():
    """Plivo Event URL — logs call lifecycle events (no action needed)."""
    call_uuid = request.values.get("CallUUID",   "unknown")
    status    = request.values.get("CallStatus", "unknown")
    duration  = request.values.get("Duration",   "?")
    logger.info(f"[Plivo] Call status — {call_uuid}: {status} ({duration}s)")
    return jsonify({"received": True}), 200


@plivo_bp.route("/test-alert", methods=["POST"])
def test_alert():
    """
    Dev-only endpoint: POST {"text": "..."} to test the full SMS alert flow
    without making a real call.
    """
    data = request.get_json() or {}
    text = data.get(
        "text",
        "This is the IRS. You owe back taxes. Wire transfer immediately to avoid arrest."
    )
    fake_result = {
        "is_spam":  True,
        "category": "highly_suspicious",
        "severity": 9,
        "confidence": 0.95,
    }
    _send_sms_alert("TEST-CALL", "TEST-NUMBER", text, fake_result)
    return jsonify({"status": "alert triggered", "text": text})


@plivo_bp.route("/health", methods=["GET"])
def health():
    """Quick check that the Plivo blueprint is alive and configured."""
    return jsonify({
        "status":    "ok",
        "auth_id":   bool(PLIVO_AUTH_ID),
        "from_num":  bool(PLIVO_FROM_NUMBER),
        "alert_num": bool(PLIVO_ALERT_NUMBER),
    })


# ── Background processing ─────────────────────────────────────────────────────
def _process_recording(recording_url: str, call_uuid: str, from_number: str):
    """
    Downloads the Plivo recording, transcribes it, runs spam detection,
    and fires an SMS alert if a scam is detected.
    Runs in a daemon thread — never blocks the main request.
    """
    tmp_path = None
    try:
        # 1. Download audio
        import requests as req
        logger.info(f"[Plivo] Downloading recording for {call_uuid}…")
        resp = req.get(recording_url, timeout=30)
        if resp.status_code != 200:
            logger.error(f"[Plivo] Download failed: HTTP {resp.status_code}")
            return

        tmp_path = os.path.join(tempfile.gettempdir(), f"plivo_{uuid.uuid4()}.mp3")
        with open(tmp_path, "wb") as f:
            f.write(resp.content)

        # 2. Transcribe (use the accurate base model — not time-critical here)
        from app import transcribe_audio, predict_spam
        logger.info(f"[Plivo] Transcribing {call_uuid}…")
        transcription = transcribe_audio(tmp_path, fast=False)

        if not transcription or transcription.startswith("Error"):
            logger.error(f"[Plivo] Transcription failed for {call_uuid}")
            return

        logger.info(f"[Plivo] Transcript ({call_uuid}): {transcription[:120]}…")

        # 3. Spam detection
        result = predict_spam(transcription)
        logger.info(
            f"[Plivo] Detection result — {result['category']} "
            f"({result['confidence']:.0%} confidence, severity {result['severity']}/10)"
        )

        # 4. Alert if spam detected
        if result["is_spam"]:
            _send_sms_alert(call_uuid, from_number, transcription, result)
        else:
            logger.info(f"[Plivo] Call {call_uuid} classified as safe — no alert sent")

    except Exception as e:
        logger.error(f"[Plivo] Error processing recording {call_uuid}: {e}", exc_info=True)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _send_sms_alert(call_uuid: str, from_number: str, transcription: str, result: dict):
    """Send an SMS via Plivo when a scam call is detected."""
    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_FROM_NUMBER, PLIVO_ALERT_NUMBER]):
        logger.warning(
            "[Plivo] SMS alert skipped — set PLIVO_FROM_NUMBER and PLIVO_ALERT_NUMBER "
            "environment variables to enable SMS alerts"
        )
        return

    try:
        import plivo  # pip install plivo
        client = plivo.RestClient(PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN)

        category = result["category"].replace("_", " ").title()
        severity = result["severity"]
        confidence = int(result["confidence"] * 100)
        snippet = transcription[:140] + ("…" if len(transcription) > 140 else "")

        message = (
            f"🚨 SCAM CALL ALERT\n"
            f"From: {from_number}\n"
            f"Risk: {category} | Severity: {severity}/10 | Confidence: {confidence}%\n"
            f'Transcript: "{snippet}"'
        )

        client.messages.create(
            src=PLIVO_FROM_NUMBER,
            dst=PLIVO_ALERT_NUMBER,
            text=message
        )
        logger.info(f"[Plivo] SMS alert sent for call {call_uuid} → {PLIVO_ALERT_NUMBER}")

    except ImportError:
        logger.error("[Plivo] 'plivo' package not installed. Run: pip install plivo")
    except Exception as e:
        logger.error(f"[Plivo] SMS alert failed for {call_uuid}: {e}")


# ── STREAMING NOTE ─────────────────────────────────────────────────────────────
# For sub-second real-time analysis during the call (instead of post-call
# recording), Plivo supports WebSocket audio streaming via PHML <Stream>.
#
# PHML to start streaming:
#   <Response>
#     <Stream streamTimeout="86400" keepCallAlive="true"
#             bidirectional="false" audioTrack="inbound"
#             contentType="audio/x-mulaw;rate=8000">
#       wss://yourhost/plivo/stream
#     </Stream>
#   </Response>
#
# Plivo sends JSON frames:  {"event": "media", "media": {"payload": "<base64 mulaw>"}}
# You need to:
#   1. pip install flask-sock
#   2. Convert mulaw → wav using audioop.ulaw2lin + wave module
#   3. Buffer ~3s of audio, write to temp file, call transcribe_audio(fast=True)
#   4. Run predict_spam and push result back to browser via SSE or another socket
#
# This adds ~200ms latency vs 10-30s for recording-based approach.
