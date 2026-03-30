"""
Semantic Embedding-Based Scam Detector
---------------------------------------
Uses OpenAI text-embedding-3-small to compute cosine similarity between the
input text and a curated reference set of known scam / ham sentences.

Why this beats keyword matching:
  - "We need to verify your personal details right now or your benefits stop"
    contains zero hotwords but is semantically close to known scam patterns.
  - "Your account is completely secure, no action needed" triggers keyword
    detectors but is semantically far from scam references.

Cost: text-embedding-3-small is ~$0.02 / 1M tokens ≈ negligible per call.

The reference embeddings are computed ONCE on first use and cached in memory.
"""

import logging
import math
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

# ── Reference sentences ──────────────────────────────────────────────────────
# Diverse paraphrases of real scam call patterns (avoid exact hotword overlap
# where possible — the model should catch meaning, not surface keywords).

SCAM_REFERENCES = [
    # Government impersonation
    "This is the Internal Revenue Service. You have unpaid taxes and officers are on their way to arrest you.",
    "I'm calling from Social Security. Your number has been suspended due to suspicious activity.",
    "This is a federal officer. There is a warrant out for your arrest unless you pay immediately.",
    "We are from Medicare and your benefits will be cancelled if you don't confirm your details now.",

    # Urgency / legal threats
    "You must act right now or face serious legal consequences. This is your final warning.",
    "A lawsuit has been filed against you. To stop the proceedings you need to make a payment today.",
    "If you do not respond in the next hour we will send law enforcement to your address.",

    # Payment demands
    "Please purchase some gift cards and read me the numbers on the back to clear your debt.",
    "You need to wire the money immediately to resolve this issue and avoid prosecution.",
    "We accept payment in Bitcoin or cryptocurrency only. You must transfer the funds today.",
    "Go to the nearest store and buy Google Play cards. Call us back with the codes.",

    # Personal information phishing
    "I need you to confirm your Social Security number and bank routing number right now.",
    "Please provide your credit card details so we can process your refund immediately.",
    "We need your full date of birth and mother's maiden name to verify your identity.",
    "Give me your online banking login details so our technician can fix the problem remotely.",

    # Prize / lottery scams
    "Congratulations! You have been randomly selected to receive fifty thousand dollars.",
    "You won our international sweepstakes. Just pay a small processing fee to receive your prize.",
    "Our records show you have an unclaimed inheritance waiting for you in a foreign bank.",

    # Tech support scams
    "We are calling from Windows technical support. Your computer is infected with a dangerous virus.",
    "Your Apple account has been compromised. I need remote access to your device to fix it.",
    "I can see on our system that your computer is sending out malicious data right now.",

    # Investment / crypto scams
    "This is a guaranteed investment opportunity. You will double your money within 30 days.",
    "Our crypto platform offers risk-free returns of 300 percent. Invest today before the offer expires.",
]

HAM_REFERENCES = [
    # Normal customer service
    "Hi, I'm calling to confirm your appointment scheduled for tomorrow afternoon.",
    "This is a reminder that your annual subscription will renew next week.",
    "Thank you for contacting our support team. How can I assist you today?",
    "I'm following up on the service request you submitted earlier this week.",
    "Your package has been delivered. Please let us know if you have any questions.",

    # Normal account communications
    "Your account is in good standing. There is nothing you need to do at this time.",
    "We have processed your refund and it should appear in 3 to 5 business days.",
    "We wanted to confirm that your recent password change was successful.",
    "Your direct deposit has been received and funds are available in your account.",

    # Normal billing / reminders
    "This is an automated reminder that your payment is due at the end of the month.",
    "We noticed your card on file will expire soon. Please update it at your convenience.",
    "Your insurance premium has been successfully processed for this month.",

    # Normal verification (non-threatening)
    "To protect your account we sent a verification code to your registered phone number.",
    "For security purposes we occasionally verify account holder information during routine checks.",
]


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Pure-Python cosine similarity — no numpy required."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingScamDetector:
    """
    Semantic scam detector. Requires an OpenAI client instance.
    Falls back gracefully (returns None) if embeddings can't be computed.
    """

    def __init__(self, openai_client):
        self.client = openai_client
        self._scam_embeddings: Optional[List[List[float]]] = None
        self._ham_embeddings:  Optional[List[List[float]]] = None
        self._ready = False

    # ── Initialisation ────────────────────────────────────────────────────────
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def initialize(self) -> bool:
        """Compute reference embeddings once and cache them in memory."""
        if self._ready:
            return True
        if not self.client:
            return False
        try:
            logger.info("EmbeddingDetector: computing reference embeddings (one-time)...")
            self._scam_embeddings = self._embed_batch(SCAM_REFERENCES)
            self._ham_embeddings  = self._embed_batch(HAM_REFERENCES)
            self._ready = True
            logger.info(f"EmbeddingDetector ready — {len(SCAM_REFERENCES)} scam / {len(HAM_REFERENCES)} ham references")
            return True
        except Exception as e:
            logger.error(f"EmbeddingDetector init failed: {e}")
            return False

    # ── Detection ─────────────────────────────────────────────────────────────
    def detect(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Returns a detection result dict, or None if the detector isn't ready.
        The caller should fall through to the next detector on None.
        """
        if not self._ready and not self.initialize():
            return None

        try:
            text_emb = self._embed_batch([text])[0]

            scam_sims = [_cosine_similarity(text_emb, ref) for ref in self._scam_embeddings]
            ham_sims  = [_cosine_similarity(text_emb, ref) for ref in self._ham_embeddings]

            max_scam = max(scam_sims)
            avg_scam = sum(scam_sims) / len(scam_sims)
            max_ham  = max(ham_sims)

            # Combined scam signal (weighted max + average)
            scam_signal = max_scam * 0.65 + avg_scam * 0.35

            # Penalise if text is even more similar to ham
            net_signal = scam_signal - max_ham * 0.45

            # Map to confidence [0, 1]
            confidence = min(1.0, max(0.0, net_signal + 0.35))

            # Category thresholds (tuned on the reference set above)
            if scam_signal > 0.72 and scam_signal > max_ham + 0.08:
                category, is_spam, severity = "highly_suspicious", True, min(10, round(scam_signal * 13))
            elif scam_signal > 0.60:
                category, is_spam, severity = "suspicious",        True, min(7,  round(scam_signal * 10))
            elif scam_signal > 0.50:
                category, is_spam, severity = "neutral",           False, min(5, round(scam_signal * 7))
            else:
                category, is_spam, severity = "safe",              False, min(3, round(scam_signal * 4))

            labels = {
                "highly_suspicious": "🚨 HIGH RISK: Potential Scam Detected!",
                "suspicious":        "⚠️ SUSPICIOUS: Possible Scam Detected",
                "neutral":           "🔍 NEUTRAL: Some Unusual Elements",
                "safe":              "✅ SAFE: No Scam Detected",
            }

            return {
                "is_spam":          is_spam,
                "prediction":       labels[category],
                "confidence":       round(confidence, 2),
                "category":         category,
                "severity":         severity,
                "detection_method": "embedding",
                "scam_similarity":  round(max_scam, 3),
                "ham_similarity":   round(max_ham, 3),
            }

        except Exception as e:
            logger.error(f"EmbeddingDetector.detect error: {e}")
            return None
