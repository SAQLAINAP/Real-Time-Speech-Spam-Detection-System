"""
Hotword-based Scam Detection Module with negation/context awareness
"""
import re
from typing import Dict, List, Any


# Words that indicate the hotword is being used in a safe/negated context
NEGATION_WORDS = frozenset([
    "not", "no", "never", "nor", "neither", "n't",
    "isn't", "aren't", "wasn't", "weren't", "don't",
    "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
    "secure", "safe", "safely", "secured", "protected",
    "legitimate", "verified", "genuine", "official",
])


class HotwordDetector:
    """
    Detects potential scams in text using hotword matching.
    Includes negation-awareness to reduce false positives.
    """

    def __init__(self, hotwords_severity: Dict[str, int] = None):
        self.hotwords_severity = hotwords_severity or {}
        self.patterns = {
            re.compile(r'\b' + re.escape(hotword) + r'\b', re.IGNORECASE): severity
            for hotword, severity in self.hotwords_severity.items()
        }

    # ── Negation check ────────────────────────────────────────────────────────
    def _is_negated(self, text: str, match_start: int, window: int = 55) -> bool:
        """
        Return True if a negation or safety word appears in the `window` characters
        immediately before the match position, suggesting the hotword is used safely.
        e.g. "your account is secure" — "account" is preceded by context that
        indicates safety, so we reduce its score.
        """
        preceding = text[max(0, match_start - window):match_start].lower()
        tokens = set(re.findall(r"\w+|n't", preceding))
        return bool(NEGATION_WORDS.intersection(tokens))

    # ── Main detection ────────────────────────────────────────────────────────
    def detect(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "is_spam": False,
                "confidence": 0.0,
                "severity": 0,
                "category": "safe",
                "matches": []
            }

        matches = []
        for pattern, base_severity in self.patterns.items():
            for match in pattern.finditer(text):
                effective_severity = base_severity

                # Reduce severity if context suggests negation / safe usage
                if self._is_negated(text, match.start()):
                    effective_severity = max(1, base_severity // 2)

                matches.append({
                    "hotword": match.group(0),
                    "severity": effective_severity,
                    "position": match.span(),
                    "negated": effective_severity < base_severity
                })

        if not matches:
            return {
                "is_spam": False,
                "confidence": 0.0,
                "severity": 0,
                "category": "safe",
                "matches": []
            }

        max_severity = max(m["severity"] for m in matches)
        avg_severity = sum(m["severity"] for m in matches) / len(matches)
        severity = min(10, round(max_severity * 0.7 + avg_severity * 0.3))

        confidence = min(1.0, (len(matches) * 0.1) + (severity / 10.0) * 0.9)

        if severity <= 3:
            category, is_spam = "safe", False
        elif severity <= 5:
            category, is_spam = "neutral", False
        elif severity <= 7:
            category, is_spam = "suspicious", True
        else:
            category, is_spam = "highly_suspicious", True

        return {
            "is_spam": is_spam,
            "confidence": round(confidence, 2),
            "severity": severity,
            "category": category,
            "matches": matches
        }

    def get_hotwords_info(self) -> Dict[str, Any]:
        return {
            "count": len(self.hotwords_severity),
            "sample": list(self.hotwords_severity.keys())[:5] if self.hotwords_severity else []
        }
