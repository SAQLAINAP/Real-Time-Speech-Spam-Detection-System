"""
Hotword-based Scam Detection Module
"""
import re
from typing import Dict, List, Tuple, Any


class HotwordDetector:
    """
    A class for detecting potential scams in text using hotword matching.
    Uses a dictionary of hotwords/phrases and their severity scores.
    """

    def __init__(self, hotwords_severity: Dict[str, int] = None):
        """
        Initialize the HotwordDetector with a dictionary of hotwords and their severity scores.
        
        Args:
            hotwords_severity (Dict[str, int]): Dictionary with hotwords as keys and severity scores as values.
        """
        # Default hotwords if none provided
        self.hotwords_severity = hotwords_severity or {}
        
        # Create case-insensitive patterns for each hotword
        self.patterns = {
            re.compile(r'\b' + re.escape(hotword) + r'\b', re.IGNORECASE): severity
            for hotword, severity in self.hotwords_severity.items()
        }
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect hotwords in text and calculate overall severity.
        
        Args:
            text (str): The text to analyze for hotwords.
            
        Returns:
            Dict[str, Any]: Results containing:
                - is_spam (bool): Whether the text is considered spam
                - confidence (float): Confidence score (0.0-1.0)
                - severity (int): Overall severity score (0-10)
                - category (str): Category based on severity ('safe', 'neutral', 'suspicious', 'highly_suspicious')
                - matches (List[Dict]): List of detected hotwords with details
        """
        if not text or len(text.strip()) == 0:
            return {
                "is_spam": False,
                "confidence": 0.0,
                "severity": 0,
                "category": "safe",
                "matches": []
            }
        
        # Find all hotword matches
        matches = []
        for pattern, severity in self.patterns.items():
            for match in pattern.finditer(text):
                matches.append({
                    "hotword": match.group(0),
                    "severity": severity,
                    "position": match.span()
                })
        
        # Calculate overall severity
        if not matches:
            severity = 0
            confidence = 0.0
            category = "safe"
            is_spam = False
        else:
            # Get max severity
            max_severity = max(match["severity"] for match in matches)
            # Calculate weighted average severity based on severity scores
            total_severity = sum(match["severity"] for match in matches)
            severity = min(10, round(max_severity * 0.7 + (total_severity / len(matches)) * 0.3))
            
            # Calculate confidence based on number of matches and their severity
            confidence = min(1.0, (len(matches) * 0.1) + (severity / 10.0) * 0.9)
            
            # Determine category
            if severity <= 3:
                category = "safe"
                is_spam = False
            elif severity <= 5:
                category = "neutral"
                is_spam = False
            elif severity <= 7:
                category = "suspicious"
                is_spam = True
            else:
                category = "highly_suspicious"
                is_spam = True
        
        return {
            "is_spam": is_spam,
            "confidence": round(confidence, 2),
            "severity": severity,
            "category": category,
            "matches": matches
        }
    
    def get_hotwords_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded hotwords.
        
        Returns:
            Dict[str, Any]: Information about the hotwords.
        """
        return {
            "count": len(self.hotwords_severity),
            "sample": list(self.hotwords_severity.keys())[:5] if self.hotwords_severity else []
        }