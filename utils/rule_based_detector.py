import re
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RuleBasedScamDetector:
    """
    A class for detecting potential scams using rule-based pattern matching.
    This serves as a fallback when ML models or API calls aren't available.
    """
    
    def __init__(self):
        """
        Initialize the rule-based scam detector with common scam indicators and their weights.
        """
        # Common scam indicators and their weights
        self.scam_indicators = {
            # Urgency patterns
            "urgent": 0.4,
            "emergency": 0.4,
            "act now": 0.4,
            "immediate action": 0.5,
            "limited time offer": 0.4,
            "expir(es|ing|ed)": 0.4,
            "do not ignore": 0.4,
            "respond immediately": 0.5,
            "time (is )?running out": 0.4,
            
            # Account security patterns
            "account.+suspend": 0.6,
            "account.+blocked": 0.6,
            "verify.+identity": 0.5,
            "verify.+account": 0.5,
            "unusual activity": 0.5,
            "suspicious activity": 0.5,
            "security breach": 0.5,
            "unauthorized access": 0.6,
            
            # Personal information patterns
            "your.+password": 0.6,
            "confirm.+details": 0.5,
            "update.+information": 0.5,
            "credit card": 0.5,
            "social security": 0.7,
            "bank.+details": 0.6,
            "account details": 0.5,
            
            # Financial scam patterns
            "lottery": 0.5,
            "won.+prize": 0.5,
            "inheritance": 0.5,
            "million.+dollars": 0.6,
            "unclaimed.+funds": 0.5,
            "money transfer": 0.5,
            "wire transfer": 0.6,
            "investment opportunity": 0.6,
            "high return": 0.5,
            "guarantee.+profit": 0.7,
            "double your money": 0.7,
            "risk.free": 0.5,
            
            # Common scam origins/contexts
            "prince": 0.4,
            "nigerian": 0.4,
            "foreigner": 0.3,
            "overseas": 0.3,
            "foreign country": 0.3,
            
            # Action inducements
            "click.+link": 0.5,
            "follow.+instructions": 0.3,
            "download.+attachment": 0.6,
            "claim your": 0.3,
            "call this number": 0.4,
            "contact us immediately": 0.5,
            
            # Gift card scams
            "gift card": 0.5,
            "itunes card": 0.7,
            "google play card": 0.7,
            "steam card": 0.7,
            "amazon gift card": 0.7,
            
            # Tax scams
            "tax refund": 0.5,
            "irs.+calling": 0.7,
            "tax.+overdue": 0.6,
            "unpaid taxes": 0.6,
            
            # Tech support scams
            "computer.+virus": 0.6,
            "microsoft.+support": 0.5,
            "apple.+support": 0.5,
            "technical support": 0.4,
            "remote.+access": 0.6,
            
            # Debt collection scams
            "overdue payment": 0.5,
            "debt collection": 0.4,
            "final notice": 0.5,
            "legal action": 0.5
        }
        
        # Compile regular expressions for faster matching
        self.compiled_patterns = {pattern: re.compile(pattern, re.IGNORECASE) for pattern in self.scam_indicators.keys()}
    
    def detect(self, text: str) -> Dict[str, Any]:
        """
        Detect scam patterns in text using rule-based approach.
        
        Args:
            text (str): The text to analyze for scam indicators.
            
        Returns:
            Dict[str, Any]: Results containing is_spam flag, confidence score, and prediction.
        """
        if not text or not isinstance(text, str):
            return {
                "is_spam": False,
                "prediction": "Unable to analyze empty text",
                "confidence": 0.0
            }
        
        # Calculate a spam score based on the presence of indicators
        spam_score = 0.0
        found_indicators = 0
        matched_patterns = []
        
        for pattern, regex in self.compiled_patterns.items():
            if regex.search(text):
                weight = self.scam_indicators[pattern]
                spam_score += weight
                found_indicators += 1
                matched_patterns.append(pattern)
        
        # Normalize the score
        if found_indicators > 0:
            # Increase the score if multiple indicators are found (pattern recognition)
            if found_indicators >= 3:
                # Multiple indicators are a stronger signal
                spam_score = min(1.0, spam_score * 1.5)
            else:
                spam_score = min(1.0, spam_score)
        
        is_spam = spam_score > 0.6  # Threshold for spam classification
        
        logger.debug(f"Scam detection result: score={spam_score}, indicators={found_indicators}, patterns={matched_patterns}")
        
        # Format prediction text
        prediction_text = "🚨 Potential Scam Detected!" if is_spam else "✅ Safe"
        
        return {
            "is_spam": is_spam,
            "prediction": prediction_text,
            "confidence": spam_score,
            "matched_patterns": matched_patterns
        }