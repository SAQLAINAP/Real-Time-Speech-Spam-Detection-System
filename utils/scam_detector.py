import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import logging

class ScamDetector:
    """
    A class for detecting potential scams in text using DistilBERT.
    """
    
    def __init__(self, model_name="AventIQ-AI/distilbert-spam-detection"):
        """
        Initialize the ScamDetector with a pre-trained model.
        
        Args:
            model_name (str): The name or path of the pre-trained model.
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()
    
    def load_model(self):
        """
        Load the tokenizer and model.
        """
        try:
            logging.info(f"Loading model: {self.model_name}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict(self, text):
        """
        Predict if the given text contains spam/scam content.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            dict: A dictionary containing the prediction results.
        """
        if not text or not isinstance(text, str):
            return {
                "is_spam": False,
                "prediction": "Unable to analyze empty text",
                "confidence": 0.0
            }
        
        try:
            self.model.eval()
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                pred_class = torch.argmax(probs).item()
                confidence = float(probs[0][pred_class].item())
            
            return {
                "is_spam": pred_class == 1,
                "prediction": "🚨 Potential Scam Detected!" if pred_class == 1 else "✅ Safe",
                "confidence": confidence
            }
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return {
                "is_spam": False,
                "prediction": "Error analyzing text",
                "confidence": 0.0
            }
    
    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information.
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "num_labels": self.model.config.num_labels if hasattr(self.model, 'config') else None
        }
