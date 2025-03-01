from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
import time

class SpamDetector:
    def __init__(self, model_name="AventIQ-AI/distilbert-spam-detection", device=None):
        """
        Initialize spam detector with DistilBERT model
        
        Args:
            model_name: HuggingFace model name
            device: Compute device (cuda or cpu)
        """
        print(f"Loading spam detection model: {model_name}")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
        print("Spam detector initialized")
    
    def predict(self, text):
        """
        Predict if text is spam or not
        
        Args:
            text: Text to classify
            
        Returns:
            dict: Prediction result with class and probability
        """
        self.model.eval()  # Set to evaluation mode
        
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()
        
        # Return prediction details
        result = {
            "text": text,
            "is_spam": pred_class == 1,
            "prediction": "Spam" if pred_class == 1 else "Not Spam",
            "confidence": confidence,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
    
    def save_result(self, result, save_dir="spam_results"):
        """
        Save prediction result to file
        
        Args:
            result: Prediction result dict
            save_dir: Directory to save results
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_dir, f"result_{timestamp}.txt")
        
        with open(filename, "w") as f:
            f.write(f"Text: {result['text']}\n")
            f.write(f"Prediction: {result['prediction']}\n")
            f.write(f"Confidence: {result['confidence']:.4f}\n")
            f.write(f"Timestamp: {result['timestamp']}\n")
        
        print(f"Result saved to {filename}")
        return filename


if __name__ == "__main__":
    # Demo usage
    detector = SpamDetector()
    
    test_messages = [
        "Congratulations! You have won a lottery of $1,000,000. Claim now!",
        "Hey, are we still meeting for dinner tonight?"
    ]
    
    for msg in test_messages:
        result = detector.predict(msg)
        print(f"Message: {msg}")
        print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']:.4f})")
        print("-" * 50)