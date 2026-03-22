import joblib
import os
import numpy as np
import sys
from scipy.sparse import hstack

# Ensure src is in path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import clean_text, get_extra_features

class SentimentPredictor:
    def __init__(self, model_type="logistic_regression"):
        # Detect directories
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
        
        # We always use the main model trained in train.py
        self.model_path = os.path.join(MODELS_DIR, f"logistic_regression.joblib")
        self.tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
        
        if not os.path.exists(self.model_path) or not os.path.exists(self.tfidf_path):
            raise FileNotFoundError(f"❌ Models or Vectorizer not found at {MODELS_DIR}. Please run training first.")
            
        self.model = joblib.load(self.model_path)
        self.tfidf = joblib.load(self.tfidf_path)

    def predict(self, text):
        """
        Deep Prediction: 
        1. Clean text (Negation Aware)
        2. TF-IDF Triple-Gram Vectorization
        3. VADER Sentiment Scoring
        4. Stack & Predict
        """
        if not isinstance(text, str) or not text.strip():
            return {"sentiment": "Neutral", "confidence": 0.0}

        # Step 1: Clean
        cleaned = clean_text(text)
        
        # Step 2: TF-IDF
        X_tfidf = self.tfidf.transform([cleaned])
        
        # Step 3: VADER scores (Extra Features)
        X_extra = np.array([get_extra_features(text)])
        
        # Step 4: Combine
        X_combined = hstack([X_tfidf, X_extra])
        
        # Step 5: Inference
        prediction = self.model.predict(X_combined)[0]
        probabilities = self.model.predict_proba(X_combined)[0]
        confidence = max(probabilities)

        return {
            "sentiment": str(prediction),
            "confidence": round(float(confidence), 2)
        }

if __name__ == "__main__":
    # Internal Test for sarcasm: "It's so sad that I'm happy today"
    try:
        p = SentimentPredictor()
        test_case = "It is so sad that I am happy today"
        result = p.predict(test_case)
        print(f"--- Sarcasm Test ---\nText: {test_case}\nResult: {result}")
    except Exception as e:
        print(f"Please run train.py first. Error: {e}")
