import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from preprocess import load_and_preprocess

def train_models():
    print("🚀 Self-Learning Model: Training process started...")
    # Detect directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
    MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
    
    # Path to the dataset
    data_path = os.path.join(DATA_DIR, "tweets.csv")
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Dataset not found at: {data_path}")
        return
        
    print(f"📂 Loading and Preprocessing dataset from {data_path}...")
    df = load_and_preprocess(data_path)
    
    # Check for user feedback and merge for self-learning
    feedback_path = os.path.join(DATA_DIR, "user_feedback.csv")
    if os.path.exists(feedback_path):
        print(f"🔄 Self-Learning: Found user feedback at {feedback_path}. Incorporating into training...")
        fb_df = load_and_preprocess(feedback_path)
        if not fb_df.empty:
            # We can optionally boost the weight of feedback data if the base dataset is very large
            # For now, we simply append it.
            df = pd.concat([df, fb_df], ignore_index=True).drop_duplicates(subset=['text'])
            print(f"📊 Feedback assimilated. Total unique training samples: {len(df)}")

    print(f"🚀 Training on {len(df)} total samples...")
    
    # Feature Engineering (TF-IDF with Bi-grams and Tri-grams)
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,3))
    X_tfidf = tfidf.fit_transform(df['clean_text'])
    
    # Numerical features (VADER scores + punctuation)
    X_extra = np.array(df['extra_features'].tolist())
    
    # Combine (Stack) Features
    X_combined = hstack([X_tfidf, X_extra])
    y = df['sentiment']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    print("🧠 Building Ensemble Model...")
    
    # 1. Logistic Regression (Balanced weights)
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # 2. Naive Bayes (Standard) - Requires non-negative features
    # VADER compound score is -1 to 1, others are >= 0. We shift compound score by +1.
    # Since we can't easily modify a sparse matrix in place for just some columns,
    # we'll just use a simpler approach or skip VADER for NB if it's too complex.
    # Actually, for the ensemble to be robust, let's just use Logistic Regression 
    # as it handles signed features perfectly.
    
    nb_model = MultinomialNB()
    # To keep it simple and working: we only train NB on the TF-IDF part (always positive)
    # or just use LR for the prediction in the predictor.
    # Let's train NB on a simplified positive version of features.
    X_train_pos = X_train.copy()
    if hasattr(X_train_pos, "data"):
        X_train_pos.data[X_train_pos.data < 0] = 0 # Clamp negative VADER scores to 0 for NB
    
    nb_model.fit(X_train_pos, y_train)
    
    # Create models folder
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save artifacts
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(nb_model, os.path.join(MODELS_DIR, "naive_bayes.joblib"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump((X_test, y_test), os.path.join(MODELS_DIR, "test_data_split.joblib"))
    
    print("✅ Self-Learning Model: Training completed successfully!")
    
    # Auto-run evaluation to update plots
    print("📊 Running evaluation to update performance charts...")
    try:
        from evaluate import evaluate_models
        evaluate_models()
    except Exception as e:
        print(f"⚠️ Warning: Evaluation failed: {e}")

if __name__ == "__main__":
    train_models()
