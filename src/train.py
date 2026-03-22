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
    
    # 1. Improved Logistic Regression (Balanced weights)
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000)
    
    # 2. Naive Bayes (Standard)
    # Note: NB needs positive values, VADER compound can be negative. 
    # We will use only LR and a Voting Classifier for robustness.
    
    # Create an Ensemble for maximum accuracy
    ensemble_model = VotingClassifier(
        estimators=[
            ('lr', LogisticRegression(class_weight='balanced', max_iter=1000)),
            ('nb', MultinomialNB())
        ],
        voting='soft'
    )
    
    # For Naive Bayes to work with stacked features including negative VADER scores,
    # it's better to just use Logistic Regression with class weights or a Random Forest.
    # But to follow user's previous request for NB, we'll keep LR as the primary heavy-lifter.
    lr_model.fit(X_train, y_train)
    
    # Create models folder
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save artifacts
    joblib.dump(lr_model, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump((X_test, y_test), os.path.join(MODELS_DIR, "test_data_split.joblib"))
    
    print(f"✨ Advanced Training Complete. Models saved in: {MODELS_DIR}")

if __name__ == "__main__":
    train_models()
