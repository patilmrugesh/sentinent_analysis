import re
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER analyzer for emotional intensity
vader_analyzer = SentimentIntensityAnalyzer()

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean_text(text):
    """
    Cleans text for structural mining.
    - Preserves negation words (not, no) for sarcasm detection.
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    
    # Remove punctuation
    # text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Keep negation words as they are CRITICAL for sarcasm/sentiment reversal
    stop_words = set(stopwords.words('english'))
    negation_words = {'not', 'no', 'nor', 'neither', 'never', 'none', 'but', 'however'}
    stop_words = stop_words - negation_words
    
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(words)

def get_extra_features(text):
    """
    Extracts numerical features beyond simple words:
    - VADER scores (Negative, Neutral, Positive, Compound)
    - Punctuation (Exclamations/Question marks often signal sarcasm)
    """
    scores = vader_analyzer.polarity_scores(text)
    
    exclam_count = text.count('!')
    quest_count = text.count('?')
    
    return [
        scores['compound'], 
        scores['neg'], 
        scores['neu'], 
        scores['pos'],
        exclam_count,
        quest_count
    ]

def load_and_preprocess(filepath):
    """
    Loads and applies deep cleaning + sentiment scoring.
    """
    try:
        # Load and let pandas infer headers first
        df = pd.read_csv(filepath)
        
        # If headers don't match or it's raw without headers
        if not all(col in df.columns for col in ['text', 'sentiment']):
            df_no_header = pd.read_csv(filepath, header=None)
            if df_no_header.shape[1] >= 4:
                df_no_header.columns = ['id', 'entity', 'sentiment', 'text'][:df_no_header.shape[1]]
                df = df_no_header
            elif df_no_header.shape[1] == 2:
                df_no_header.columns = ['text', 'sentiment']
                df = df_no_header
    except Exception as e:
        df = pd.read_csv(filepath)

    # Filter only Positive & Negative
    if 'sentiment' in df.columns:
        df = df[df['sentiment'].astype(str).str.title().isin(['Positive', 'Negative'])]

    # Drop nulls
    df = df.dropna(subset=['text', 'sentiment'])

    print("⏳ Deep Preprocessing & Sentiment Scoring... (this takes a moment)")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Calculate extra numerical features
    extra_feats = df['text'].apply(get_extra_features).tolist()
    df['extra_features'] = extra_feats

    # remove empty cleaning results
    df = df[df['clean_text'].str.strip() != ""]

    return df