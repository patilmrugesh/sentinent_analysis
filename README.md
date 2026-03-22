# 🐦 Tweet Sentiment Analysis System

Professional End-to-End Machine Learning Project to classify Tweet sentiments: **Positive** vs **Negative**.

---

## 📁 Project Structure

```
sentiment-analysis/
│── app/
│   └── app.py               # Streamlit-based UI
│── data/
│   └── tweets.csv           # Raw dataset (Twitter sentiments)
│   └── generate_tweets.py   # Script to generate sample data
│── models/
│   ├── logistic_regression.joblib  # Trained LR model
│   ├── naive_bayes.joblib          # Trained NB model
│   └── tfidf_vectorizer.joblib     # TF-IDF feature extractor
│── src/
│   ├── preprocess.py        # Text cleaning and preprocessing logic
│   ├── train.py             # Script for model training
│   ├── evaluate.py          # Script for performance evaluation
│   └── predict.py           # Logic for manual sentiment prediction
│── results/
│   ├── cm_logistic_regression.png   # Confusion Matrix for LR
│   ├── cm_naive_bayes.png           # Confusion Matrix for NB
│   └── model_comparison.png         # Accuracy comparison chart
│── requirements.txt         # Necessary libraries for the project
└── README.md                # Project documentation
```

---

## ⚙️ Features

1. **Text Preprocessing**: Automated cleaning including URL removal, mentions, hashtags, punctuation, and lemmatization.
2. **Feature Engineering**: Uses TF-IDF for numerical representation (max 5000 features).
3. **Multiple Models**: Compare Logistic Regression and Naive Bayes performance.
4. **Evaluation**: Automated accuracy calculations, classification reports, and confusion matrix visualizations.
5. **Interactive UI**: Built with Streamlit for real-time sentiment analysis with confidence scores.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data and Train Models
Run the training script which automatically handles data generation:
```bash
cd sentiment-analysis/src
python train.py
```

### 3. Evaluate Performance
```bash
python evaluate.py
```

### 4. Run Streamlit App
```bash
cd ../app
streamlit run app.py
```

---

## 🧠 Technologies Used
- **Python**: Core language
- **Scikit-learn**: Machine learning and preprocessing
- **Pandas/Numpy**: Data manipulation
- **Matplotlib/Seaborn**: Visualizations
- **Streamlit**: Web interface
- **NLTK**: Natural Language Processing
