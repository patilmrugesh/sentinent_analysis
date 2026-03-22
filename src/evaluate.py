import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

def evaluate_models():
    # Detect directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))
    RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
    
    # Path constants
    TEST_DATA_PATH = os.path.join(MODELS_DIR, "test_data_split.joblib")
    
    # Load test data and models
    if not os.path.exists(TEST_DATA_PATH):
        print(f"❌ Test data split not found at {TEST_DATA_PATH}. Training first...")
        from train import train_models
        train_models()
        
    X_test, y_test = joblib.load(TEST_DATA_PATH)
    lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    nb_model = joblib.load(os.path.join(MODELS_DIR, "naive_bayes.joblib"))
    
    models = {
        "Logistic Regression": lr_model,
        "Naive Bayes": nb_model
    }
    
    accuracies = {}
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Accuracy
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        
        print(f"--- Model Evaluation: {name} ---")
        print(f"Accuracy: {acc:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        
        # Save plots to results folder
        os.makedirs(RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(RESULTS_DIR, f"cm_{name.lower().replace(' ', '_')}.png"))
        plt.close()
        
    # Model Comparison Chart
    plt.figure(figsize=(8,5))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='viridis')
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy Score")
    plt.ylim(0, 1)
    
    # Save the plot
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison.png"))
    plt.close()
    
    print(f"\n✅ Evaluation results and plots saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate_models()
