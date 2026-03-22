import streamlit as st
import joblib
import os
import sys

# Adding src to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import SentimentPredictor

# Streamlit UI Configuration
st.set_page_config(page_title="Tweet Sentiment Analysis", layout="wide")

# Theme / CSS
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
    }
    .stButton > button {
        border-radius: 20px;
        background-color: #4CAF50;
        color: white;
    }
    .sentiment-box {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.title("🐦 Tweet Sentiment Analysis System")
st.write("Professional Machine Learning project for predicting tweet sentiment: Positive vs Negative.")
st.divider()

# Sidebar for analysis/options
with st.sidebar:
    st.header("Project Overview")
    st.info("Uses TF-IDF Vectorization and Logistic Regression for analysis. "
            "Model trained on curated sentiment data.")
    
    st.subheader("Model Configuration")
    model_choice = st.selectbox("Choose Model", ["Logistic Regression", "Naive Bayes"])
    model_file = "logistic_regression" if model_choice == "Logistic Regression" else "naive_bayes"

# Prediction Section
st.subheader("Predict Sentiment")
user_input = st.text_input("Enter your tweet text here:", placeholder="Ex: Having an amazing day at the beach!")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        try:
            predictor = SentimentPredictor(model_file)
            result = predictor.predict(user_input)
            
            sentiment = result["sentiment"]
            score = result["confidence"]
            
            # Display based on prediction
            if sentiment == "Positive":
                st.success(f"### Result: **{sentiment}** ✅")
                st.write(f"Confidence Score: `{score*100}%`")
                # st.balloons() # Removed balloons to keep UI cleaner for feedback
            elif sentiment == "Negative":
                st.error(f"### Result: **{sentiment}** ❌")
                st.write(f"Confidence Score: `{score*100}%`")
            else:
                st.info(f"### Result: **{sentiment}** ℹ️")

            # --- Feedback Mechanism for Self-Learning ---
            st.divider()
            st.subheader("Was this prediction correct?")
            
            col_fb1, col_fb2 = st.columns(2)
            
            with col_fb1:
                if st.button("👍 Yes, Correct"):
                    st.toast("Thank you for your feedback! Keeping the model sharp.", icon="✅")
            
            with col_fb2:
                if st.button("👎 No, Incorrect"):
                    st.session_state.show_feedback_form = True

            if st.session_state.get('show_feedback_form'):
                with st.form("feedback_form"):
                    st.info("Help the model learn! Tell us what the sentiment should have been.")
                    correct_sentiment = st.radio("Correct Sentiment:", ["Positive", "Negative"])
                    submitted = st.form_submit_button("Submit Correction")
                    
                    if submitted:
                        # Save feedback to a CSV for retraining
                        FEEDBACK_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'user_feedback.csv'))
                        feedback_data = pd.DataFrame([{
                            'text': user_input,
                            'sentiment': correct_sentiment,
                            'timestamp': pd.Timestamp.now()
                        }])
                        
                        # Append to file
                        if not os.path.exists(FEEDBACK_FILE):
                            feedback_data.to_csv(FEEDBACK_FILE, index=False)
                        else:
                            feedback_data.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
                        
                        st.success("Feedback saved! The model will learn from this in the next training cycle.")
                        print(f"✅ Feedback Added: '{user_input[:50]}...' -> Corrected to: {correct_sentiment}")
                        st.session_state.show_feedback_form = False

        except FileNotFoundError:
            st.error("Error: Trained model files not found. Please run Training first.")

# --- Self-Learning / Retraining Control ---
st.sidebar.divider()
st.sidebar.subheader("🧠 Self-Learning Center")
FEEDBACK_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'user_feedback.csv'))

if os.path.exists(FEEDBACK_FILE):
    fb_df = pd.read_csv(FEEDBACK_FILE)
    num_feedbacks = len(fb_df)
    st.sidebar.write(f"Collected feedbacks: **{num_feedbacks}**")
    
    if num_feedbacks >= 5: # Minimal threshold for retraining
        st.sidebar.warning("New patterns detected. Retrain to improve accuracy!")
        if st.sidebar.button("Retrain Model Now"):
            with st.status("🧠 Model is learning from new data...", expanded=True) as status:
                st.write("Merging data...")
                # Call retraining script
                import subprocess
                TRAIN_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'train.py'))
                result = subprocess.run([sys.executable, TRAIN_SCRIPT], capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.write(result.stdout)
                    status.update(label="✨ Model updated successfully!", state="complete", expanded=False)
                    st.sidebar.success("Model successfully learned from your feedback!")
                else:
                    st.write(result.stderr)
                    status.update(label="❌ Training failed.", state="error")
    else:
        st.sidebar.info(f"Need {5 - num_feedbacks} more feedbacks to trigger retraining.")
else:
    st.sidebar.info("No feedback collected yet.")

# Visualize evaluation results if available
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
COMPARISON_PLOT = os.path.join(RESULTS_DIR, "model_comparison.png")

if os.path.exists(COMPARISON_PLOT):
    st.divider()
    st.subheader("Model Performance Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(COMPARISON_PLOT, caption="Accuracy Comparison")
    with col2:
        cm_file = os.path.join(RESULTS_DIR, f"cm_{model_file}.png")
        if os.path.exists(cm_file):
            st.image(cm_file, caption=f"Confusion Matrix ({model_choice})")

st.markdown("---")
st.caption("Sentinent Analysis Project with Self-Learning Capabilities")
