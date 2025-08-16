import streamlit as st
import joblib
import os
import re
import pandas as pd
from textblob import TextBlob

# ================= Page Configuration =================
st.set_page_config(
    page_title="MindGuard AI",
    page_icon="🧠",
    layout="wide"
)

# ================= Custom CSS =================
st.markdown("""
    <style>
        /* Keep your existing CSS */
        .highlight {
            background: rgba(255,255,0,0.3);
            padding: 2px 5px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# ================= Model Loading =================
@st.cache_resource
def load_model(path):
    if not os.path.exists(path): return None
    return joblib.load(path)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

model = load_model("artifacts/ml_tfidf_logreg.joblib")

# ================= Sidebar =================
st.sidebar.title("🧭 Navigation")
menu = st.sidebar.radio("Go to:", ["Home", "Analyze Text", "Batch Upload", "About"])

# ================= Home =================
if menu == "Home":
    # Centered Title and Subtitle
    st.markdown("""
        <div style="text-align: center;">
            <h1>Mental Health Prediction</h1>
            <p>On Your Social Media and Addressing Bias</p>
        </div>
    """, unsafe_allow_html=True)

    # Spacer
    st.write("")

    # Create three columns with a ratio that pushes the middle one to the center
    col1, col2, col3 = st.columns([2, 4, 2])

    # Place the image in the middle column
    with col2:
        st.image(
            "D://Mental_health_detection//assets//mental_health.png",
            width=600  # Set your desired small size
        )

# ================= Analyze Text =================
elif menu == "Analyze Text":
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    if model is None:
        st.error("❌ Model not found. Please train and save the model first.")
    else:
        user_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste your text here...",
            height=160
        )

        if st.button("🔍 Analyze Text"):
            if user_input.strip():
                with st.spinner("Analyzing text..."):
                    cleaned_input = preprocess_text(user_input)
                    prediction = model.predict([cleaned_input])[0]
                    probability = model.predict_proba([cleaned_input])[0]
                    confidence = probability.max()

                    # Extra Insights
                    word_count = len(user_input.split())
                    sentiment = TextBlob(user_input).sentiment.polarity
                    sentiment_label = "Positive 😀" if sentiment > 0 else "Negative 😞" if sentiment < 0 else "Neutral 😐"

                # Show result
                if prediction == "suicide":
                    st.markdown(f"""
                        <div class="result high-risk">
                            <div class="result-icon">⚠️</div>
                            <div class="result-title">High Risk Detected</div>
                            <div class="result-subtitle">Potential indicators of suicidal ideation found.</div>
                            <div class="confidence-badge">Confidence: {(confidence * 100):.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="result low-risk">
                            <div class="result-icon">✅</div>
                            <div class="result-title">Low Risk Detected</div>
                            <div class="result-subtitle">No strong signs of suicidal ideation found.</div>
                            <div class="confidence-badge">Confidence: {(confidence * 100):.1f}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Insights
                st.subheader("📊 Text Insights")
                st.write(f"**Word Count:** {word_count}")
                st.write(f"**Sentiment:** {sentiment_label}")

            else:
                st.warning("⚠️ Please enter some text to analyze.")

    st.markdown('</div>', unsafe_allow_html=True)

# ================= Batch Upload =================
elif menu == "Batch Upload":
    st.subheader("📂 Upload Text File or CSV")
    uploaded_file = st.file_uploader("Upload .txt or .csv", type=["txt", "csv"])

    if uploaded_file and model:
        if uploaded_file.name.endswith(".txt"):
            texts = [uploaded_file.read().decode("utf-8")]
        else:
            df = pd.read_csv(uploaded_file)
            if "text" not in df.columns:
                st.error("CSV must contain a 'text' column.")
            else:
                texts = df["text"].dropna().tolist()

        results = []
        for t in texts:
            cleaned = preprocess_text(t)
            pred = model.predict([cleaned])[0]
            prob = model.predict_proba([cleaned])[0].max()
            results.append({"text": t, "prediction": pred, "confidence": round(prob * 100, 2)})

        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

        st.download_button(
            "📥 Download Results",
            data=results_df.to_csv(index=False),
            file_name="analysis_results.csv",
            mime="text/csv"
        )

# ================= About =================
elif menu == "About":
    st.markdown("""
        ## ℹ️ About MindGuard AI
        - Built with **Logistic Regression + TF-IDF**  
        - Detects **potential suicidal ideation** in text  
        - Provides **confidence score & sentiment analysis**  
        - Supports **single text & batch analysis**  
    """)
    st.info("⚠️ This tool is for awareness only, not a replacement for professional help.")
