# 🧠 MindGuard AI

**MindGuard AI** is an AI-powered web application that analyzes social media text for potential indicators of suicidal ideation. It provides real-time predictions, confidence scores, and additional text insights such as sentiment and word count.

---

## 🚀 Features

- Detects potential **suicidal ideation** in text.
- Provides a **confidence score** for the prediction.
- Displays **text insights**: word count and sentiment analysis.
- Supports **single text input** and **batch CSV uploads**.
- Clean, **mobile-friendly UI** with professional colors and responsive design.

---

## 📂 Dataset

The dataset used in this project is **not included** in the repository due to size and licensing restrictions.

You can download it from Kaggle:

[Kaggle Dataset Link]([YOUR_KAGGLE_DATASET_LINK_HERE](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch))

> After downloading, place the dataset in the `artifacts/` folder (or the folder specified in the code) to run the app successfully.

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/mindguard-ai.git
cd mindguard-ai
```
## ⚡ Usage

1. Run the Streamlit app:

```bash
python -m streamlit run app.py
```
2. Navigate through the sidebar menu:
- Analyze Text – Enter text manually to get predictions and insights.
- Batch Upload – Upload a .txt or .csv file containing multiple text entries for batch analysis.
- About – Learn more about the project, features, and methodology.
3. Text Analysis Output:
- Prediction: High Risk (⚠️) or Low Risk (✅)
- Confidence Score: Probability of prediction
- Text Insights: Word count and sentiment (Positive, Neutral, Negative)
## 💻 Tech Stack
- Python – Core programming language.
- Streamlit – Web application framework.
- Scikit-learn – Logistic Regression + TF-IDF for ML model.
- TextBlob – Sentiment analysis of text.
- Pandas & NumPy – Data manipulation and preprocessing.
## ⚠️ Disclaimer
This tool is for educational and awareness purposes only.
It is not a substitute for professional mental health support.
If you or someone you know is in crisis, seek help immediately from a qualified professional.
