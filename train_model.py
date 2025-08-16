import pandas as pd
import re
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ===== 1. Load Data from Kaggle =====
# Assumes you have downloaded the dataset and renamed it to 'mental_health.csv'
DATA_PATH = "data/mental_health.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ CSV file not found at {DATA_PATH}. Please download it from Kaggle.")

df = pd.read_csv(DATA_PATH)

# ===== 2. Prepare and Clean Data =====
# The dataset might have an unnecessary first column, let's drop it if it exists.
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Standardize column names
df.rename(columns={'class': 'label'}, inplace=True)

# Drop any rows with missing text
df.dropna(subset=['text'], inplace=True)

print("✅ Data loaded successfully.")
print("Class distribution from the new dataset:")
print(df['label'].value_counts())


# ===== 3. Preprocessing Function =====
def preprocess_text(text):
    """Cleans text by removing punctuation, numbers, and converting to lowercase."""
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text) # Remove text in brackets
    text = re.sub(r'\W', ' ', text) # Remove all non-word characters
    text = re.sub(r'\s+', ' ', text) # Replace multiple spaces with a single space
    return text

df['text_cleaned'] = df['text'].apply(preprocess_text)


# ===== 4. Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    df["text_cleaned"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

# ===== 5. Create and Train the Pipeline =====
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=7500)), # Increased max_features for a larger dataset
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear')) # Using a robust solver
])

print("\n⏳ Training model on the new Kaggle dataset...")
pipeline.fit(X_train, y_train)
print("✅ Model training complete.")


# ===== 6. Evaluate Model =====
y_pred = pipeline.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))


# ===== 7. Save Confusion Matrix =====
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
plt.savefig(CONFUSION_MATRIX_PATH)
print(f"📈 Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


# ===== 8. Save Model =====
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "ml_tfidf_logreg.joblib")
joblib.dump(pipeline, MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
