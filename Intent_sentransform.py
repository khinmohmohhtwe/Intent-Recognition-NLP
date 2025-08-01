import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

# Load dataset
df = pd.read_csv('inten_sen_data.csv')

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

df['text'] = df['text'].apply(clean_text)

# Encode intents
le = LabelEncoder()
df['label'] = le.fit_transform(df['intent'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Load sentence transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text to embeddings
X_train_vec = embedder.encode(X_train.tolist())
X_test_vec = embedder.encode(X_test.tolist())

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))

# Get label names present in test set
labels_present = sorted(set(y_test))
target_names_present = le.inverse_transform(labels_present)

print("\nClassification Report:\n", classification_report(
    y_test, y_pred,
    labels=labels_present,
    target_names=target_names_present,
    zero_division=0
))

# Save model and encoder (embedder is loaded dynamically in app)
joblib.dump(model, 'intent_clf.joblib')
joblib.dump(le, 'label_encoder.joblib')
