import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import re

# Load dataset
df = pd.read_csv('intent_recog_data.csv')

# Clean text
def clean_text(text):
    text = text.lower()                            # Lowercase
    text = re.sub(r'[^\w\s]','',text)  # remove punctuation
    text = re.sub(r'\s+',' ',text)     # remove extra space
    return text

df['text'] = df['text'].apply(clean_text)

# Encode intents  # we must convert them into numbers
le = LabelEncoder()
df['label'] = le.fit_transform(df['intent'])

# Split data into Train and Test

X_train,X_test,y_train,y_test = train_test_split(df['text'],df['label'],test_size=0.2,random_state=42)

# Feature Extraction (Text Vectorization)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model Training
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

# Initialize the model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_vec,y_train)

# Predict on test data
y_pred = model.predict(X_test_vec)

# Evaluate
print("Accuracy:",accuracy_score(y_test,y_pred))
#print("\nClassification Report:\n",classification_report(y_test,y_pred,target_names=le.classes_))



# Step 1: Find only the labels present in y_test
labels_present = sorted(set(y_test))

# Step 2: Map those back to original class names using the label encoder
target_names_present = le.inverse_transform(labels_present)

# Step 3: Generate classification report with labels specified
print("\nClassification Report:\n", classification_report(
    y_test,
    y_pred,
    labels=labels_present,
    target_names=target_names_present,
    zero_division=0  # or 1 depending on your reporting preference
))

import joblib

# Save model, vectorizer, label encoder
joblib.dump(model, 'intent_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(le, 'label_encoder.pkl')