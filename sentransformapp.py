import streamlit as st
from sentence_transformers import SentenceTransformer
import joblib

# Load model, label encoder, and embedder
@st.cache_resource
def load_models():
    clf = joblib.load("intent_clf.joblib")            # Trained classifier
    le = joblib.load("label_encoder.joblib")          # Label encoder
    embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Embedder model
    return clf, le, embedder

clf, le, embedder = load_models()

st.title("🚖 Intent Recognition")
st.write("Enter a message, and we'll predict its intent using a sentence embedding model!")

# Text input
user_input = st.text_input("💬 Type your message:")

# Predict and display
if user_input:
    input_vec = embedder.encode([user_input])
    pred_label = clf.predict(input_vec)[0]
    intent = le.inverse_transform([pred_label])[0]
    st.success(f"🔍 Predicted Intent: **{intent}**")
