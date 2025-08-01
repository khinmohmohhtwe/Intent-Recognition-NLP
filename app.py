import streamlit as st
import joblib
import os

st.set_page_config(page_title="Intent Recognition", page_icon="💬")
st.title("🚖 Intent Recognition App")
st.markdown("This app predicts the **intent** behind user input.")

# Load model and assets
try:
    model = joblib.load('intent_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    st.error(f"❌ Error loading model or assets: {e}")
    st.stop()

# Input section
user_input = st.text_input("Enter a user query:")

if user_input:
    try:
        input_vec = vectorizer.transform([user_input])
        pred = model.predict(input_vec)
        pred_label = label_encoder.inverse_transform([pred])[0]

        st.success(f"🎯 **Predicted Intent:** `{pred_label}`")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
