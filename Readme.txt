#  Intent Recognition NLP App

This project predicts the **intent** behind user messages using two different approaches:

- ✅ TF-IDF + Logistic Regression (Traditional ML)
- ✅ SentenceTransformer + Logistic Regression (Modern Semantic Embeddings)

It includes two Streamlit apps where users can input messages and get real-time intent predictions.

---

##  Features

- Predicts user intent from short messages like "Change drop-off location"
- Supports both traditional and transformer-based NLP models
- Built with Python, Streamlit, scikit-learn, and sentence-transformers
- Label encoding for intent classes
- Preprocessed text with lowercase, punctuation removal, and whitespace normalization

---


## ⚙️ How It Works

### TF-IDF Approach
1. Text is vectorized using TF-IDF
2. Logistic Regression is used for classification

### SentenceTransformer Approach
1. Text is embedded using `all-MiniLM-L6-v2` SBERT model
2. Logistic Regression classifier predicts the intent based on semantic vectors

---

## 📊 Example Intents

| User Message                        | Predicted Intent     |
|------------------------------------|----------------------|
|I want to book a ride to the airport| BookRide             |
| Update my drop-off location	     | Change Destination   |
| I want to cancel my booking        | CancelBooking        |
| What's the price to go to Orchard? | PricingInquiry       |

---

## 💻 Run the App

### Install Dependencies

```bash
pip install -r requirements.txt


