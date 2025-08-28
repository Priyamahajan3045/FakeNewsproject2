import streamlit as st
import pickle

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Article:")

if st.button("Predict"):
    if user_input.strip() != "":
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]
        if prediction == 0:
            st.error("🔴 Fake News Detected!")
        else:
            st.success("🟢 Real News Detected!")
    else:
        st.warning("⚠️ Please enter some text!")
