import pickle
import re
import string

# --- Text Cleaning Function ---
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# --- Load Model and Vectorizer ---
print("Loading model and vectorizer...")
try:
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Error: 'fake_news_model.pkl' or 'tfidf_vectorizer.pkl' not found.")
    print("Please make sure these files are in the same directory as this script.")
    exit()

# --- Prediction Function ---
def predict_news(text):
    processed_text = wordopt(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Real News" if prediction[0] == 1 else "Fake News"

# --- Interactive Mode ---
if __name__ == "__main__":
    print("\n--- Fake News Prediction ---")
    while True:
        user_input = input("\nEnter a news article (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting... Bye!")
            break
        prediction = predict_news(user_input)
        print("Prediction:", prediction)
