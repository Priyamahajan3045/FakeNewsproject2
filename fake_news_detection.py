import pandas as pd
import numpy as np
import re
import string
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the datasets
print("Loading datasets...")
df_fake = pd.read_csv('Fake_news.csv')
df_true = pd.read_csv('True_news.csv')

# Add a target column to each dataframe
df_fake['label'] = 0  # 0 for Fake news
df_true['label'] = 1  # 1 for True news

# Combine datasets and drop unnecessary columns
df = pd.concat([df_fake, df_true], axis=0)
df.drop(['title', 'subject', 'date'], axis=1, inplace=True)

# Shuffle the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Text preprocessing function
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

# Apply the preprocessing function to the text column
print("Cleaning and preprocessing text...")
df['text'] = df['text'].apply(wordopt)

# Split data into training and testing sets
x = df['text']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Initialize and fit the TfidfVectorizer
print("Training TfidfVectorizer...")
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Initialize and train the PassiveAggressiveClassifier model
print("Training PassiveAggressiveClassifier model...")
PAC = PassiveAggressiveClassifier(max_iter=50)
PAC.fit(xv_train, y_train)

# Make predictions on the test set
print("Making predictions on test data...")
y_pred = PAC.predict(xv_test)

# Evaluate the model
score = accuracy_score(y_test, y_pred)
print("\n--- Model Evaluation ---")
print(f"Accuracy: {round(score*100, 2)}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer to .pkl files
print("\nSaving model and vectorizer...")
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(PAC, f)
    
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorization, f)

print("Files 'fake_news_model.pkl' and 'tfidf_vectorizer.pkl' saved successfully!")