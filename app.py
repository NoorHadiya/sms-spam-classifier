import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load or fit TF-IDF vectorizer
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
except FileNotFoundError:
    st.error("Vectorizer not found. Please train the model first.")
    st.stop()

# Load or fit model
# Load or fit model
try:
    model = pickle.load(open('model.pkl', 'rb'))
    model_fitted = hasattr(model, "classes_")
    # Check if the model is fitted
    if not hasattr(model, "classes_"):

        # Default data for fitting the model
        X_default = [
            "Sorry, I'll call later in meeting.",            # Example non-spam message
            "free offer, click here"  # Example spam message
        ]
        y_default = [0, 1]  # Labels corresponding to non-spam and spam messages respectively
        X_default_tfidf = tfidf.transform(X_default)

        # Set force_alpha attribute to False
        model.force_alpha = False

        # Fit the model with default data
        model.fit(X_default_tfidf, y_default)
except FileNotFoundError:
    st.error("Model not found. Please train the model first.")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    # Predict
    result = model.predict(vector_input)[0]
    # Display result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
