from flask import Flask, render_template, request
import string
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download("stopwords")

# Initialize the Flask app
app = Flask(__name__)

# Load the model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

def preprocess_text(text):
    """Preprocess input text by removing punctuation, converting to lowercase, and removing stopwords."""
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words("english"))
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text

def detect(input_text):
    """Detect plagiarism in the given input text."""
    input_text = preprocess_text(input_text)
    vectorized_text = tfidf_vectorizer.transform([input_text])
    result = model.predict(vectorized_text)
    return "Plagiarism Detected" if result[0] == 1 else "No Plagiarism"

# Define route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define route for handling form submission
@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    if request.method == 'POST':
        input_text = request.form['text']
        result = detect(input_text)
        return render_template('index.html', result=result, input_text=input_text)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
