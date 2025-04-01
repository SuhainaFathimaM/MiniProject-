import string
import pickle
import nltk
from nltk.corpus import stopwords
from flask import Flask, render_template, request
import pytesseract
from PIL import Image
import fitz  # PyMuPDF

nltk.download("stopwords")

# Load the model and TF-IDF vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

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

# Function to extract text from a PDF file using PyMuPDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to extract text from an image using Tesseract OCR
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for processing the uploaded files and detecting plagiarism
@app.route('/detect', methods=['POST'])
def detect_plagiarism():
    result = None
    input_text = None
    
    if 'text' in request.form:
        # Process text input from the textarea
        input_text = request.form['text']
        result = detect(input_text)
    
    if 'pdf' in request.files:
        # Process PDF file upload
        pdf_file = request.files['pdf']
        pdf_text = extract_text_from_pdf(pdf_file)
        result = detect(pdf_text)
    
    if 'image' in request.files:
        # Process image file upload
        image_file = request.files['image']
        image_text = extract_text_from_image(image_file)
        result = detect(image_text)
    
    return render_template('index.html', result=result, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
