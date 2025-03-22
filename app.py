from flask import Flask, render_template, request, jsonify
import os
import re
import string
import math
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)

# Download NLTK resources
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_resources()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return filtered_tokens

def get_shingles(tokens, k=3):
    return [' '.join(tokens[i:i+k]) for i in range(len(tokens) - k + 1)]

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def cosine_similarity(counter1, counter2):
    terms = set(counter1).union(counter2)
    dot_product = sum(counter1.get(k, 0) * counter2.get(k, 0) for k in terms)
    magnitude1 = math.sqrt(sum(val ** 2 for val in counter1.values()))
    magnitude2 = math.sqrt(sum(val ** 2 for val in counter2.values()))
    return dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0

def analyze_plagiarism(suspect_text, source_texts):
    results = []
    suspect_tokens = preprocess_text(suspect_text)
    suspect_shingles = set(get_shingles(suspect_tokens))
    suspect_counter = Counter(suspect_tokens)
    
    for source_name, source_text in source_texts.items():
        source_tokens = preprocess_text(source_text)
        source_shingles = set(get_shingles(source_tokens))
        source_counter = Counter(source_tokens)

        jaccard_sim = jaccard_similarity(suspect_shingles, source_shingles)
        cosine_sim = cosine_similarity(suspect_counter, source_counter)
        common_shingles = suspect_shingles.intersection(source_shingles)

        results.append({
            'source_name': source_name,
            'jaccard_similarity': round(jaccard_sim * 100, 2),
            'cosine_similarity': round(cosine_sim * 100, 2),
            'common_phrases': list(common_shingles)[:10],
            'overall_similarity': round((jaccard_sim + cosine_sim) * 50, 2),
        })

    results.sort(key=lambda x: x['overall_similarity'], reverse=True)
    return results


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(force=True)
    suspect_text = data.get('suspect_text', '')
    source_texts = data.get('source_texts', {})

    plagiarism_results = analyze_plagiarism(suspect_text, source_texts)
  
    return jsonify({
        'plagiarism_results': plagiarism_results,
        
    })

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, port=5000)