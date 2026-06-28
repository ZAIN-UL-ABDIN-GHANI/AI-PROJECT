# 📄 Text Plagiarism Analysing Tool

> An AI-powered web application that detects plagiarism between documents using NLP similarity algorithms — built with Flask, NLTK, and vanilla JavaScript.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.2.3-black?style=flat&logo=flask)
![NLTK](https://img.shields.io/badge/NLTK-3.8.1-green?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 🧠 What It Does

This tool allows users to paste a suspect document and compare it against one or more source texts. It uses three NLP-based similarity metrics to detect potential plagiarism and highlights common matching phrases — all in real time through a clean, responsive web interface.

---

## ✨ Features

- 🔍 **Multi-source comparison** — compare one text against multiple sources simultaneously
- 📊 **Three similarity metrics** — Jaccard, Cosine, and an overall combined score
- 💬 **Common phrase extraction** — highlights overlapping 3-word shingles between texts
- 🎨 **Visual similarity meter** — animated progress bar with colour-coded severity levels
- ➕ **Dynamic source management** — add or remove source documents on the fly
- 📱 **Responsive UI** — works on desktop and mobile

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| NLP | NLTK (tokenization, stopword removal) |
| Algorithms | Jaccard Similarity, Cosine Similarity, Shingling |
| Frontend | HTML, CSS, Vanilla JavaScript |
| Icons | Font Awesome 6 |

---

## 🔬 How the Algorithms Work

### 1. Text Preprocessing
Every input text goes through a pipeline before comparison:
- Lowercasing
- Punctuation removal
- Word tokenization (via NLTK)
- Stopword filtering (common words like "the", "is" are removed)

### 2. Shingling
Tokens are converted into overlapping 3-word sequences (shingles). For example:
```
"machine learning is powerful" → ["machine learning is", "learning is powerful"]
```
Shingles capture phrase-level similarity, not just word overlap.

### 3. Jaccard Similarity
Measures the ratio of shared shingles to total shingles across both texts:
```
Jaccard = |A ∩ B| / |A ∪ B|
```
High Jaccard = many common phrases.

### 4. Cosine Similarity
Compares word frequency vectors using the dot product formula:
```
Cosine = (A · B) / (|A| × |B|)
```
High Cosine = similar word distribution and vocabulary.

### 5. Overall Similarity Score
```
Overall = (Jaccard + Cosine) × 50
```
Blends both metrics into a single 0–100% score.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/text-plagiarism-tool.git
cd text-plagiarism-tool

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

The app will start at **http://localhost:5000**

---

## 📁 Project Structure

```
text-plagiarism-tool/
│
├── app.py                  # Flask backend + NLP logic
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Frontend UI
└── README.md
```

---

## 📦 Dependencies

```
Flask==2.2.3
nltk==3.8.1
numpy==2.2.4
Werkzeug==2.2.3
Jinja2==3.1.6
```

Full list in `requirements.txt`.

---

## 🖥️ Usage

1. Open the app at `http://localhost:5000`
2. Paste the **suspect text** in the left panel
3. Paste one or more **source texts** in the right panel (add more with the **+ Add Source** button)
4. Click **Analyze Text**
5. View results — similarity scores, severity badge, animated meter, and common phrases

### Similarity Severity Levels

| Score | Level | Colour |
|-------|-------|--------|
| 0–40% | Low Similarity | 🟢 Green |
| 40–70% | Moderate Similarity | 🟡 Yellow |
| 70–100% | High Similarity | 🔴 Red |

---

## 📸 Screenshots

> *(Add screenshots of the UI here after running the app)*

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 👤 Author

**Zain Ul Abdin Ghani**  
AI Engineer | CS Graduate — SZABIST University  
📧 zainulaabdinghani15@gmail.com  
🌐 [zainulabdinghani.vercel.app](https://zainulabdinghani.vercel.app)

---

## 📄 License

This project is licensed under the MIT License.
