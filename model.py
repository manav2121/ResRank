import os
import PyPDF2
import docx2txt
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

def extract_text(file_path):
    """Extract text from PDF, DOCX, or TXT file"""
    text = ""
    if file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    elif file_path.endswith(".docx"):
        text = docx2txt.process(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    return text


def match_resumes(job_description, resume_folder):
    """Rank resumes based on similarity with JD"""
    resumes, candidates = [], []

    for file in os.listdir(resume_folder):
        file_path = os.path.join(resume_folder, file)
        text = extract_text(file_path)
        if text.strip():
            resumes.append(text)
            candidates.append(file)

    if not resumes:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([job_description] + resumes)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    ranking = sorted(list(zip(candidates, similarities)), key=lambda x: x[1], reverse=True)
    return ranking
