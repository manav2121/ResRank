
import streamlit as st
import PyPDF2, docx2txt, io, base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper Functions ---
def extract_text_bytes(file_bytes, file_name):
    text = ""
    if file_name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    elif file_name.endswith(".docx"):
        temp_file = f"temp_{file_name}"
        with open(temp_file, "wb") as f:
            f.write(file_bytes)
        text = docx2txt.process(temp_file)
        import os
        os.remove(temp_file)
    elif file_name.endswith(".txt"):
        text = file_bytes.decode("utf-8")
    return text

def get_file_b64(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")

# --- Streamlit UI ---
st.set_page_config(page_title="ResRank", layout="wide")
st.title("🥇 ResRank - Resume Ranking App")

job_description = st.text_area(" Job Description:", height=100)
uploaded_files = st.file_uploader(
    "Upload Resumes ", type=["pdf","docx","txt"], accept_multiple_files=True
)

if "resume_data" not in st.session_state:
    st.session_state.resume_data = {}

if uploaded_files:
    for file in uploaded_files:
        st.session_state.resume_data[file.name] = file.read()

# --- Process and Rank ---
if uploaded_files and job_description:
    candidates = []
    resumes_text = []
    for name, data in st.session_state.resume_data.items():
        text = extract_text_bytes(data, name)
        if text.strip():
            candidates.append(name)
            resumes_text.append(text)

    vectors = TfidfVectorizer(stop_words="english").fit_transform([job_description]+resumes_text)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    # Sort candidates by similarity descending
    sorted_candidates = sorted(zip(candidates, similarities), key=lambda x: x[1], reverse=True)

    st.subheader("Candidate Rankings :")
    
    # Create table using columns for interactivity
    st.markdown(
        """
        <style>
        .header {text-align:center; font-weight:bold;}
        </style>
        """,
        unsafe_allow_html=True
    )

    for rank, (candidate, score) in enumerate(sorted_candidates, start=1):
        cols = st.columns([1, 3, 2, 2])  # Rank, Name, Match %, Preview
        cols[0].markdown(f"<div class='header'>{rank}</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div class='header'>{candidate}</div>", unsafe_allow_html=True)
        cols[2].markdown(f"<div class='header'>{score*100:.2f}%</div>", unsafe_allow_html=True)
        if cols[3].button("view", key=f"view_{candidate}"):
            file_bytes = st.session_state.resume_data[candidate]
            st.expander(f"view: {candidate}", expanded=True)
            if candidate.endswith(".pdf"):
                b64 = get_file_b64(file_bytes)
                st.markdown(
                    f'<embed src="data:application/pdf;base64,{b64}" width="100%" height="800px" type="application/pdf">',
                    unsafe_allow_html=True
                )
            else:
                text = extract_text_bytes(file_bytes, candidate)
                st.text_area("Resume Content", text, height=800)
            st.download_button("⬇️ Download Resume", data=file_bytes, file_name=candidate)


