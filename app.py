import streamlit as st
import PyPDF2, docx2txt, io, base64, os, re, hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# =========================
# Page Setup & Small Styles
# =========================
st.set_page_config(page_title="ResRank – Resume Ranker", layout="wide", menu_items={
    "Get Help": None, "Report a bug": None, "About": "ResRank • TF-IDF based resume ranking"
})

PRIMARY_COLOR = "#4F46E5"  # indigo-ish
st.markdown(
    f"""
    <style>
      .score-bar {{
        height: 10px;
        background: #eee;
        border-radius: 999px;
        overflow: hidden;
      }}
      .score-fill {{
        height: 100%;
        background: linear-gradient(90deg, {PRIMARY_COLOR}, #22d3ee);
      }}
      .pill {{
        display:inline-block; padding: 4px 8px; border-radius: 999px; background:#f1f5f9; margin-right:6px; font-size:12px;
      }}
      .mark {{
        background: #fde68a;
        padding: 0 2px;
        border-radius: 2px;
      }}
      .stMarkdown p {{ margin-bottom: 0.4rem; }}
    </style>
    """,
    unsafe_allow_html=True
)

# =====================
# Helpers & Cache Layer
# =====================
def file_bytes_to_key(file_bytes: bytes) -> str:
    return hashlib.sha1(file_bytes).hexdigest()

@st.cache_data(show_spinner=False)
def extract_text_bytes(file_bytes: bytes, file_name: str) -> str:
    text = ""
    if file_name.lower().endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in reader.pages:
            # PyPDF2 extract_text can return None
            text += page.extract_text() or ""
    elif file_name.lower().endswith(".docx"):
        temp_file = f"temp_{hashlib.md5(file_bytes).hexdigest()}.docx"
        with open(temp_file, "wb") as f:
            f.write(file_bytes)
        text = docx2txt.process(temp_file) or ""
        os.remove(temp_file)
    elif file_name.lower().endswith(".txt"):
        # try utf-8; fallback latin-1
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1", errors="ignore")
    return text

def get_file_b64(file_bytes):
    return base64.b64encode(file_bytes).decode("utf-8")

def highlight_terms(text: str, keywords: set) -> str:
    if not text or not keywords:
        return text
    # Simple word-boundary highlight (case-insensitive)
    def repl(match):
        return f"<span class='mark'>{match.group(0)}</span>"

    # Sort by length to avoid partial overlaps (longer first)
    ordered = sorted(list(keywords), key=len, reverse=True)
    for kw in ordered:
        if not kw.strip():
            continue
        pattern = re.compile(rf"(?i)\\b{re.escape(kw)}\\b")
        text = pattern.sub(repl, text)
    return text

def jd_keywords(jd_text: str, top_k: int = 25) -> list:
    # Basic keyword pick: TF-IDF on JD split into sentences/lines to surface repeated terms
    parts = [p.strip() for p in re.split(r"[\\n\\.;]", jd_text) if p.strip()]
    if not parts:
        return []
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=2000)
    X = vec.fit_transform(parts)
    scores = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    top_idx = scores.argsort()[::-1][:top_k]
    return [vocab[i] for i in top_idx if len(vocab[i]) > 2]

# ===============
# Sidebar Controls
# ===============
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.caption("Tune filters and display options.")

    min_match = st.slider("Minimum match %", 0, 100, 0, step=1)
    top_n = st.slider("Show top N candidates", 1, 100, 50, step=1)
    show_keywords = st.toggle("Show extracted JD keywords", value=True)
    show_preview_kw = st.toggle("Highlight keywords in preview", value=True)
    st.divider()
    st.caption("Need persistent uploads? Consider using a DB or S3; local files are ephemeral on Render.")

# =================
# Header & Input UI
# =================
st.markdown("<h1>🚀 ResRank – Resume Ranker</h1>", unsafe_allow_html=True)
st.caption("Paste a job description, upload resumes (PDF/DOCX/TXT), and get ranked matches. Uses TF-IDF + cosine similarity.")

col_left, col_right = st.columns([1.1, 0.9], vertical_alignment="top")

with col_left:
    jd_default = ""
    job_description = st.text_area("📋 Job Description", value=jd_default, height=220, placeholder="Paste the JD here...")

with col_right:
    uploaded_files = st.file_uploader(
        "📎 Upload Resumes",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        help="You can select multiple files."
    )
    # Session state for uploads
    if "resume_data" not in st.session_state:
        st.session_state.resume_data = {}

    # Add new uploads
    if uploaded_files:
        for f in uploaded_files:
            st.session_state.resume_data[f.name] = f.read()

    # Manage uploaded list
    if st.session_state.resume_data:
        st.markdown("**Uploaded files**")
        to_remove = st.selectbox(
            "Remove a file (optional)",
            options=["—"] + list(st.session_state.resume_data.keys()),
            index=0
        )
        cols_rm = st.columns(2)
        if to_remove != "—" and cols_rm[0].button("Remove selected"):
            st.session_state.resume_data.pop(to_remove, None)
            st.toast(f"Removed {to_remove}", icon="🗑️")
        if cols_rm[1].button("Clear all"):
            st.session_state.resume_data.clear()
            st.toast("Cleared all uploads", icon="🧹")

st.divider()

# ======================
# Compute & Rank Section
# ======================
if st.session_state.resume_data and job_description.strip():
    with st.spinner("Computing TF-IDF vectors and ranking resumes..."):
        candidates, resumes_text = [], []
        for name, data in st.session_state.resume_data.items():
            text = extract_text_bytes(data, name)
            if text and text.strip():
                candidates.append(name)
                resumes_text.append(text)

        if not candidates:
            st.info("No extractable text found in uploaded resumes.")
        else:
            vec = TfidfVectorizer(stop_words="english")
            X = vec.fit_transform([job_description] + resumes_text)
            sims = cosine_similarity(X[0:1], X[1:]).flatten()
            df = pd.DataFrame({
                "Candidate": candidates,
                "Match %": (sims * 100).round(2)
            }).sort_values("Match %", ascending=False).reset_index(drop=True)
            # Apply filters
            if min_match > 0:
                df = df[df["Match %"] >= min_match].reset_index(drop=True)
            df = df.head(top_n)

            # ===========
            # Results UI
            # ===========
            st.subheader("🏆 Candidate Rankings")

            # Pretty table with a score bar + action buttons
            # Build an HTML table manually for better control
            def row_html(idx, row):
                pct = float(row["Match %"])
                width = max(0.0, min(100.0, pct))
                name = row["Candidate"]
                return f"""
                <tr>
                  <td style="padding:10px 8px; text-align:center;">{idx+1}</td>
                  <td style="padding:10px 8px;">{name}</td>
                  <td style="padding:10px 8px; width:260px;">
                    <div style="display:flex; align-items:center; gap:10px;">
                      <div class="score-bar" style="flex:1;"><div class="score-fill" style="width:{width}%;"></div></div>
                      <div style="width:64px; text-align:right;">{pct:.2f}%</div>
                    </div>
                  </td>
                </tr>
                """

            table_rows = "\n".join(row_html(i, r) for i, r in df.iterrows())
            st.markdown(
                f"""
                <div style="overflow:auto; border:1px solid #e5e7eb; border-radius:12px;">
                  <table style="width:100%; border-collapse:collapse;">
                    <thead style="background:#f8fafc;">
                      <tr>
                        <th style="padding:10px 8px; text-align:center; width:60px;">Rank</th>
                        <th style="padding:10px 8px; text-align:left;">Candidate</th>
                        <th style="padding:10px 8px; text-align:left;">Match</th>
                      </tr>
                    </thead>
                    <tbody>
                      {table_rows}
                    </tbody>
                  </table>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Action row: select a candidate to preview + download ranked CSV
            st.markdown("### 🔎 Preview & Export")
            colA, colB, colC = st.columns([0.8, 0.2, 0.2])
            selection = colA.selectbox(
                "Select a candidate to preview",
                options=["—"] + df["Candidate"].tolist(),
                index=0
            )
            csv_data = df.to_csv(index=False).encode("utf-8")
            colB.download_button(
                "⬇️ Download Rankings (CSV)",
                data=csv_data,
                file_name="rankings.csv",
                mime="text/csv",
                use_container_width=True
            )
            colC.write("")  # spacer

            # Show JD keywords (optional)
            if show_keywords:
                keys = jd_keywords(job_description, top_k=20)
                if keys:
                    st.caption("Top JD keywords (auto-extracted):")
                    st.markdown(" ".join([f"<span class='pill'>{k}</span>" for k in keys]), unsafe_allow_html=True)

            # ==================
            # Single Preview Box
            # ==================
            if selection != "—":
                st.markdown("---")
                st.markdown(f"#### 📄 Preview: **{selection}**")

                file_bytes = st.session_state.resume_data[selection]
                if selection.lower().endswith(".pdf"):
                    b64 = get_file_b64(file_bytes)
                    st.markdown(
                        f'<embed src="data:application/pdf;base64,{b64}" width="100%" height="800px" type="application/pdf">',
                        unsafe_allow_html=True
                    )
                else:
                    raw_text = extract_text_bytes(file_bytes, selection)
                    if show_preview_kw and show_keywords:
                        keys = set(jd_keywords(job_description, top_k=40))
                        highlighted = highlight_terms(raw_text, keys)
                        st.markdown(f"<div style='white-space:pre-wrap;'>{highlighted}</div>", unsafe_allow_html=True)
                    else:
                        st.text_area("Resume Content", raw_text, height=800)

                st.download_button("⬇️ Download Resume", data=file_bytes, file_name=selection, use_container_width=True)

else:
    st.info("Paste a Job Description and upload at least one resume to get rankings.")

# ==================
# Footer tiny helper
# ==================
with st.expander("ℹ️ Tips & Notes", expanded=False):
    st.write(
        "- Results are TF-IDF based. Consider adding embedding models later for semantic matching.\n"
        "- If a PDF is image-only (no text layer), text extraction will be empty. Use OCR upstream if needed.\n"
        "- On Render, storage is ephemeral. Don’t rely on local files for persistence."
    )
