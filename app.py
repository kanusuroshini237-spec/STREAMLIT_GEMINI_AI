import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
import os


# --------------------------------------------------
# PAGE CONFIGURATION (must be the first Streamlit call)
# --------------------------------------------------

st.set_page_config(
    page_title="Gemini RAG",
    page_icon="📚",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #F4F6F9;
    --surface: #FFFFFF;
    --ink: #1B2433;
    --muted: #5B6577;
    --line: #DCE1E8;
    --primary: #0E5A6B;
    --primary-dark: #0A4553;
    --primary-soft: #E4F1F4;
}

/* ---------- Base ---------- */
html, body, .stApp, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, 'Segoe UI', sans-serif;
}

.stApp {
    background: var(--bg);
    color: var(--ink);
}

/* Hide default Streamlit chrome */
#MainMenu, footer, [data-testid="stToolbar"] {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Centered, readable content width */
.block-container {
    max-width: 920px;
    padding-top: 2.5rem;
    padding-bottom: 4rem;
}

/* ---------- Typography ---------- */
h1, h2, h3, h4 {
    color: var(--ink);
    font-weight: 600;
    letter-spacing: -0.01em;
}

h3 {
    font-size: 1.15rem;
}

p, label, span, li {
    color: var(--ink);
}

/* ---------- Hero ---------- */
.hero {
    background: var(--primary);
    border-radius: 14px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.75rem;
}

.hero h1 {
    color: #FFFFFF;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    padding: 0;
}

.hero p {
    color: #CFE6EB;
    font-size: 1rem;
    margin: 0;
    line-height: 1.55;
}

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] > label {
    font-weight: 500;
    color: var(--ink);
}

[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1.5px dashed #AEB8C6;
    border-radius: 12px;
    padding: 1.5rem;
    transition: border-color 0.15s ease, background 0.15s ease;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--primary);
    background: var(--primary-soft);
}

[data-testid="stFileUploaderDropzone"] * {
    color: var(--muted);
}

[data-testid="stFileUploaderDropzone"] button {
    background: var(--surface);
    color: var(--primary);
    border: 1px solid var(--primary);
    border-radius: 8px;
    font-weight: 500;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: var(--primary);
    color: #FFFFFF;
    border: 1px solid var(--primary);
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-weight: 500;
    font-size: 0.95rem;
    transition: background 0.15s ease, box-shadow 0.15s ease;
}

.stButton > button:hover {
    background: var(--primary-dark);
    border-color: var(--primary-dark);
    color: #FFFFFF;
    box-shadow: 0 2px 8px rgba(14, 90, 107, 0.25);
}

.stButton > button:focus-visible {
    outline: 3px solid rgba(14, 90, 107, 0.35);
    outline-offset: 2px;
}

.stButton > button:active {
    background: var(--primary-dark);
    color: #FFFFFF;
}

/* ---------- Text input ---------- */
.stTextInput > label {
    font-weight: 500;
}

.stTextInput input {
    background: var(--surface);
    color: var(--ink);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    font-size: 1rem;
}

.stTextInput input::placeholder {
    color: #8C95A4;
}

.stTextInput input:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(14, 90, 107, 0.15);
}

/* ---------- Alerts (success / info / warning / error) ---------- */
[data-testid="stAlert"] {
    border-radius: 10px;
    border: 1px solid var(--line);
}

/* ---------- Bordered containers (used for the answer) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 4px solid var(--primary);
    border-radius: 10px;
    padding: 0.4rem 0.6rem;
}

/* ---------- Expander ---------- */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
}

[data-testid="stExpander"] summary {
    font-weight: 500;
    color: var(--ink);
}

[data-testid="stExpander"] summary:hover {
    color: var(--primary);
}

/* ---------- Divider ---------- */
hr {
    border: none;
    border-top: 1px solid var(--line);
    margin: 2rem 0 1.5rem 0;
}

/* ---------- Spinner ---------- */
.stSpinner > div > div {
    border-top-color: var(--primary);
}

/* ---------- Starter question pills ---------- */
.st-key-suggest button {
    background: var(--surface);
    color: var(--primary);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.35rem 0.9rem;
    font-size: 0.85rem;
    width: 100%;
}

.st-key-suggest button:hover {
    background: var(--primary-soft);
    border-color: var(--primary);
    color: var(--primary-dark);
    box-shadow: none;
}

/* ---------- Download button (outlined style) ---------- */
.stDownloadButton > button {
    background: var(--surface);
    color: var(--primary);
    border: 1px solid var(--primary);
    border-radius: 8px;
    padding: 0.55rem 1.4rem;
    font-weight: 500;
}

.stDownloadButton > button:hover {
    background: var(--primary-soft);
    color: var(--primary-dark);
    border-color: var(--primary-dark);
}

/* ---------- Progress bar ---------- */
[data-testid="stProgress"] > div > div > div > div {
    background-color: var(--primary);
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] h3 {
    font-size: 1rem;
    margin-bottom: 0.25rem;
}

.step {
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
    padding: 0.55rem 0;
}

.step-num {
    flex: 0 0 1.6rem;
    height: 1.6rem;
    border-radius: 50%;
    background: var(--primary-soft);
    color: var(--primary);
    font-weight: 600;
    font-size: 0.8rem;
    display: flex;
    align-items: center;
    justify-content: center;
}

.step-text {
    font-size: 0.9rem;
    line-height: 1.45;
    color: var(--muted);
}

.step-text b {
    color: var(--ink);
    font-weight: 600;
}

.doc-card {
    background: var(--primary-soft);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: var(--primary-dark);
    word-break: break-word;
}

.doc-card b {
    color: var(--primary-dark);
}

/* ---------- Stat chips ---------- */
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin: 0.75rem 0 1rem 0;
}

.chip {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 0.3rem 0.9rem;
    font-size: 0.85rem;
    color: var(--muted);
}

.chip b {
    color: var(--primary);
    font-weight: 600;
}

/* ---------- History ---------- */
.q-label {
    font-weight: 600;
    color: var(--ink);
    margin: 1.25rem 0 0.4rem 0;
}

.app-footer {
    text-align: center;
    color: #8C95A4;
    font-size: 0.8rem;
    margin-top: 3rem;
}

/* One gentle entrance for the header only */
.hero {
    animation: heroIn 0.5s ease-out both;
}

@keyframes heroIn {
    from { opacity: 0; transform: translateY(-6px); }
    to { opacity: 1; transform: none; }
}

@media (prefers-reduced-motion: reduce) {
    .hero { animation: none; }
}

/* ---------- Mobile ---------- */
@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .hero {
        padding: 1.4rem 1.25rem;
    }
    .hero h1 {
        font-size: 1.5rem;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Optional: set GEMINI_MODEL in .env to switch models without editing code
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing in the .env file.")
    st.stop()


# --------------------------------------------------
# GEMINI CLIENT
# --------------------------------------------------

client = genai.Client(api_key=API_KEY)


# --------------------------------------------------
# CHROMADB
# --------------------------------------------------

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="gemini_rag_collection"
)


# --------------------------------------------------
# HERO / TITLE
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>📚 Chat with your PDF</h1>
        <p>Upload a document, then ask questions and get answers
        grounded in its content. Powered by Gemini and ChromaDB.</p>
    </div>
    """,
    unsafe_allow_html=True
)


# --------------------------------------------------
# SETTINGS (sidebar)
# --------------------------------------------------

with st.sidebar:

    with st.expander("⚙️ Settings"):

        chunk_size = st.slider(
            "Chunk size (characters)", 300, 2000, 1000, 100
        )

        chunk_overlap = st.slider(
            "Chunk overlap", 0, 400, 150, 50
        )

        top_k = st.slider(
            "Chunks to retrieve", 1, 8, 3
        )

        st.caption(
            "Chunk settings apply the next time you process a PDF."
        )


# --------------------------------------------------
# PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_text_from_pdf(pdf_file):

    reader = PdfReader(pdf_file)

    pages = []

    for number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text()

        if page_text and page_text.strip():
            pages.append((number, page_text))

    return pages


# --------------------------------------------------
# TEXT CHUNKING
# --------------------------------------------------

def create_chunks(pages, chunk_size=1000, overlap=150):

    chunks = []
    chunk_pages = []

    # Each chunk starts `step` characters after the previous one,
    # so neighbouring chunks share `overlap` characters.
    step = max(1, chunk_size - min(overlap, chunk_size // 2))

    for page_number, page_text in pages:

        start = 0

        while start < len(page_text):

            chunk = page_text[start:start + chunk_size]

            if chunk.strip():
                chunks.append(chunk)
                chunk_pages.append(page_number)

            start += step

    return chunks, chunk_pages


# --------------------------------------------------
# GEMINI EMBEDDING
# --------------------------------------------------

def create_embedding(text):

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values


# --------------------------------------------------
# STORE CHUNKS IN CHROMADB
# --------------------------------------------------

def store_chunks(chunks, chunk_pages, progress=None):

    # Clear previous document
    try:
        chroma_client.delete_collection("gemini_rag_collection")
    except:
        pass

    global collection

    collection = chroma_client.get_or_create_collection(
        name="gemini_rag_collection"
    )

    embeddings = []

    for i, chunk in enumerate(chunks):

        embedding = create_embedding(chunk)

        embeddings.append(embedding)

        if progress is not None:
            progress.progress(
                (i + 1) / len(chunks),
                text=f"Embedding chunk {i + 1} of {len(chunks)}"
            )

    ids = []

    for i in range(len(chunks)):
        ids.append(f"chunk_{i}")

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"page": p} for p in chunk_pages]
    )

    return len(chunks)


# --------------------------------------------------
# ASK GEMINI
# --------------------------------------------------

def generate_answer(question, context):

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text or "The model returned an empty response."

    except Exception as error:
        return (
            f"**Could not generate an answer.** {error}\n\n"
            "Check that `GEMINI_MODEL` in your `.env` is a valid model "
            "name and that your API key has access to it."
        )


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# PROCESS PDF
# --------------------------------------------------

if uploaded_file is not None:

    st.success(f"Uploaded: {uploaded_file.name}")

    if st.button("🔄 Process PDF"):

        with st.spinner("Reading PDF..."):

            pages = extract_text_from_pdf(uploaded_file)
            text = "\n".join(page_text for _, page_text in pages)

        if not text.strip():

            st.error(
                "Could not extract text from this PDF."
            )

        else:

            with st.spinner("Creating chunks..."):

                chunks, chunk_pages = create_chunks(
                    pages,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap
                )

            progress_bar = st.progress(0, text="Starting embeddings...")

            number_of_chunks = store_chunks(
                chunks, chunk_pages, progress=progress_bar
            )

            progress_bar.empty()

            st.success("PDF processed successfully. You can now ask questions.")

            st.markdown(
                f"""
                <div class="chips">
                    <span class="chip">Pages <b>{len(pages)}</b></span>
                    <span class="chip">Characters <b>{len(text):,}</b></span>
                    <span class="chip">Chunks <b>{number_of_chunks}</b></span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.session_state.pdf_processed = True
            st.session_state.doc_name = uploaded_file.name
            st.session_state.doc_chunks = number_of_chunks
            st.session_state.history = []


# --------------------------------------------------
# QUESTION SECTION
# --------------------------------------------------

if st.session_state.get("pdf_processed", False):

    st.divider()

    st.subheader("💬 Ask a question")

    def set_question(text_value):
        st.session_state.question_input = text_value

    st.caption("Try a starter question:")

    with st.container(key="suggest"):

        s1, s2, s3 = st.columns(3)

        starters = [
            "Summarize this document",
            "What are the key points?",
            "List important terms and definitions",
        ]

        for col, label in zip((s1, s2, s3), starters):
            with col:
                st.button(
                    label,
                    key=f"starter_{label}",
                    on_click=set_question,
                    args=(label,)
                )

    question = st.text_input(
        "Enter your question:",
        key="question_input",
        placeholder="e.g. What are the main conclusions of this document?"
    )

    answered_now = False

    if st.button("🔍 Get Answer"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("Searching document..."):

                # Create embedding for question
                question_embedding = create_embedding(
                    question
                )

                # Search ChromaDB
                results = collection.query(
                    query_embeddings=[question_embedding],
                    n_results=max(1, min(top_k, collection.count()))
                )

            documents = results["documents"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]

            context = "\n\n".join(documents)

            with st.spinner("Gemini is generating the answer..."):

                answer = generate_answer(
                    question,
                    context
                )

            st.subheader("🤖 Answer")

            with st.container(border=True):
                st.markdown(answer)

            source_pages = sorted({m["page"] for m in metadatas})

            st.markdown(
                '<div class="chips">'
                + "".join(
                    f'<span class="chip">Source: page <b>{p}</b></span>'
                    for p in source_pages
                )
                + "</div>",
                unsafe_allow_html=True
            )

            with st.expander("📖 View retrieved context"):

                for i, doc in enumerate(documents):

                    st.markdown(
                        f"**Chunk {i + 1}** "
                        f"<span class='chip'>page <b>{metadatas[i]['page']}</b></span> "
                        f"<span class='chip'>distance <b>{distances[i]:.3f}</b>"
                        f" (lower is closer)</span>",
                        unsafe_allow_html=True
                    )

                    st.write(doc)

            st.session_state.history.insert(0, (question, answer))
            answered_now = True

    # Earlier questions in this session
    history = st.session_state.get("history", [])
    earlier = history[1:] if answered_now else history

    if earlier:

        st.divider()
        st.subheader("🕘 Earlier questions")

        for q, a in earlier:

            st.markdown(
                f'<div class="q-label">{q}</div>',
                unsafe_allow_html=True
            )

            with st.container(border=True):
                st.markdown(a)

    if history:

        st.divider()

        export_text = "\n\n".join(
            f"Q: {q}\nA: {a}" for q, a in reversed(history)
        )

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "⬇️ Download Q&A",
                data=export_text,
                file_name="qa_history.txt",
                mime="text/plain"
            )

        with col2:
            if st.button("🗑️ Clear history"):
                st.session_state.history = []
                st.rerun()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("### How it works")

    st.markdown(
        """
        <div class="step"><div class="step-num">1</div>
        <div class="step-text"><b>Upload</b> a PDF with selectable text.</div></div>
        <div class="step"><div class="step-num">2</div>
        <div class="step-text"><b>Process</b> it to split and index the content.</div></div>
        <div class="step"><div class="step-num">3</div>
        <div class="step-text"><b>Ask</b> a question and get an answer from the document.</div></div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.get("pdf_processed", False):

        st.markdown("### Current document")

        st.markdown(
            f"""
            <div class="doc-card">
                <b>{st.session_state.get("doc_name", "")}</b><br>
                {st.session_state.get("doc_chunks", 0)} chunks indexed
            </div>
            """,
            unsafe_allow_html=True
        )


st.markdown(
    '<div class="app-footer">Answers are generated only from your uploaded document.</div>',
    unsafe_allow_html=True
)