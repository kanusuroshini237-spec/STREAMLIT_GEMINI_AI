# 📚 Chat with your PDF

A Streamlit app that lets you ask questions about a PDF and get answers grounded in its content. It uses **Retrieval-Augmented Generation (RAG)** with the Gemini API for embeddings and answers, and ChromaDB as the vector store.

## Features

- **Upload and index a PDF** with a progress bar while embeddings are created
- **Grounded answers**: the model answers only from the retrieved text, and says so when the answer isn't in the document
- **Source pages**: every retrieved chunk shows its page number and a distance score (lower means closer to your question)
- **Starter questions**: one-click prompts such as "Summarize this document"
- **Adjustable settings**: chunk size, chunk overlap, and number of chunks to retrieve
- **Session history**: review earlier questions, download the Q&A as a text file, or clear it
- **Clean, responsive UI** with custom CSS and a sidebar guide

## How it works

1. The PDF is read page by page with `pypdf`.
2. Each page is split into overlapping chunks.
3. Each chunk is embedded with `gemini-embedding-001` and stored in ChromaDB along with its page number.
4. Your question is embedded the same way, and the closest chunks are retrieved.
5. The retrieved chunks and your question are sent to a Gemini model, which writes the answer.

## Requirements

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey)

## Installation

```bash
# 1. Clone or download the project, then enter its folder
cd your-project-folder

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install streamlit google-genai python-dotenv pypdf chromadb
```

## Configuration

Create a file named `.env` in the same folder as `app.py`:

```
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
```

- `GEMINI_API_KEY` is **required**. Write it with no quotes and no spaces around `=`.
- `GEMINI_MODEL` is **optional**. If you leave it out, the app uses `gemini-3.5-flash-lite`. Set it to any Gemini model your key can access.

Add `.env` to your `.gitignore` so your key is never uploaded to GitHub.

## Run

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Usage

1. Upload a PDF.
2. Click **Process PDF** and wait for indexing to finish.
3. Type a question, or pick a starter question.
4. Click **Get Answer**.
5. Open **View retrieved context** to see which chunks and pages the answer came from.

### Settings (sidebar)

| Setting | Default | Notes |
| --- | --- | --- |
| Chunk size (characters) | 1000 | Applies the next time you process a PDF |
| Chunk overlap | 150 | Applies the next time you process a PDF |
| Chunks to retrieve | 3 | Applies to your next question |

Larger chunk sizes keep more context together. More retrieved chunks help broad questions such as summaries, but send more text to the model.

## Project structure

```
.
├── app.py       # The whole app: UI, CSS, and RAG logic
├── .env         # Your API key (not committed)
└── README.md
```

## Limitations

- **Text-based PDFs only.** Scanned PDFs without selectable text are not supported because there is no OCR step.
- **One document at a time.** Processing a new PDF replaces the previous one.
- **In-memory storage.** ChromaDB runs in memory, so the index and history are lost when the app restarts.
- **Partial summaries on long documents.** Only the top retrieved chunks are used, so raise "Chunks to retrieve" for broader questions.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `GEMINI_API_KEY is missing` | Check that `.env` is in the folder you run `streamlit run` from and the variable name is exact |
| "Model not found" or similar error in the answer | Set `GEMINI_MODEL` in `.env` to a valid model name from the Google AI docs |
| "Could not extract text from this PDF" | The PDF is probably scanned images; use a text-based PDF |
| Styles look off after a Streamlit upgrade | Streamlit's internal element names can change; check the CSS block at the top of `app.py` |

## Tech stack

- [Streamlit](https://streamlit.io/) for the interface
- [Gemini API](https://ai.google.dev/) (`google-genai`) for embeddings and answers
- [ChromaDB](https://www.trychroma.com/) for vector search
- [pypdf](https://pypdf.readthedocs.io/) for PDF text extraction
- [python-dotenv](https://pypi.org/project/python-dotenv/) for environment variables
