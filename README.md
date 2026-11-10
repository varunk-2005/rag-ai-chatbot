# RAG PDF Chatbot

A retrieval-augmented question-answering app built with **Python**, **Streamlit**, **FAISS** and **Google Gemini**. Upload one or more PDFs, and ask questions about them. Answers are generated only from the passages retrieved from your documents, and the model is told to say so when the answer is not there.

## Features

- **Multi-PDF upload:** several PDFs are combined into one searchable index.
- **Local embeddings:** text is embedded on your machine with `all-MiniLM-L6-v2`, so only the final answer step calls an external API.
- **FAISS vector search:** the 4 most similar chunks are retrieved for each question.
- **Grounded answers:** the prompt restricts Gemini to the retrieved context and asks it to reply "answer is not available in the context" otherwise.
- **Chat history:** the latest exchange is shown first, with earlier messages, timestamps and source file names below.
- **CSV export:** download the whole conversation from the sidebar.
- **Controls:** remove the last exchange, or reset the chat and the index.
- **Rate-limit handling:** a Gemini quota error shows a friendly message instead of crashing.

## How It Works

```
   INDEXING (once per upload)                 ANSWERING (every question)

   PDF files                                  User question
       │                                           │
       ▼                                           ▼
   PyPDF2: extract text                       Embed question (MiniLM)
       │                                           │
       ▼                                           ▼
   Split into chunks                          FAISS similarity search
   (1,000 chars, 100 overlap)                 (top 4 chunks)
       │                                           │
       ▼                                           ▼
   Embed chunks (MiniLM, 384-dim)             Build prompt: context + question
       │                                           │
       ▼                                           ▼
   FAISS index (in session memory) ─────────► Gemini 2.5 Flash ──► Answer
```

| Step | Function | What it does |
|---|---|---|
| Extract | `get_pdf_text` | Reads every page of every uploaded PDF with PyPDF2 and joins the text. |
| Chunk | `get_text_chunks` | Splits the text with `RecursiveCharacterTextSplitter` (chunk size 1,000, overlap 100). |
| Index | `build_vector_store` | Embeds the chunks with `sentence-transformers/all-MiniLM-L6-v2` and stores them in FAISS. |
| Retrieve and answer | `get_answer` | Finds the 4 nearest chunks, puts them in the prompt, and calls `gemini-2.5-flash`. |
| Display | `render_chat_history` | Renders the latest exchange and the earlier messages. |

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF parsing | PyPDF2 |
| Chunking | LangChain text splitters |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`), run locally |
| Vector store | FAISS (CPU) |
| LLM | Google Gemini 2.5 Flash (`google-generativeai`) |
| Language | Python 3.10+ |

## Getting Started

### Prerequisites

- Python 3.10 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Installation

```bash
git clone https://github.com/varunk-2005/rag-ai-chatbot.git
cd rag-ai-chatbot/ai-chatbot-main
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The first run downloads the embedding model (about 90 MB).

### Configuration

Create a `.env` file next to `app.py`:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

On Streamlit Community Cloud, add the same key under *App settings → Secrets*.

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Usage

1. Upload one or more PDFs in the sidebar.
2. Click **Submit & Process** and wait for "Ready".
3. Type a question in the chat box.
4. Use **Clear last** or **Reset** in the sidebar, or download the history as CSV.

A `sample.pdf` is included for a quick test.

## Project Structure

```
rag-ai-chatbot/
└── ai-chatbot-main/
    ├── app.py              # Whole app: ingestion, retrieval, generation, UI
    ├── requirements.txt
    ├── sample.pdf          # Example document
    └── .gitignore
```

## Design Decisions

- **Local embeddings, hosted generation.** Embedding runs on the user's machine at no cost, and only the 4 retrieved chunks are sent to Gemini. This keeps API usage low and avoids sending whole documents out.
- **Chunk size 1,000 with 100 overlap.** Large enough to keep a paragraph's meaning together, with overlap so a sentence cut at a boundary still appears whole in one chunk.
- **Index built on button click, not on every rerun.** Streamlit reruns the script on each interaction; building the index only on **Submit & Process** and keeping it in `st.session_state` avoids re-embedding the documents for every question.
- **Explicit "not in context" instruction.** The prompt tells the model to decline instead of guessing, which reduces hallucinated answers.

## Limitations and Future Improvements

- **No source citations.** Answers do not show which page or chunk they came from. Next step: keep page numbers as chunk metadata and display them.
- **Each question is answered on its own.** Earlier turns are displayed but not used, so follow-ups like "explain that further" do not work. Next step: rewrite follow-ups into standalone questions.
- **The index lives in session memory.** It is lost on refresh. Next step: save and load the FAISS index from disk.
- **Text-only PDFs.** Scanned PDFs need OCR, and tables and images are not handled.
- **No evaluation set or automated tests yet.** Next step: a small question set with expected answers, including questions that are not in the document, to measure answer and refusal accuracy.
- **Chat messages are rendered as raw HTML.** User input and model output should be escaped, or rendered with `st.chat_message`, before this is deployed publicly.
- **Dependencies are not pinned** in `requirements.txt`.

## Author

**Varun Kumar** · [GitHub](https://github.com/varunk-2005)
