# Agentic RAG Assistant with MCP

A PDF question-answering agent built with **Python**, **Streamlit** and **Google Gemini**. Instead of a fixed retrieve-then-answer pipeline, an agent loop decides which tool to use for each question: search the uploaded documents, do a calculation, or call a tool exposed over the **Model Context Protocol (MCP)**. Answers cite the source file and page.

## Features

- **Agent loop (ReAct):** the model writes a thought, picks a tool, reads the result and repeats, up to 6 steps, before giving a final answer.
- **Document search with citations:** PDFs are split into chunks that keep their file name and page number, so answers can point to the page they came from.
- **Semantic retrieval:** chunks and questions are embedded with the Gemini embedding API and compared by cosine similarity. A keyword (TF-IDF) fallback is used when no API key is set.
- **MCP client written from scratch:** connects to MCP servers over stdio using JSON-RPC 2.0, performs the handshake, discovers tools and calls them.
- **Built-in MCP server** with three tools: current date and time, a safe math evaluator, and system info.
- **Document search as an MCP server:** `mcp_rag_server.py` exposes the PDF search so other MCP clients can use it.
- **Visible reasoning:** each thought, tool call and result is shown in the UI.
- **Chat export** to CSV.

## Architecture

```
                    ┌──────────────────────┐
                    │   Streamlit UI       │  app.py
                    └──────────┬───────────┘
                               ▼
                    ┌──────────────────────┐
                    │   Agent loop         │  agent.py
                    │ Thought → Action →   │
                    │ Observation → Answer │
                    └─────┬──────────┬─────┘
                          │          │
              ┌───────────┘          └───────────┐
              ▼                                  ▼
   ┌──────────────────────┐           ┌──────────────────────┐
   │  RAG engine          │           │  MCP client          │  mcp_client.py
   │  rag_engine.py       │           │  JSON-RPC over stdio │
   │  extract → chunk →   │           └──────────┬───────────┘
   │  embed → search      │                      ▼
   └──────────────────────┘           ┌──────────────────────┐
                                      │  MCP server          │  mcp_server.py
                                      │  get_current_datetime│
                                      │  eval_math           │
                                      │  system_info         │
                                      └──────────────────────┘
```

| File | Role |
|---|---|
| `app.py` | Streamlit UI: API key, PDF upload, chat, reasoning display, CSV export. |
| `agent.py` | The agent loop. Builds the prompt with the tool list, parses the model's chosen action, runs the tool and feeds the result back. |
| `rag_engine.py` | PDF text extraction, chunking (1,000 characters, 150 overlap), embeddings and cosine-similarity search with NumPy. |
| `mcp_client.py` | Starts MCP servers as subprocesses, does the `initialize` handshake, lists and calls tools. |
| `mcp_server.py` | A small MCP server with three tools. |
| `mcp_rag_server.py` | A second MCP server that exposes document search to external MCP clients. |

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| LLM | Google Gemini (`gemini-3.6-flash`) |
| Embeddings | Gemini embedding API (`gemini-embedding-2`) |
| Vector search | NumPy cosine similarity, in memory |
| PDF parsing | pypdf |
| Chunking | LangChain text splitters |
| Tool protocol | Model Context Protocol, JSON-RPC 2.0 over stdio |
| Language | Python 3.11 |

## Getting Started

### Prerequisites

- Python 3.11 or newer
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/)

### Installation

```bash
git clone https://github.com/varunk-2005/rag-ai-chatbot.git
cd rag-ai-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
```

Then put your key in `.env`:

```env
GOOGLE_API_KEY=your_gemini_api_key
```

### Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload one or more PDFs, click **Process & Build Index**, then ask a question. A `sample.pdf` is included.

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io/), create an app from the repository with `app.py` as the main file.
3. Under *Advanced settings → Secrets*, add:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key"
   ```

## MCP

The app starts `mcp_server.py` as a subprocess and talks to it over stdin and stdout:

1. `initialize`: client and server exchange protocol version and capabilities.
2. `tools/list`: the server returns each tool's name, description and JSON input schema. These are added to the agent's prompt.
3. `tools/call`: when the agent picks an MCP tool, the client sends the call and returns the result to the agent.

To use the document search from another MCP client, register `mcp_rag_server.py` in that client's configuration:

```json
{
  "mcpServers": {
    "rag-documents": {
      "command": "python3",
      "args": ["/path/to/rag-ai-chatbot/mcp_rag_server.py"],
      "env": { "GOOGLE_API_KEY": "your_api_key" }
    }
  }
}
```

It indexes the PDFs found in its working directory when it starts.

## Design Decisions

- **Agent loop instead of a fixed pipeline.** A question may need a document search, a calculation, or both. Letting the model choose the tool handles these without separate code paths.
- **Hosted embeddings instead of a local model.** A local sentence-transformers model needs PyTorch, which is too large for a free 1 GB host. Calling the embedding API keeps the app small, at the cost of one API call per question.
- **NumPy instead of a vector database.** For a few hundred chunks, a matrix multiply is fast and has no extra dependency. A vector database would matter at much larger scale.
- **Math without `eval`.** The calculator parses the expression into a syntax tree and only runs whitelisted operators and functions, so the model cannot be tricked into running arbitrary code.
- **Iteration limit.** The loop stops after 6 steps so a confused model cannot run forever.

## Limitations and Future Improvements

- **No evaluation set yet.** Retrieval and answer quality have not been measured. Next step: a question set with expected answers and pages, including questions the documents cannot answer.
- **Text-parsed tool calls.** The agent reads `Action:` and `Action Input:` from the model's text with regular expressions, which can fail on badly formatted output. Native function calling would be more reliable.
- **The index lives in session memory** and is lost on refresh.
- **Text-only PDFs.** Scanned PDFs need OCR; tables and images are not handled.
- **No guardrails on document content.** Text inside a PDF is passed to the model as-is, so instructions hidden in a document could influence it.
- **No automated tests yet.**
- **The MCP document server and the Streamlit app keep separate indexes.**

## How This Was Built

The first version was a single-file retrieve-and-answer app. The agent loop and MCP layer were scaffolded with an AI coding tool, then reviewed, tested and corrected by hand: retired model names were replaced, an unsafe `eval` calculator was rewritten, a web-fetch tool was removed, and unmeasured benchmark claims were deleted.

## Author

**Varun Kumar** · [GitHub](https://github.com/varunk-2005)
