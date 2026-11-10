# 🤖 Agentic RAG Assistant with Model Context Protocol (MCP)

An intelligent, multi-step **AI Research Agent** built with **Streamlit**, **Google Gemini**, and the open **Model Context Protocol (MCP)** standard.

Unlike traditional passive RAG pipelines that only retrieve chunks once, this application transforms RAG into an autonomous **Agent**:
- **Autonomous Reasoning (ReAct)**: Plans multi-step problem solving, decides when to search documents, summarizes, or executes external tools.
- **Model Context Protocol (MCP) Client**: Dynamically connects to MCP servers to execute external tools (web scraping, real-time datetime, math computation, system diagnostics).
- **RAG as an MCP Server**: Exposes the PDF knowledge base as a standalone MCP server (`mcp_rag_server.py`) so external clients (Claude Desktop, Cursor, Antigravity) can query your documents.
- **Streamlit Deployable**: Optimized for Streamlit Community Cloud, Docker, and Render. Defaults to Google Generative AI Embeddings (`models/text-embedding-004`) to eliminate PyTorch memory overhead and prevent Cloud OOM (Error 137) crashes.

---

## 🏗️ Architecture

```
                                  ┌───────────────────────────┐
                                  │      Streamlit UI         │
                                  │ (Chat, Controls, Status)  │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │      RAG AI Agent         │
                                  │   (ReAct Reasoning Loop)  │
                                  └──────┬─────────────┬──────┘
                                         │             │
                    ┌────────────────────┘             └───────────────────┐
                    ▼                                                      ▼
     ┌─────────────────────────────┐                        ┌─────────────────────────────┐
     │      PDF RAG Engine         │                        │      MCP Client Manager     │
     │  (Chunking, Embedding,      │                        │  (JSON-RPC 2.0 over Stdio)  │
     │   Vector Similarity Search) │                        └──────────────┬──────────────┘
     └─────────────────────────────┘                                       │
                                                ┌──────────────────────────┴──────────────────────────┐
                                                ▼                                                     ▼
                                 ┌─────────────────────────────┐                       ┌─────────────────────────────┐
                                 │   Built-in MCP Server       │                       │    External MCP Servers     │
                                 │ - fetch_web_url             │                       │  (Brave Search, GitHub,     │
                                 │ - get_current_datetime      │                       │   PostgreSQL, Filesystem)   │
                                 │ - eval_math                 │                       └─────────────────────────────┘
                                 │ - system_info               │
                                 └─────────────────────────────┘
```

---

## ✨ Features

- **Multi-Step Agent Reasoning**: Breaks complex questions into iterative reasoning steps (`Thought -> Action -> Observation -> Final Answer`).
- **Interactive Step Visualizer**: Watch the agent think and use tools in real-time inside expandable `st.status` drawers.
- **Model Context Protocol (MCP) Integration**:
  - Connects to any standard MCP server over stdio.
  - Auto-discovers tool schemas and registers them for LLM function calling.
  - Includes a built-in reference MCP server with 4 active tools.
- **RAG MCP Server**: Includes `mcp_rag_server.py` to allow external tools (Claude Desktop, Cursor) to search your uploaded PDF index.
- **Zero-RAM Cloud Embeddings**: Uses Google's `models/text-embedding-004` to avoid the 1GB RAM limit on Streamlit Community Cloud.
- **Multiple Gemini Models**: Switch seamlessly between `gemini-2.0-flash`, `gemini-1.5-flash`, and `gemini-1.5-pro`.
- **Export & History**: Download full conversations and citations in CSV format.

---

## 🚀 Quickstart (Local)

### 1. Clone & Install Dependencies
```bash
cd ai-chatbot-main
pip install -r requirements.txt
```

### 2. Configure Your API Key
Create a `.env` file from the template:
```bash
cp .env.example .env
```
Add your free Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
*(You can get a free Gemini API key from [Google AI Studio](https://aistudio.google.com/)).*

### 3. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud (1-Click)

1. Push your repository to **GitHub**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your repository, branch (`main`), and set the main file path to:
   ```text
   app.py
   ```
4. In **Advanced Settings**, add your secret:
   ```toml
   GOOGLE_API_KEY = "your_actual_gemini_api_key"
   ```
5. Click **Deploy!** Your agentic chatbot is live with zero configuration needed.

---

## 🔌 Model Context Protocol (MCP) Guide

### Built-in MCP Tools
The application comes pre-packaged with a local MCP server (`mcp_server.py`) offering:
- `fetch_web_url`: Scrapes and extracts readable text from any web URL.
- `get_current_datetime`: Injects live date, UTC, and local time into the agent context.
- `eval_math`: Computes expressions safely using Python's `math` engine.
- `system_info`: Inspects host OS and environment information.

### Adding External MCP Servers
You can register additional MCP servers in `mcp_servers.json`:
```json
{
  "mcpServers": {
    "builtin_tools": {
      "command": "python3",
      "args": ["mcp_server.py"]
    },
    "custom_node_server": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-fetch"]
    }
  }
}
```

### Using the PDF RAG as an MCP Server in Claude Desktop / Cursor
You can expose this project's PDF knowledge base to Claude Desktop or Cursor!
Add this to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "rag-documents": {
      "command": "python3",
      "args": ["/path/to/ai-chatbot-main/mcp_rag_server.py"],
      "env": {
        "GOOGLE_API_KEY": "your_api_key"
      }
    }
  }
}
```

---

## 🐳 Docker Deployment

To run in Docker (Render, Cloud Run, AWS, Hugging Face Spaces):

```bash
docker build -t rag-ai-agent .
docker run -p 8501:8501 -e GOOGLE_API_KEY="your_api_key" rag-ai-agent
```

---

## 📂 Project Structure

```text
ai-chatbot-main/
├── .streamlit/
│   └── config.toml           # Streamlit theme & server configuration
├── app.py                    # Streamlit UI & Chat frontend
├── agent.py                  # Multi-step ReAct Agent & tool dispatcher
├── rag_engine.py             # PDF text extraction, chunking, embeddings & vector store
├── mcp_client.py             # JSON-RPC 2.0 MCP Client implementation
├── mcp_server.py             # Reference MCP Server (web fetch, math, datetime)
├── mcp_rag_server.py         # Exposes PDF RAG as an MCP Server to external tools
├── mcp_servers.json          # MCP Server definitions
├── sample.pdf                # Sample test document
├── requirements.txt          # Production-ready dependencies
├── Dockerfile                # Production Docker container
├── .env.example              # Environment variables template
└── README.md                 # Full documentation
```
