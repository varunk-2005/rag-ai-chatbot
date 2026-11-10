"""
Standalone Model Context Protocol (MCP) Server for RAG Knowledge Base.
Allows external MCP clients (Claude Desktop, Cursor, Antigravity) to query
the uploaded documents and RAG vector index via standard MCP tools.

Tools exposed:
- search_rag_documents: Semantic search over indexed documents.
- list_rag_documents: Lists indexed files and summary statistics.
- read_pdf_sample: Inspects sample text from available documents.
"""

import sys
import json
import os
import glob
from typing import Dict, Any, List

# RAG imports
from rag_engine import extract_text_from_pdfs, chunk_documents, RAGVectorStore

# In-memory store for the MCP server session
STORE: Dict[str, Any] = {
    "vector_store": None,
    "indexed_files": []
}

def init_default_index():
    """Initializes index from any local PDFs in current directory if available."""
    pdf_files = glob.glob("*.pdf") + glob.glob("**/*.pdf", recursive=True)
    if pdf_files and STORE["vector_store"] is None:
        try:
            pages = extract_text_from_pdfs(pdf_files[:5])
            chunks = chunk_documents(pages)
            api_key = os.getenv("GOOGLE_API_KEY")
            store = RAGVectorStore(api_key=api_key)
            store.build_index(chunks)
            STORE["vector_store"] = store
            STORE["indexed_files"] = [os.path.basename(f) for f in pdf_files[:5]]
        except Exception:
            pass


RAG_TOOLS = [
    {
        "name": "search_rag_documents",
        "description": "Searches the PDF knowledge base for relevant passages, excerpts, and page citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query or keyword phrase to find in the documents."},
                "top_k": {"type": "integer", "description": "Number of top matching excerpts to return (default 4)."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_rag_documents",
        "description": "Lists all documents currently indexed in the RAG knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


def handle_rag_request(req: dict) -> dict:
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})

    if method == "initialize":
        init_default_index()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "pdf-rag-mcp-server",
                    "version": "1.0.0"
                }
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": RAG_TOOLS}
        }

    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {})

        if name == "search_rag_documents":
            query = args.get("query", "")
            top_k = args.get("top_k", 4)
            store: RAGVectorStore = STORE.get("vector_store")
            if not store or not store.chunks:
                # Try fallback: text search in local PDFs
                output = f"No active vector store index. Query: '{query}'. Please upload and index documents in the Streamlit app."
            else:
                matches = store.similarity_search(query, top_k=top_k)
                if not matches:
                    output = "No matching passages found in the knowledge base."
                else:
                    lines = [f"Found {len(matches)} relevant excerpts:"]
                    for idx, m in enumerate(matches, 1):
                        meta = m["metadata"]
                        lines.append(f"\n[{idx}] Document: {meta.get('source')} (Page {meta.get('page')}) - Score: {m['score']}")
                        lines.append(f"Content:\n{m['text']}\n" + "-"*40)
                    output = "\n".join(lines)

        elif name == "list_rag_documents":
            store = STORE.get("vector_store")
            if store:
                output = store.get_document_summary()
            else:
                pdf_files = glob.glob("*.pdf")
                output = f"Discovered PDF files on disk: {pdf_files if pdf_files else 'None'}"

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Tool '{name}' not found"}
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": output}]
            }
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"}
        }


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if "id" not in req:
                continue
            resp = handle_rag_request(req)
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
