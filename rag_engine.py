"""
RAG Engine for PDF Document Ingestion, Chunking, Embeddings, and Retrieval.
Tables are read with pdfplumber and rewritten row by row, so each value keeps
its row and column labels (see table_to_text).
Supports:
1. Google Gemini embeddings (semantic search) - used when an API key is available.
2. Pure-Python TF-IDF fallback (keyword search) - used when there is no API key.
"""

import os
import io
import re
import math
from typing import List, Dict, Any, Optional
from collections import Counter
import numpy as np

# PDF extraction: try pypdf first (modern), then PyPDF2
try:
    import pypdf
    PDF_READER_CLS = pypdf.PdfReader
except ImportError:
    try:
        import PyPDF2
        PDF_READER_CLS = PyPDF2.PdfReader
    except ImportError:
        PDF_READER_CLS = None

# Optional table-aware PDF parser (used for pages that contain tables)
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Optional LangChain text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN_SPLITTER = True
except ImportError:
    HAS_LANGCHAIN_SPLITTER = False

# Google Generative AI
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# Gemini embedding model. Change it here only; it is used for both documents and queries.
EMBEDDING_MODEL = "models/gemini-embedding-2"


class DocumentChunk:
    """Represents a text chunk with associated metadata."""
    def __init__(self, text: str, metadata: Dict[str, Any], chunk_id: int):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id
        self.embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata
        }


def _clean_cell(cell: Any) -> str:
    """Flattens a table cell to one line of text."""
    return " ".join(str(cell or "").split())


def table_to_text(table: List[List[Any]]) -> str:
    """
    Converts one extracted table into self-contained text, one paragraph per row.

    Plain text extraction reads a table as one long stream, so a cell loses its
    row and column labels. Here every row is rewritten so that each value is
    stated together with the labels it belongs to.

    Two layouts are handled:
    1. Grid (matrix) tables: a header row, then a second header row whose first
       cells are empty (for example dates on row 1 and time slots on row 2),
       with row labels in the leading columns. Each non-empty cell becomes
       "<column header> <sub-header>: <cell>" under its row labels.
    2. Record tables: one header row, one record per row. Each row becomes
       "<header>: <value>; <header>: <value>; ...".
    """
    rows = [[_clean_cell(c) for c in row] for row in table if row]
    rows = [r for r in rows if any(r)]
    if len(rows) < 2:
        return ""

    header = rows[0]
    second = rows[1]
    # Leading empty cells in the second row mean it is a sub-header, and tell us
    # how many leading columns are row labels.
    n_labels = 0
    for cell in second:
        if cell:
            break
        n_labels += 1
    is_grid = 0 < n_labels < len(header) and len(rows) > 2

    paragraphs = []
    if is_grid:
        last_labels = [""] * n_labels
        for row in rows[2:]:
            # Merged cells come out empty: carry the label down from the row above
            labels = []
            for i in range(n_labels):
                if row[i]:
                    last_labels[i] = row[i]
                    for j in range(i + 1, n_labels):
                        last_labels[j] = ""
                labels.append(last_labels[i])
            entries = []
            for col in range(n_labels, min(len(row), len(header))):
                if row[col]:
                    column = " ".join(x for x in (header[col], second[col]) if x)
                    entries.append(f"{column}: {row[col]}")
            if entries:
                title = ", ".join(x for x in labels if x)
                paragraphs.append(f"{title}\n" + "\n".join(f"- {e}" for e in entries))
    else:
        for row in rows[1:]:
            pairs = [f"{h}: {v}" for h, v in zip(header, row) if v]
            if pairs:
                paragraphs.append("; ".join(pairs))

    return "\n\n".join(paragraphs)


def _extract_page_texts(source: Any) -> List[Dict[str, Any]]:
    """
    Returns one dict per page of a PDF: {"text": ..., "is_table": bool}.
    Pages with tables are converted with table_to_text(); other pages use
    plain text extraction.
    """
    if HAS_PDFPLUMBER:
        pages = []
        with pdfplumber.open(source) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                table_text = "\n\n".join(t for t in (table_to_text(tb) for tb in tables) if t)
                if table_text:
                    pages.append({"text": table_text, "is_table": True})
                else:
                    pages.append({"text": page.extract_text() or "", "is_table": False})
        return pages

    if PDF_READER_CLS is None:
        raise ImportError("Install 'pdfplumber' or 'pypdf' to read PDF files.")
    reader = PDF_READER_CLS(source)
    return [{"text": page.extract_text() or "", "is_table": False} for page in reader.pages]


def extract_text_from_pdfs(pdf_files: List[Any]) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from uploaded PDF files (or file paths).
    Returns a list of page dicts: [{'text': ..., 'source': filename, 'page': 1}, ...]
    """
    pages_data = []

    for pdf_file in pdf_files:
        if isinstance(pdf_file, str):
            # A file path
            filename = os.path.basename(pdf_file)
            with open(pdf_file, "rb") as f:
                page_texts = _extract_page_texts(io.BytesIO(f.read()))
        else:
            # Streamlit UploadedFile (file-like object)
            filename = getattr(pdf_file, "name", str(pdf_file))
            pdf_bytes = io.BytesIO(pdf_file.read())
            pdf_file.seek(0)
            page_texts = _extract_page_texts(pdf_bytes)

        for page_idx, page in enumerate(page_texts):
            if page["text"].strip():
                pages_data.append({
                    "text": page["text"].strip(),
                    "source": filename,
                    "page": page_idx + 1,
                    "is_table": page["is_table"]
                })

    return pages_data


def _chunk_table_text(text: str, chunk_size: int) -> List[str]:
    """
    Chunks text produced by table_to_text().
    - A grid row (a title line followed by "- " entries) is one chunk. If it is
      longer than chunk_size it is split, and the title is repeated in each piece.
    - Record rows (one line each) are packed together up to chunk_size.
    """
    segments: List[str] = []
    packed = ""
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        lines = para.split("\n")
        if len(lines) > 1:
            title, entries = lines[0], lines[1:]
            piece = title
            for entry in entries:
                if len(piece) + len(entry) + 1 > chunk_size and piece != title:
                    segments.append(piece)
                    piece = title
                piece += "\n" + entry
            segments.append(piece)
        else:
            if packed and len(packed) + len(para) + 1 > chunk_size:
                segments.append(packed)
                packed = ""
            packed = f"{packed}\n{para}" if packed else para
    if packed:
        segments.append(packed)
    return segments


def chunk_documents(pages_data: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[DocumentChunk]:
    """
    Splits page text into overlapping chunks, preserving source and page metadata.
    """
    chunks: List[DocumentChunk] = []
    chunk_counter = 0

    # Table pages: keep each table row whole so its labels stay with its values
    table_pages = [p for p in pages_data if p.get("is_table")]
    pages_data = [p for p in pages_data if not p.get("is_table")]
    for item in table_pages:
        for segment in _chunk_table_text(item["text"], chunk_size):
            chunks.append(DocumentChunk(
                text=segment,
                metadata={"source": item["source"], "page": item["page"]},
                chunk_id=chunk_counter
            ))
            chunk_counter += 1

    if HAS_LANGCHAIN_SPLITTER:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        for item in pages_data:
            text = item["text"]
            split_texts = splitter.split_text(text)
            for text_segment in split_texts:
                if text_segment.strip():
                    chunks.append(DocumentChunk(
                        text=text_segment.strip(),
                        metadata={"source": item["source"], "page": item["page"]},
                        chunk_id=chunk_counter
                    ))
                    chunk_counter += 1
    else:
        # Robust pure-Python chunking fallback
        for item in pages_data:
            text = item["text"]
            start = 0
            text_len = len(text)
            while start < text_len:
                end = min(start + chunk_size, text_len)
                segment = text[start:end]
                if segment.strip():
                    chunks.append(DocumentChunk(
                        text=segment.strip(),
                        metadata={"source": item["source"], "page": item["page"]},
                        chunk_id=chunk_counter
                    ))
                    chunk_counter += 1
                start += chunk_size - chunk_overlap

    return chunks


def _tokenize(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer."""
    return re.findall(r'\b[a-zA-Z0-9_]+\b', text.lower())


class PurePythonTFIDF:
    """Zero-dependency TF-IDF vectorizer for offline or fallback search."""
    def __init__(self, max_features: int = 256):
        self.max_features = max_features
        self.vocabulary_: Dict[str, int] = {}
        self.idf_: Dict[str, float] = {}

    def fit(self, texts: List[str]):
        doc_count = len(texts)
        doc_freq = Counter()
        for t in texts:
            words = set(_tokenize(t))
            for w in words:
                doc_freq[w] += 1

        # Select most common terms
        most_common = [w for w, _ in doc_freq.most_common(self.max_features)]
        self.vocabulary_ = {w: i for i, w in enumerate(most_common)}
        
        # Compute smooth IDF
        for w, df in doc_freq.items():
            if w in self.vocabulary_:
                self.idf_[w] = math.log((1 + doc_count) / (1 + df)) + 1.0

    def transform(self, texts: List[str]) -> List[List[float]]:
        dim = len(self.vocabulary_)
        vectors = []
        for t in texts:
            vec = [0.0] * dim
            tokens = _tokenize(t)
            tf = Counter(tokens)
            total = len(tokens) or 1
            for word, count in tf.items():
                if word in self.vocabulary_:
                    idx = self.vocabulary_[word]
                    vec[idx] = (count / total) * self.idf_.get(word, 1.0)
            vectors.append(vec)
        return vectors


class RAGVectorStore:
    """
    In-memory vector store: embeddings are kept in a normalized NumPy matrix
    and searched with cosine similarity. Uses Gemini embeddings by default.
    """

    def __init__(self, api_key: Optional[str] = None, embedding_provider: str = "gemini"):
        self.api_key = api_key
        self.embedding_provider = embedding_provider
        self.chunks: List[DocumentChunk] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.document_stats: Dict[str, Any] = {}
        self.fallback_tfidf: Optional[PurePythonTFIDF] = None

    def _compute_gemini_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Computes embeddings with the Gemini embedding model, in batches."""
        if not HAS_GENAI:
            raise ImportError("google-generativeai is not installed.")
        
        genai.configure(api_key=self.api_key)
        embeddings = []
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = genai.embed_content(
                model=EMBEDDING_MODEL,
                content=batch,
                task_type="retrieval_document"
            )
            if "embedding" in result:
                batch_embeddings = result["embedding"]
                if isinstance(batch_embeddings[0], float):
                    embeddings.append(batch_embeddings)
                else:
                    embeddings.extend(batch_embeddings)

        return embeddings

    def _compute_query_gemini_embedding(self, query: str) -> List[float]:
        """Computes a single query embedding."""
        genai.configure(api_key=self.api_key)
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        return result["embedding"]

    def build_index(self, chunks: List[DocumentChunk], progress_callback: Optional[callable] = None) -> None:
        """Embeds chunks and creates a normalized numpy matrix for cosine similarity search."""
        if not chunks:
            return

        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]

        if progress_callback:
            progress_callback(0.2, "Computing vector embeddings...")

        if self.embedding_provider == "gemini" and self.api_key:
            embeddings = self._compute_gemini_embeddings(texts)
        else:
            # No API key: fall back to keyword (TF-IDF) vectors
            self.fallback_tfidf = PurePythonTFIDF(max_features=256)
            self.fallback_tfidf.fit(texts)
            embeddings = self.fallback_tfidf.transform(texts)

        for chunk, emb in zip(self.chunks, embeddings):
            chunk.embedding = emb

        matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        self.embeddings_matrix = matrix / norms

        sources = set(c.metadata.get("source", "Unknown") for c in chunks)
        total_words = sum(len(c.text.split()) for c in chunks)
        self.document_stats = {
            "total_documents": len(sources),
            "documents": list(sources),
            "total_chunks": len(chunks),
            "total_words": total_words
        }

        if progress_callback:
            progress_callback(1.0, "Vector store index ready!")

    def similarity_search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """Embeds the query, runs cosine similarity search and returns top_k matching chunks with scores."""
        if self.embeddings_matrix is None or not self.chunks:
            return []

        if self.embedding_provider == "gemini" and self.api_key:
            query_emb = self._compute_query_gemini_embedding(query)
        elif self.fallback_tfidf is not None:
            query_emb = self.fallback_tfidf.transform([query])[0]
        else:
            return []

        q_vec = np.array(query_emb, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec /= q_norm

        scores = np.dot(self.embeddings_matrix, q_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            chunk = self.chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": round(score, 4)
            })

        return results

    def get_document_summary(self) -> str:
        """Returns a concise summary of the loaded documents."""
        if not self.chunks:
            return "No documents currently loaded."

        summary_lines = [
            f"Indexed Documents ({self.document_stats.get('total_documents', 0)}):",
        ]
        for doc in self.document_stats.get("documents", []):
            doc_chunks = [c for c in self.chunks if c.metadata.get("source") == doc]
            summary_lines.append(f"- **{doc}**: {len(doc_chunks)} chunks (~{sum(len(c.text.split()) for c in doc_chunks)} words)")

        summary_lines.append(f"\nTotal: {self.document_stats.get('total_chunks', 0)} chunks, {self.document_stats.get('total_words', 0)} words.")
        return "\n".join(summary_lines)
