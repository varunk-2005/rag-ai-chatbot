"""
RAG Engine for PDF Document Ingestion, Chunking, Embeddings, and Retrieval.
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


def extract_text_from_pdfs(pdf_files: List[Any]) -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from uploaded PDF files (or file paths).
    Returns a list of page dicts: [{'text': ..., 'source': filename, 'page': 1}, ...]
    """
    if PDF_READER_CLS is None:
        raise ImportError("Neither 'pypdf' nor 'PyPDF2' is installed. Please install 'pypdf'.")

    pages_data = []

    for pdf_file in pdf_files:
        filename = getattr(pdf_file, "name", str(pdf_file))
        
        # If it's a file path or string
        if isinstance(pdf_file, str):
            with open(pdf_file, "rb") as f:
                reader = PDF_READER_CLS(f)
                num_pages = len(reader.pages)
                for page_idx in range(num_pages):
                    page = reader.pages[page_idx]
                    text = page.extract_text() or ""
                    if text.strip():
                        pages_data.append({
                            "text": text.strip(),
                            "source": os.path.basename(filename),
                            "page": page_idx + 1
                        })
        else:
            # Streamlit UploadedFile (file-like object)
            pdf_bytes = io.BytesIO(pdf_file.read())
            pdf_file.seek(0)
            reader = PDF_READER_CLS(pdf_bytes)
            num_pages = len(reader.pages)
            for page_idx in range(num_pages):
                page = reader.pages[page_idx]
                text = page.extract_text() or ""
                if text.strip():
                    pages_data.append({
                        "text": text.strip(),
                        "source": filename,
                        "page": page_idx + 1
                    })

    return pages_data


def chunk_documents(pages_data: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 150) -> List[DocumentChunk]:
    """
    Splits page text into overlapping chunks, preserving source and page metadata.
    """
    chunks: List[DocumentChunk] = []
    chunk_counter = 0

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
