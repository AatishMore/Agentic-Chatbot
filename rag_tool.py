

from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import os, uuid, json, hashlib
import numpy as np
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

#  Persistent registry path 
REGISTRY_PATH = "temp_docs/.indexed_registry.json"


def _load_registry() -> dict:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return {}


def _save_registry(registry: dict):
    os.makedirs("temp_docs", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _file_hash(pdf_path: str) -> str:
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class RAGClient:
    def __init__(self):
        self.qdrant = QdrantClient(
            url=os.getenv("QDRANT_HOST"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60,
        )
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = "documents"
        self._ensure_collection()

    def _ensure_collection(self):
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection not in existing:
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    #  Semantic Chunking 
    def _split_into_sentences(self, text: str) -> list[str]:
        import re
        raw = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw if len(s.strip()) > 20]

    def _semantic_chunks(
        self,
        text: str,
        similarity_threshold: float = 0.45,
        max_chunk_tokens: int = 600,
    ) -> list[str]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text]

        print(f"[RAG] Semantic chunking: {len(sentences)} sentences detected")
        embeddings = self.embedder.encode(sentences, show_progress_bar=False)

        chunks, current, current_emb = [], [sentences[0]], [embeddings[0]]

        for i in range(1, len(sentences)):
            centroid = np.mean(current_emb, axis=0)
            sim = np.dot(centroid, embeddings[i]) / (
                np.linalg.norm(centroid) * np.linalg.norm(embeddings[i]) + 1e-8
            )
            word_count = sum(len(s.split()) for s in current)

            if sim < similarity_threshold or word_count > max_chunk_tokens:
                chunks.append(" ".join(current))
                current, current_emb = [sentences[i]], [embeddings[i]]
            else:
                current.append(sentences[i])
                current_emb.append(embeddings[i])

        if current:
            chunks.append(" ".join(current))

        print(f"[RAG] Semantic chunking produced {len(chunks)} chunks")
        return chunks

    # PDF Ingestion with Duplicate Guard 
    def add_pdf(self, pdf_path: str) -> tuple[bool, str]:
        registry = _load_registry()
        content_hash = _file_hash(pdf_path)
        filename = os.path.basename(pdf_path)

        if content_hash in registry:
            already = registry[content_hash]
            print(f"[RAG] Skipping '{filename}' — already indexed as '{already}'")
            return False, f"'{filename}' was already indexed. No duplicates added."

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "".join(p.extract_text() or "" for p in reader.pages)

        chunks = self._semantic_chunks(text)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=self.embedder.encode(chunk).tolist(),
                payload={"text": chunk, "source": pdf_path, "file_hash": content_hash},
            )
            for chunk in chunks
        ]
        self.qdrant.upsert(collection_name=self.collection, points=points)

        registry[content_hash] = filename
        _save_registry(registry)
        print(f"[RAG] Indexed {len(points)} semantic chunks from '{filename}'")
        return True, f"'{filename}' indexed successfully ({len(points)} chunks)."

    # Return raw chunks 
    def search(self, query: str) -> str:
        """
        Returns raw relevant text chunks joined by separator.
        Returns 'NO_RESULTS' if nothing passes the similarity threshold.
        No LLM is called here — formatting is handled by the manager node.
        """
        print(f"[RAG TOOL] Searching internal docs for: {query}")
        vec = self.embedder.encode(query).tolist()

        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=vec,
            limit=8,
            with_payload=True,
            with_vectors=False,
        )

        relevant_chunks = [
            h.payload.get("text", "")
            for h in results.points
            if h.score and h.score > 0.3 and h.payload.get("text")
        ]

        if not relevant_chunks:
            return "NO_RESULTS"

        
        return "\n\n---CHUNK---\n\n".join(relevant_chunks)



_rag_client = RAGClient()


def add_pdf(pdf_path: str) -> tuple[bool, str]:
    return _rag_client.add_pdf(pdf_path)


def get_indexed_files() -> dict:
    return _load_registry()


@tool
def rag_search(query: str) -> str:
    """
    Search internal documents stored in the vector database.
    Returns raw text chunks — no formatting, no LLM.
    Use this tool when the user asks about:
    - internal documentation
    - company policies
    - concepts explained in uploaded PDFs
    - technical knowledge stored in internal docs
    - definitions or explanations from internal knowledge
    Do NOT use this for news, recent events, or public information.
    """
    return _rag_client.search(query)
