from typing import List, Dict
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class QueryEngine:
    """Retrieve relevant chunks for a query"""

    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = FAISSVectorStore()

        try:
            self.vector_store.load_index()
            logger.info("Query engine initialized")
        except Exception as e:
            logger.error(f"Vector index not found: {e}")
            raise

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant chunks"""
        query_embedding = self.embedder.model.encode([query])
        results = self.vector_store.search(query_embedding, k=top_k)

        retrieved_chunks = []
        for chunk, score in results:
            retrieved_chunks.append({
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'relevance_score': 1 / (1 + score)  # Convert distance to similarity
            })

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks for query: {query[:50]}")
        return retrieved_chunks

    def format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"[Document {i}] (Source: {chunk['metadata']['source']})\n"
                f"{chunk['content']}\n"
            )

        return "\n---\n".join(context_parts)