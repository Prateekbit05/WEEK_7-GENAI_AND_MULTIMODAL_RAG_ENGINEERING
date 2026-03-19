"""Hybrid Retriever: Semantic + Keyword + Metadata Filtering"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.retriever.bm25_retriever import BM25Retriever
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class HybridRetriever:
    """Hybrid retrieval: semantic + keyword + metadata filtering"""
    
    def __init__(self, config_path: str = 'src/config/config.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        retrieval_config = config.get('retrieval', {})
        self.semantic_weight = retrieval_config.get('semantic_weight', 0.7)
        self.keyword_weight = retrieval_config.get('keyword_weight', 0.3)
        
        self.embedder = Embedder(config_path)
        self.vector_store = FAISSVectorStore(config_path)
        self.bm25_retriever = BM25Retriever(config_path)
        
        try:
            self.vector_store.load_index()
            self.bm25_retriever.load_index()
            logger.info("✓ Hybrid retriever initialized")
        except FileNotFoundError:
            logger.warning("⚠️  Indices not found. Run ingestion first.")
    
    def _apply_metadata_filter(self, results, filters):
        """Filter results by metadata (year, type, etc.)"""
        if not filters:
            return results
        
        filtered = []
        for chunk, score in results:
            metadata = chunk.get('metadata', {})
            match = True
            
            for key, value in filters.items():
                chunk_value = str(metadata.get(key, '')).lower()
                filter_value = str(value).lower()
                
                # Check in content as well for flexible matching
                content = chunk.get('content', '').lower()
                
                if filter_value not in chunk_value and filter_value not in content:
                    # Also check tags
                    tags = metadata.get('tags', [])
                    if not any(filter_value in str(tag).lower() for tag in tags):
                        match = False
                        break
            
            if match:
                filtered.append((chunk, score))
        
        logger.debug(f"🔍 Filter {filters}: {len(results)} → {len(filtered)} results")
        return filtered
    
    def _normalize_scores(self, scores):
        if not scores or len(scores) == 1:
            return scores
        scores = np.array(scores)
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return [1.0] * len(scores)
        return ((scores - min_s) / (max_s - min_s)).tolist()
    
    def _weighted_fusion(self, semantic_results, keyword_results, top_k):
        sem_scores = self._normalize_scores([s for _, s in semantic_results])
        kw_scores = self._normalize_scores([s for _, s in keyword_results])
        
        chunk_scores = {}
        
        for i, (chunk, _) in enumerate(semantic_results):
            chunk_id = hash(chunk['content'][:100])
            chunk_scores[chunk_id] = {'chunk': chunk, 'score': sem_scores[i] * self.semantic_weight}
        
        for i, (chunk, _) in enumerate(keyword_results):
            chunk_id = hash(chunk['content'][:100])
            if chunk_id in chunk_scores:
                chunk_scores[chunk_id]['score'] += kw_scores[i] * self.keyword_weight
            else:
                chunk_scores[chunk_id] = {'chunk': chunk, 'score': kw_scores[i] * self.keyword_weight}
        
        final = [(d['chunk'], d['score']) for d in chunk_scores.values()]
        final.sort(key=lambda x: x[1], reverse=True)
        return final[:top_k]
    
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict] = None,
                 use_semantic: bool = True, use_keyword: bool = True) -> List[Dict]:
        """
        Hybrid retrieval with metadata filtering
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters e.g. {"year": "2024", "type": "policy"}
            use_semantic: Enable semantic search
            use_keyword: Enable keyword search (fallback)
        """
        retrieval_k = top_k * 3
        semantic_results = []
        keyword_results = []
        
        # Semantic search
        if use_semantic:
            query_embedding = self.embedder.model.encode([query])
            semantic_results = self.vector_store.search(query_embedding, k=retrieval_k)
            semantic_results = self._apply_metadata_filter(semantic_results, filters)
        
        # Keyword search (fallback)
        if use_keyword:
            keyword_results = self.bm25_retriever.search(query, top_k=retrieval_k)
            keyword_results = self._apply_metadata_filter(keyword_results, filters)
        
        # Fusion
        if not semantic_results and keyword_results:
            logger.info("📊 Using keyword-only (semantic empty)")
            final_results = keyword_results[:top_k]
        elif semantic_results and not keyword_results:
            logger.info("📊 Using semantic-only (keyword empty)")
            final_results = semantic_results[:top_k]
        elif semantic_results and keyword_results:
            final_results = self._weighted_fusion(semantic_results, keyword_results, top_k)
        else:
            final_results = []
        
        # Format output
        retrieved = []
        for chunk, score in final_results:
            retrieved.append({
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'relevance_score': float(score),
                'retrieval_method': 'hybrid'
            })
        
        logger.info(f"🔍 Hybrid retrieval: {len(retrieved)} results (filters: {filters})")
        return retrieved
if __name__ == "__main__":
    retriever = HybridRetriever()
    
    # Step 1: No filters — check if retrieval works at all
    print("\n--- TEST 1: No filters ---")
    results = retriever.retrieve(
        query="Explain how credit underwriting works",
        top_k=3
    )
    if not results:
        print("Still no results — retrieval itself is broken.")
    else:
        for i, r in enumerate(results, 1):
            print(f"\nResult #{i} | Score: {r['relevance_score']:.4f}")
            print(f"Metadata: {r['metadata']}")
            print(f"Content: {r['content'][:200]}")
    
    # Step 2: See what metadata keys actually exist in your data
    print("\n--- TEST 2: What metadata do your chunks have? ---")
    query_embedding = retriever.embedder.model.encode(["test"])
    raw = retriever.vector_store.search(query_embedding, k=3)
    for chunk, score in raw:
        print(f"Metadata keys: {list(chunk.get('metadata', {}).keys())}")
        print(f"Metadata values: {chunk.get('metadata', {})}")
        print()