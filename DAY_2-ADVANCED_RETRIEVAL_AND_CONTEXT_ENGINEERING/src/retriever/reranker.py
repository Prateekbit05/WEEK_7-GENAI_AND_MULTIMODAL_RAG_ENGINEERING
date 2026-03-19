"""Reranker using Cross-Encoder with Sigmoid Normalization"""

from pathlib import Path
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder
import numpy as np
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Reranker:
    """Rerank with cross-encoder for higher precision"""
    
    def __init__(self, config_path: str = 'src/config/config.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        rerank_config = config.get('reranking', {})
        self.model_name = rerank_config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.batch_size = rerank_config.get('batch_size', 16)
        
        logger.info(f"🎯 Loading reranker: {self.model_name}")
        self.model = CrossEncoder(self.model_name)
        logger.info(f"✓ Reranker loaded")
    
    def _sigmoid(self, scores: np.ndarray) -> np.ndarray:
        """Normalize raw logits to 0-1 range using sigmoid"""
        return 1 / (1 + np.exp(-scores))
    
    def _confidence_label(self, score: float) -> str:
        """Map normalized score to confidence label"""
        if score >= 0.6:
            return "high"
        elif score >= 0.3:
            return "medium"
        else:
            return "low"
    
    def rerank(self, query: str, chunks: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """
        Rerank chunks using cross-encoder scores normalized via sigmoid.

        Args:
            query: Search query
            chunks: List of chunk dicts with 'content' and 'relevance_score'
            top_k: Number of top results to return (None = return all)

        Returns:
            Reranked list of chunks with updated scores and confidence labels
        """
        if not chunks:
            return []
        
        pairs = [(query, chunk['content']) for chunk in chunks]
        raw_scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # Normalize raw logits → 0 to 1
        normalized_scores = self._sigmoid(np.array(raw_scores))
        
        for i, chunk in enumerate(chunks):
            chunk['rerank_score']   = float(normalized_scores[i])
            chunk['original_score'] = chunk.get('relevance_score', 0.0)
            chunk['confidence']     = self._confidence_label(float(normalized_scores[i]))
        
        reranked = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)
        
        if top_k:
            reranked = reranked[:top_k]
        
        # Update relevance_score to normalized rerank score
        for chunk in reranked:
            chunk['relevance_score'] = chunk['rerank_score']
        
        logger.info(f"✓ Reranked {len(chunks)} → {len(reranked)} chunks")
        return reranked


if __name__ == "__main__":
    reranker = Reranker()

    sample_chunks = [
        {
            'content': 'Banking and mortgage organizations provide financial services including credit underwriting.',
            'metadata': {'source': 'organizations-10000.csv', 'row_id': 1},
            'relevance_score': 0.85
        },
        {
            'content': 'Real estate companies manage property listings and mortgage applications.',
            'metadata': {'source': 'organizations-10000.csv', 'row_id': 2},
            'relevance_score': 0.72
        },
        {
            'content': 'Technology firms develop software solutions for enterprise clients.',
            'metadata': {'source': 'organizations-10000.csv', 'row_id': 3},
            'relevance_score': 0.61
        },
        {
            'content': 'Insurance companies assess risk and process claims for policyholders.',
            'metadata': {'source': 'organizations-10000.csv', 'row_id': 4},
            'relevance_score': 0.55
        },
        {
            'content': 'Retail chains distribute consumer goods across multiple locations.',
            'metadata': {'source': 'organizations-10000.csv', 'row_id': 5},
            'relevance_score': 0.43
        },
    ]

    query = "banking and mortgage companies with large number of employees"

    print(f"\nQuery: {query}")
    print(f"Chunks before reranking: {len(sample_chunks)}")
    print("=" * 60)

    reranked = reranker.rerank(query, sample_chunks, top_k=3)

    print(f"\nTop {len(reranked)} after reranking:\n")
    for i, chunk in enumerate(reranked, 1):
        print(f"--- Result #{i} ---")
        print(f"Rerank Score:   {chunk['rerank_score']:.4f}  (0-1 normalized)")
        print(f"Original Score: {chunk['original_score']:.4f}")
        print(f"Confidence:     {chunk['confidence']}")
        print(f"Content:        {chunk['content']}")
        print()