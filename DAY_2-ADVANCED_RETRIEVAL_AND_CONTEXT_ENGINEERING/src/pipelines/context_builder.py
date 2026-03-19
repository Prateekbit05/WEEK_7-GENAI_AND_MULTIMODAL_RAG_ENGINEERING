"""Context Builder: MMR, Deduplication, Window Optimization"""

import numpy as np
import hashlib
import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ContextBuilder:
    """Advanced context building with MMR and deduplication"""
    
    def __init__(self, config_path: str = 'src/config/config.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        context_config = config.get('context', {})
        self.max_tokens = context_config.get('max_tokens', 3000)
        self.mmr_lambda = context_config.get('mmr_lambda', 0.5)
        self.enable_deduplication = context_config.get('enable_deduplication', True)
        self.enable_mmr = context_config.get('enable_mmr', True)
        
        try:
            import tiktoken
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
        
        logger.info(f"📦 Context Builder initialized")
    
    def _count_tokens(self, text: str) -> int:
        if self.encoding:
            return len(self.encoding.encode(text))
        return int(len(text.split()) * 1.33)
    
    def _compute_hash(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm != 0 else 0.0
    
    def deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not self.enable_deduplication:
            return chunks
        
        seen = set()
        unique = []
        for chunk in chunks:
            h = self._compute_hash(chunk.get('content', ''))
            if h not in seen:
                seen.add(h)
                unique.append(chunk)
        
        logger.info(f"🔄 Deduplication: {len(chunks)} → {len(unique)}")
        return unique
    
    def apply_mmr(self, chunks: List[Dict], query_embedding: Optional[np.ndarray] = None, top_k: int = 10) -> List[Dict]:
        if not self.enable_mmr or len(chunks) <= top_k:
            return chunks[:top_k]
        
        if not all('embedding' in c for c in chunks):
            return chunks[:top_k]
        
        embeddings = np.array([c['embedding'] for c in chunks])
        selected = []
        remaining = list(range(len(chunks)))
        
        while len(selected) < top_k and remaining:
            if not selected:
                best = remaining[0]
            else:
                mmr_scores = []
                for idx in remaining:
                    rel = chunks[idx].get('relevance_score', 0.0)
                    max_sim = max([self._cosine_similarity(embeddings[idx], embeddings[s]) for s in selected])
                    mmr = self.mmr_lambda * rel - (1 - self.mmr_lambda) * max_sim
                    mmr_scores.append((idx, mmr))
                best = max(mmr_scores, key=lambda x: x[1])[0]
            
            selected.append(best)
            remaining.remove(best)
        
        logger.info(f"🎲 MMR: {len(chunks)} → {len(selected)}")
        return [chunks[i] for i in selected]
    
    def optimize_context_window(self, chunks: List[Dict]) -> Tuple[List[Dict], int]:
        optimized = []
        total_tokens = 0
        
        for chunk in chunks:
            content = chunk.get('content', '')
            tokens = self._count_tokens(content)
            
            if total_tokens + tokens <= self.max_tokens:
                optimized.append(chunk)
                total_tokens += tokens
                chunk['token_count'] = tokens
            else:
                break
        
        logger.info(f"📏 Context: {len(chunks)} → {len(optimized)} chunks, {total_tokens} tokens")
        return optimized, total_tokens
    
    def add_source_tracking(self, chunks: List[Dict]) -> List[Dict]:
        for i, chunk in enumerate(chunks):
            meta = chunk.get('metadata', {})
            chunk['source_id'] = f"{meta.get('source', 'unknown')}::{meta.get('chunk_id', i)}"
            chunk['retrieval_rank'] = i + 1
            chunk['retrieval_timestamp'] = datetime.datetime.now().isoformat()
            
            score = chunk.get('relevance_score', 0.0)
            chunk['confidence'] = 'high' if score >= 0.8 else 'medium' if score >= 0.5 else 'low'
        
        return chunks
    
    def build_context(self, chunks: List[Dict], query_embedding: Optional[np.ndarray] = None,
                      apply_mmr: bool = True, apply_dedup: bool = True, max_chunks: int = 10) -> Dict:
        logger.info(f"🏗️  Building context from {len(chunks)} chunks...")
        
        if apply_dedup:
            chunks = self.deduplicate_chunks(chunks)
        
        if apply_mmr:
            chunks = self.apply_mmr(chunks, query_embedding, top_k=max_chunks)
        else:
            chunks = chunks[:max_chunks]
        
        chunks, total_tokens = self.optimize_context_window(chunks)
        chunks = self.add_source_tracking(chunks)
        
        # Format context
        parts = []
        sources = []
        for i, chunk in enumerate(chunks, 1):
            src = chunk.get('metadata', {}).get('source', 'unknown')
            parts.append(f"[Document {i}] (Source: {src}, Confidence: {chunk.get('confidence')})\n{chunk['content']}\n")
            sources.append({
                'source_id': chunk.get('source_id'),
                'source': src,
                'relevance_score': chunk.get('relevance_score', 0.0),
                'confidence': chunk.get('confidence')
            })
        
        return {
            'context': "\n---\n".join(parts),
            'chunks': chunks,
            'sources': sources,
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'max_tokens': self.max_tokens,
            'token_utilization': total_tokens / self.max_tokens if self.max_tokens > 0 else 0
        }
