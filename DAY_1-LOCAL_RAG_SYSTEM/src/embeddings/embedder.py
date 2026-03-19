"""
Embedder Module
Generates embeddings using local Sentence Transformers model
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Embedder:
    """Generate embeddings for text chunks using local models"""
    
    def __init__(self, config_path: str = 'src/config/config.yaml'):
        # Load configuration
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        embedding_config = config.get('embedding', {})
        self.model_name = embedding_config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
        self.batch_size = embedding_config.get('batch_size', 32)
        self.embedding_dim = embedding_config.get('dimension', 384)
        
        logger.info(f"🤖 Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        
        # Create embeddings directory
        Path('src/data/embeddings').mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Model loaded successfully (dim={self.embedding_dim})")
    
    def embed_chunks(self, chunks: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        """Generate embeddings for text chunks"""
        texts = [chunk['content'] for chunk in chunks]
        
        logger.info(f"🔢 Generating embeddings for {len(texts)} chunks...")
        logger.info(f"   Batch size: {self.batch_size}")
        
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for better similarity
        )
        
        logger.info(f"✓ Generated embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
        
        return embeddings, chunks
    
    def save_embeddings(self, embeddings: np.ndarray, chunks: List[Dict], 
                       filename: str = 'embeddings.pkl'):
        """Save embeddings and chunks to disk"""
        data = {
            'embeddings': embeddings,
            'chunks': chunks,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim
        }
        
        path = Path('src/data/embeddings') / filename
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        file_size = path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"💾 Saved {len(embeddings)} embeddings to {path} ({file_size:.2f} MB)")
    
    def load_embeddings(self, filename: str = 'embeddings.pkl') -> Tuple[np.ndarray, List[Dict]]:
        """Load embeddings from disk"""
        path = Path('src/data/embeddings') / filename
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = data['embeddings']
        chunks = data['chunks']
        
        logger.info(f"📂 Loaded {len(embeddings)} embeddings from {path}")
        logger.info(f"   Model: {data.get('model_name', 'unknown')}")
        logger.info(f"   Dimension: {data.get('embedding_dim', embeddings.shape[1])}")
        
        return embeddings, chunks


if __name__ == "__main__":
    embedder = Embedder()
    
    # Test with sample chunks
    sample_chunks = [
        {
            'content': 'This is a test document about machine learning.',
            'metadata': {'source': 'test1.txt'}
        },
        {
            'content': 'Another document about artificial intelligence.',
            'metadata': {'source': 'test2.txt'}
        }
    ]
    
    embeddings, chunks = embedder.embed_chunks(sample_chunks)
    
    print(f"\n✅ Generated embeddings")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Sample embedding (first 5 dims): {embeddings[0][:5]}")
    
    # Test save/load
    embedder.save_embeddings(embeddings, chunks, 'test_embeddings.pkl')
    loaded_emb, loaded_chunks = embedder.load_embeddings('test_embeddings.pkl')
    
    print(f"\n✅ Save/Load test successful")
    print(f"   Embeddings match: {np.allclose(embeddings, loaded_emb)}")