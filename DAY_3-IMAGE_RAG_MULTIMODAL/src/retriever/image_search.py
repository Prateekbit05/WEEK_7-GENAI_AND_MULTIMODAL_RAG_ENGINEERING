"""
Image Search Engine
Supports: Text→Image, Image→Image, Image→Text Answer
"""

from pathlib import Path
from typing import List, Dict, Union, Optional
import sys
from PIL import Image
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.clip_embedder import CLIPEmbedder
from src.vectorstore.multimodal_store import MultimodalVectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageSearchEngine:
    """
    Multimodal search engine supporting:
    - Text → Image
    - Image → Image
    - Image → Text Answer
    """
    
    def __init__(self):
        self.clip_embedder = CLIPEmbedder()
        self.vector_store = MultimodalVectorStore()
        
        try:
            self.vector_store.load_index()
            logger.info("🚀 Image Search Engine initialized")
        except FileNotFoundError:
            logger.warning("⚠️  Vector index not found. Run ingestion first.")
    
    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Text → Image search
        
        Args:
            query: Text query (e.g., "engineering diagram")
            top_k: Number of results
        
        Returns:
            List of search results with metadata
        """
        logger.info(f"🔍 Text → Image search: '{query}'")
        
        # Generate text embedding
        text_embedding = self.clip_embedder.embed_text(query)
        
        # Search
        results = self.vector_store.search_text_to_image(text_embedding[0], k=top_k)
        
        # Format results
        formatted_results = []
        for i, (meta, similarity) in enumerate(results, 1):
            formatted_results.append({
                'rank': i,
                'image_path': meta['image_path'],
                'image_name': meta['image_name'],
                'similarity_score': float(similarity),
                'caption': meta.get('caption', ''),
                'ocr_text': meta.get('ocr_text', ''),
                'search_type': 'text_to_image'
            })
        
        logger.info(f"✓ Found {len(formatted_results)} results")
        return formatted_results
    
    def search_by_image(self, image_path: Union[str, Path], top_k: int = 5) -> List[Dict]:
        """
        Image → Image search
        
        Args:
            image_path: Path to query image
            top_k: Number of results
        
        Returns:
            List of similar images
        """
        logger.info(f"🔍 Image → Image search: {Path(image_path).name}")
        
        # Generate image embedding
        image_embedding = self.clip_embedder.embed_image(image_path)
        
        # Search
        results = self.vector_store.search_image_to_image(image_embedding, k=top_k + 1)
        
        # Filter out the query image itself
        formatted_results = []
        for i, (meta, similarity) in enumerate(results):
            if meta['image_path'] != str(image_path):  # Skip query image
                formatted_results.append({
                    'rank': len(formatted_results) + 1,
                    'image_path': meta['image_path'],
                    'image_name': meta['image_name'],
                    'similarity_score': float(similarity),
                    'caption': meta.get('caption', ''),
                    'ocr_text': meta.get('ocr_text', ''),
                    'search_type': 'image_to_image'
                })
                
                if len(formatted_results) >= top_k:
                    break
        
        logger.info(f"✓ Found {len(formatted_results)} similar images")
        return formatted_results
    
    def search_with_answer(
        self,
        query: Union[str, Path],
        top_k: int = 5,
        query_type: str = 'text'
    ) -> Dict:
        """
        Image → Text Answer (retrieves images + generates text answer)
        
        Args:
            query: Text query or image path
            top_k: Number of results
            query_type: 'text' or 'image'
        
        Returns:
            Dict with search results and generated answer
        """
        logger.info(f"🔍 Search with answer (type={query_type})")
        
        # Perform search
        if query_type == 'text':
            results = self.search_by_text(query, top_k)
        else:
            results = self.search_by_image(query, top_k)
        
        # Generate answer from captions and OCR
        answer_parts = []
        for i, result in enumerate(results[:3], 1):  # Use top 3 for answer
            caption = result.get('caption', '')
            ocr = result.get('ocr_text', '')
            
            if caption:
                answer_parts.append(f"{i}. {caption}")
            if ocr and len(ocr) > 10:
                answer_parts.append(f"   Text content: {ocr[:100]}...")
        
        answer = "\n".join(answer_parts) if answer_parts else "No relevant information found."
        
        return {
            'query': str(query),
            'query_type': query_type,
            'results': results,
            'answer': answer,
            'total_results': len(results)
        }
    
    def visualize_results(
        self,
        results: List[Dict],
        query_text: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Visualize search results"""
        
        n_results = len(results)
        if n_results == 0:
            print("No results to visualize")
            return
        
        # Create figure
        cols = min(3, n_results)
        rows = (n_results + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if n_results == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if rows > 1 else axes
        
        # Plot results
        for i, result in enumerate(results):
            ax = axes[i] if n_results > 1 else axes[0]
            
            img_path = result['image_path']
            img = Image.open(img_path)
            
            ax.imshow(img)
            ax.axis('off')
            
            title = f"Rank {result['rank']} (Score: {result['similarity_score']:.3f})\n"
            title += f"{result['image_name']}\n"
            if result.get('caption'):
                title += f"Caption: {result['caption'][:50]}..."
            
            ax.set_title(title, fontsize=10)
        
        # Hide empty subplots
        for i in range(n_results, len(axes)):
            axes[i].axis('off')
        
        if query_text:
            fig.suptitle(f"Query: {query_text}", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"💾 Saved visualization to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    print("Testing Image Search Engine...\n")
    
    engine = ImageSearchEngine()
    
    # Test text → image search
    results = engine.search_by_text("diagram", top_k=3)
    
    print(f"\n✅ Text → Image search returned {len(results)} results:")
    for r in results:
        print(f"   {r['rank']}. {r['image_name']} (score: {r['similarity_score']:.3f})")
