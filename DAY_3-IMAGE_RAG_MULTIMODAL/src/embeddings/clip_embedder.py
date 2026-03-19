"""
CLIP Embedder for Multimodal RAG
Generates embeddings for both images and text using open_clip
"""

import torch
import open_clip
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Union
import yaml

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# All supported image extensions (case-insensitive covered via multiple globs)
IMAGE_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.JPG', '*.JPEG', '*.PNG',
                    '*.bmp', '*.BMP', '*.tiff', '*.TIFF', '*.webp', '*.WEBP']


def find_images(directory: Path) -> List[Path]:
    """Find all images in a directory regardless of extension case"""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(directory.glob(ext))
        images.extend(directory.glob(f'**/{ext}'))
    # Deduplicate
    seen, unique = set(), []
    for p in images:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return sorted(unique)


class CLIPEmbedder:
    """Generate CLIP embeddings for images and text using open_clip"""

    def __init__(self, config_path: str = 'src/config/config.yaml'):
        with open(config_path) as f:
            config = yaml.safe_load(f)

        clip_config     = config.get('clip', {})
        # open_clip uses 'ViT-B-32' style names (hyphens, not slashes)
        raw_name        = clip_config.get('model_name', 'ViT-B/32')
        self.model_name = raw_name.replace('/', '-')          # ViT-B/32 → ViT-B-32
        self.pretrained = clip_config.get('pretrained', 'openai')
        self.dimension  = clip_config.get('dimension', 512)
        self.batch_size = clip_config.get('batch_size', 32)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"🖼️  Loading CLIP model: {self.model_name} "
                    f"(pretrained={self.pretrained}) on {self.device}")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

        logger.info(f"✓ CLIP model loaded (dim={self.dimension})")

    # ── Image embedding ───────────────────────────────────────────────────────

    def embed_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Generate L2-normalised embedding for a single image → (512,)"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()

    def embed_images(self, image_paths: List[Union[str, Path]]) -> np.ndarray:
        """Generate L2-normalised embeddings for multiple images → (N, 512)"""
        embeddings = []

        logger.info(f"🔢 Generating CLIP embeddings for {len(image_paths)} images...")

        for i in range(0, len(image_paths), self.batch_size):
            batch_paths  = image_paths[i:i + self.batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_images.append(self.preprocess(img))
                except Exception as e:
                    logger.warning(f"⚠️  Skipping {path}: {e}")
                    continue

            if not batch_images:
                continue

            batch_tensor = torch.stack(batch_images).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(batch_tensor)
                features = features / features.norm(dim=-1, keepdim=True)

            embeddings.append(features.cpu().numpy())

            processed = i + len(batch_paths)
            if processed % (self.batch_size * 10) == 0 or processed == len(image_paths):
                logger.info(f"   Processed {processed}/{len(image_paths)} images")

        if not embeddings:
            logger.error("❌ No embeddings generated!")
            return np.array([])

        all_embeddings = np.vstack(embeddings)
        logger.info(f"✓ Generated {all_embeddings.shape[0]} image embeddings "
                    f"(shape: {all_embeddings.shape})")
        return all_embeddings

    # ── Text embedding ────────────────────────────────────────────────────────

    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Generate L2-normalised embedding for text → (N, 512)"""
        if isinstance(text, str):
            text = [text]

        tokens = self.tokenizer(text).to(self.device)

        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy()

    # ── Similarity ────────────────────────────────────────────────────────────

    def compute_similarity(self,
                           image_embedding: np.ndarray,
                           text_embedding:  np.ndarray) -> float:
        """Cosine similarity between one image embedding and one text embedding"""
        # Both are already L2-normalised → dot product = cosine similarity
        img = image_embedding.flatten()
        txt = text_embedding.flatten()
        return float(np.dot(img, txt))


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nTesting CLIPEmbedder (open_clip backend)...\n")

    embedder = CLIPEmbedder()

    # ── Text embedding test ──────────────────────────────────────────────────
    queries = [
        "a diagram of a system architecture",
        "engineering blueprint with measurements",
        "scanned invoice document",
    ]
    text_embs = embedder.embed_text(queries)
    print(f"✅ Text embeddings shape : {text_embs.shape}")
    print(f"   First query norm      : {np.linalg.norm(text_embs[0]):.4f}  (should be 1.0)")

    # ── Image discovery ──────────────────────────────────────────────────────
    image_dir = Path('src/data/raw/images')

    print(f"\n🔍 Scanning: {image_dir.resolve()}")

    # Show exactly what files and extensions exist
    if image_dir.exists():
        all_files = list(image_dir.iterdir())
        suffixes  = sorted(set(f.suffix.lower() for f in all_files if f.is_file()))
        print(f"   Total files : {len(all_files)}")
        print(f"   Extensions  : {suffixes}")
        print(f"   First 5     : {[f.name for f in all_files[:5]]}")
    else:
        print(f"   ❌ Directory does not exist: {image_dir}")

    sample_images = find_images(image_dir)[:5] if image_dir.exists() else []

    if sample_images:
        print(f"\n✅ Found {len(sample_images)} images")
        for p in sample_images[:5]:
            print(f"   {p.name}")

        # Single image embed
        img_emb = embedder.embed_image(sample_images[0])
        print(f"\n✅ Single image embedding shape : {img_emb.shape}")
        print(f"   Image embedding norm         : {np.linalg.norm(img_emb):.4f}  (should be 1.0)")

        # Batch embed
        batch_embs = embedder.embed_images(sample_images)
        print(f"✅ Batch embeddings shape        : {batch_embs.shape}")

        # Cross-modal similarities
        print(f"\n✅ Cross-modal similarities for: {sample_images[0].name}")
        for q, te in zip(queries, text_embs):
            s   = embedder.compute_similarity(img_emb, te)
            bar = "█" * max(0, int(s * 40))
            print(f"   {s:.4f}  {bar}  {q}")
    else:
        print(f"\n⚠️  No images found in {image_dir}")
        print("   Text embedding test passed — CLIP is working correctly.\n")