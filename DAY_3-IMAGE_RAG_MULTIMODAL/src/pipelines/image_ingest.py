"""
Image Ingestion Pipeline
Processes images: OCR extraction + CLIP embeddings + BLIP captions → Vector DB
"""

import json
from pathlib import Path
from typing import List, Dict
import yaml
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.embeddings.clip_embedder import CLIPEmbedder, find_images
from src.pipelines.ocr_extractor import OCRExtractor
from src.pipelines.image_captioner import ImageCaptioner
from src.vectorstore.multimodal_store import MultimodalVectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ImageIngestionPipeline:
    """Complete pipeline for ingesting images into multimodal RAG"""

    def __init__(self, config_path: str = 'src/config/config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        image_config   = self.config.get('image', {})
        self.input_path = Path(image_config.get('input_path', 'src/data/raw/images/'))

        # Initialize components
        self.clip_embedder = CLIPEmbedder(config_path)
        self.ocr_extractor = OCRExtractor(config_path)
        self.captioner     = ImageCaptioner(config_path)
        self.vector_store  = MultimodalVectorStore(config_path)

        logger.info("🚀 Image Ingestion Pipeline initialized")

    def load_images(self) -> List[Path]:
        """
        Load all supported images from input directory.
        Uses find_images() which handles all extensions and subdirectories.
        """
        logger.info(f"📂 Loading images from {self.input_path}")

        if not self.input_path.exists():
            logger.error(f"❌ Input path does not exist: {self.input_path}")
            return []

        image_files = find_images(self.input_path)

        # Debug: show what was found
        if image_files:
            suffixes = sorted(set(p.suffix.lower() for p in image_files))
            logger.info(f"✓ Found {len(image_files)} images "
                        f"(extensions: {suffixes})")
        else:
            # Show what IS in the directory to help diagnose
            all_files = list(self.input_path.rglob('*'))
            all_files = [f for f in all_files if f.is_file()]
            suffixes  = sorted(set(f.suffix.lower() for f in all_files))
            logger.warning(f"⚠️  No images found in {self.input_path}")
            logger.warning(f"   Total files  : {len(all_files)}")
            logger.warning(f"   Extensions   : {suffixes}")
            logger.warning(f"   First 5 files: {[f.name for f in all_files[:5]]}")

        return sorted(image_files)

    def run_pipeline(self, save_outputs: bool = True):
        """Execute complete ingestion pipeline"""

        logger.info("=" * 80)
        logger.info("🚀 STARTING MULTIMODAL IMAGE INGESTION PIPELINE")
        logger.info("=" * 80)

        try:
            # Step 1: Load images
            logger.info("\n[STEP 1/5] 📂 Loading images...")
            image_paths = self.load_images()

            if not image_paths:
                logger.error("❌ No images found! Check input_path in config.yaml")
                logger.error(f"   Current path: {self.input_path.resolve()}")
                return False

            logger.info(f"✓ Loaded {len(image_paths)} images")

            # Step 2: Generate CLIP embeddings
            logger.info("\n[STEP 2/5] 🔢 Generating CLIP embeddings...")
            embeddings = self.clip_embedder.embed_images(image_paths)
            logger.info(f"✓ Generated embeddings (shape: {embeddings.shape})")

            # Step 3: Extract OCR text
            logger.info("\n[STEP 3/5] 📝 Extracting OCR text...")
            ocr_results = self.ocr_extractor.extract_batch(image_paths)

            if save_outputs:
                self.ocr_extractor.save_ocr_results(ocr_results)

            logger.info(f"✓ Extracted text from {len(ocr_results)} images")

            # Step 4: Generate captions
            logger.info("\n[STEP 4/5] 🖼️  Generating BLIP captions...")
            caption_results = self.captioner.generate_captions_batch(image_paths)

            if save_outputs:
                self.captioner.save_captions(caption_results)

            logger.info(f"✓ Generated {len(caption_results)} captions")

            # Step 5: Build metadata and create vector index
            logger.info("\n[STEP 5/5] 🗄️  Creating multimodal vector index...")

            metadata = []
            for i, img_path in enumerate(image_paths):
                meta = {
                    'image_path':     str(img_path),
                    'image_name':     img_path.name,
                    'ocr_text':       ocr_results[i].get('text', ''),
                    'ocr_confidence': ocr_results[i].get('confidence', 0),
                    'caption':        caption_results[i].get('caption', ''),
                    'embedding_index': i,
                }
                metadata.append(meta)

            self.vector_store.create_index(embeddings, metadata)
            self.vector_store.save_index()

            logger.info(f"✓ Vector index created with {len(metadata)} items")

            # ── Summary ──────────────────────────────────────────────────────
            logger.info("\n" + "=" * 80)
            logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)

            print("\n" + "=" * 80)
            print("📊 PIPELINE EXECUTION SUMMARY")
            print("=" * 80)
            print(f"✔ Images processed : {len(image_paths)}")
            print(f"✔ CLIP embeddings  : {embeddings.shape}")
            print(f"✔ OCR extractions  : {len(ocr_results)}")
            print(f"✔ BLIP captions    : {len(caption_results)}")
            print(f"✔ Vector DB        : ✓")
            print("=" * 80)
            print(f"\n📁 Output Files:")
            print(f"   • OCR results  : outputs/ocr/ocr_results.json")
            print(f"   • Captions     : outputs/captions/captions.json")
            print(f"   • Vector index : {self.vector_store.index_path}")
            print(f"   • Metadata     : {self.vector_store.metadata_path}")

            return True

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def run_ingestion():
    """Main entry point"""
    pipeline = ImageIngestionPipeline()
    success  = pipeline.run_pipeline(save_outputs=True)
    return success


if __name__ == "__main__":
    success = run_ingestion()
    sys.exit(0 if success else 1)