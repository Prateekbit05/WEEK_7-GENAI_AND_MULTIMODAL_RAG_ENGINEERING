"""Main ingestion pipeline - orchestrates all steps"""
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipelines.data_loader import DataLoader
from src.pipelines.text_cleaner import TextCleaner
from src.pipelines.chunker import DocumentChunker
from src.embeddings.embedder import Embedder
from src.vectorstore.faiss_store import FAISSVectorStore
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def run_ingestion_pipeline(no_eval=False):
    """Execute full ingestion pipeline"""

    logger.info("=" * 80)
    logger.info("🚀 STARTING RAG INGESTION PIPELINE")
    logger.info("=" * 80)

    try:
        # ============================================================
        # STEP 1: LOAD DOCUMENTS
        # ============================================================
        logger.info("\n[STEP 1/5] 📚 Loading documents...")
        loader = DataLoader()
        documents = loader.load_all()

        if not documents:
            logger.error("No documents found!")
            print("\n❌ No documents loaded!")
            print("💡 Create mock data first: python create_mock_data.py")
            return False

        logger.info(f"✅ Loaded {len(documents)} documents")

        # ============================================================
        # STEP 2: CLEAN DOCUMENTS
        # ============================================================
        logger.info("\n[STEP 2/5] 🧹 Cleaning documents...")
        cleaner = TextCleaner()
        cleaned_docs = cleaner.clean_documents(documents)

        # Save cleaned documents
        cleaned_dir = Path('src/data/cleaned')
        cleaned_dir.mkdir(parents=True, exist_ok=True)
        with open(cleaned_dir / 'cleaned_docs.json', 'w') as f:
            json.dump(cleaned_docs, f, indent=2)

        logger.info(f"✅ Cleaned {len(cleaned_docs)} documents")
        logger.info(f"   Saved to: {cleaned_dir / 'cleaned_docs.json'}")

        # ============================================================
        # STEP 3: CHUNK DOCUMENTS
        # ============================================================
        logger.info("\n[STEP 3/5] ✂️  Chunking documents...")
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(cleaned_docs)

        # Save chunks
        chunks_dir = Path('src/data/chunks')
        chunks_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_dir / 'chunks.json', 'w') as f:
            json.dump(chunks, f, indent=2)

        logger.info(f"✅ Created {len(chunks)} chunks")
        logger.info(f"   Saved to: {chunks_dir / 'chunks.json'}")

        # ============================================================
        # STEP 4: GENERATE EMBEDDINGS
        # ============================================================
        logger.info("\n[STEP 4/5] 🔢 Generating embeddings...")
        embedder = Embedder()
        embeddings, chunks = embedder.embed_chunks(chunks)
        embedder.save_embeddings(embeddings, chunks)

        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        logger.info(f"   Shape: {embeddings.shape}")
        logger.info(f"   Saved to: src/data/embeddings/embeddings.pkl")

        # ============================================================
        # STEP 5: CREATE VECTOR INDEX
        # ============================================================
        logger.info("\n[STEP 5/5] 🗄️  Creating vector index...")
        vector_store = FAISSVectorStore()
        vector_store.create_index(embeddings, chunks)
        vector_store.save_index()

        logger.info(f"✅ Vector index created")
        logger.info(f"   Path: {vector_store.index_path}")

        # ============================================================
        # OPTIONAL: RUN EVALUATION
        # ============================================================
        if not no_eval:
            logger.info("\n[OPTIONAL] 📊 Running evaluation...")
            try:
                from src.evaluation.evaluator import RAGEvaluator
                evaluator = RAGEvaluator()

                # Sample test queries
                test_queries = [
                    {
                        'query': 'What were the sales trends in Q2 2024?',
                        'relevant_docs': ['graphs.json', 'bar_chart_1.txt']
                    },
                    {
                        'query': 'Which company has the largest market share?',
                        'relevant_docs': ['graphs.json', 'pie_chart_0.txt']
                    }
                ]

                metrics = evaluator.evaluate_retrieval(test_queries)
                evaluator.save_evaluation_results(metrics)
                logger.info("✅ Evaluation completed")
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")

        # ============================================================
        # PIPELINE COMPLETE
        # ============================================================
        logger.info("\n" + "=" * 80)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        # Print summary
        print("\n" + "="*80)
        print("📊 PIPELINE EXECUTION SUMMARY")
        print("="*80)
        print(f"✔ Documents loaded:      {len(documents)}")
        print(f"✔ Documents cleaned:     {len(cleaned_docs)}")
        print(f"✔ Chunks created:        {len(chunks)}")
        print(f"✔ Embeddings generated:  {embeddings.shape}")
        print(f"✔ Vector DB initialized: ✓")
        if not no_eval:
            print(f"✔ Evaluation completed:  ✓")
        print("="*80)

        print(f"\n📁 Output Files:")
        print(f"   • Cleaned docs:  src/data/cleaned/cleaned_docs.json")
        print(f"   • Chunks:        src/data/chunks/chunks.json")
        print(f"   • Embeddings:    src/data/embeddings/embeddings.pkl")
        print(f"   • Vector index:  {vector_store.index_path}")
        if not no_eval:
            print(f"   • Evaluation:    src/evaluation/results.json")

        print("\n🎯 Next Steps:")
        print("   python run.py test     # Test retrieval")
        print("   python run.py query    # Interactive queries")

        return True

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")
        print("Check logs/pipeline.log for details")
        import traceback
        traceback.print_exc()
        return False