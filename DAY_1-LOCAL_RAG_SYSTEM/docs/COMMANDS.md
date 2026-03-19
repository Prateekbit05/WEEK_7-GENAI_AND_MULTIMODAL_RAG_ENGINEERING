# ⚡ COMMANDS.md — Day 1: Local RAG System + Pipeline Architecture

> Quick-reference for every command you need to run, test, and debug the Local RAG pipeline.

---

## 📦 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate              # Linux / macOS
venv\Scripts\activate                 # Windows

# Install all dependencies
pip install -r requirements.txt

# Verify key packages
python -c "import faiss; print('FAISS OK')"
python -c "import sentence_transformers; print('SentenceTransformers OK')"
python -c "import langchain; print('LangChain OK')"
```

---

## 📁 2. Data Preparation

```bash
# Place raw documents into the data folder
cp your_docs/*.pdf      src/data/raw/
cp your_docs/*.txt      src/data/raw/
cp your_docs/*.docx     src/data/raw/
cp your_docs/*.csv      src/data/raw/

# Generate mock/sample data (if real data unavailable)
python create_mock_data.py

# Explore raw data structure
python src/utils/explore_data.py

# Fix any data encoding or formatting issues
bash fix_data.sh

# Process customer CSV specifically
python process_customers.py
```

---

## 🔄 3. Full Ingestion Pipeline

```bash
# Run the complete ingestion pipeline (load → clean → chunk → embed → index)
python run_pipeline.py

# Run via the ingest module directly
python -m src.pipelines.ingest

# Run the main entry point
python run.py

# Run with verbose logging
python run.py --verbose

# Run on a specific file only
python run.py --source src/data/raw/sample.pdf
```

---

## 🧩 4. Individual Pipeline Steps

```bash
# Step 1 — Load and clean documents
python -c "
from src.pipelines.data_loader import DataLoader
loader = DataLoader()
docs = loader.load_all()
print(f'Loaded {len(docs)} documents')
"

# Step 2 — Chunk documents
python -c "
from src.pipelines.chunker import Chunker
chunker = Chunker(chunk_size=600, overlap=100)
chunks = chunker.chunk_all(docs)
print(f'Created {len(chunks)} chunks')
"

# Step 3 — Generate embeddings
python -c "
from src.embeddings.embedder import Embedder
embedder = Embedder()
embedder.embed_and_store(chunks)
print('Embeddings generated')
"

# Step 4 — Verify FAISS index was created
ls -lh src/vectorstore/
python -c "
import faiss
index = faiss.read_index('src/vectorstore/index.faiss')
print(f'FAISS index: {index.ntotal} vectors, dim={index.d}')
"
```

---

## 🔍 5. Querying the RAG System

```bash
# Run a single query via CLI
python -c "
from src.retriever.query_engine import QueryEngine
engine = QueryEngine()
results = engine.query('What are the main topics in these documents?', top_k=5)
for r in results:
    print(r['score'], r['text'][:100])
"

# Run the full RAG pipeline (retrieve + generate)
python -c "
from src.generator.response_generator import ResponseGenerator
gen = ResponseGenerator()
answer = gen.generate('Summarize the key findings from the documents.')
print(answer)
"

# Interactive query loop
python run.py --interactive
```

---

## 📊 6. Evaluation

```bash
# Run the full evaluation suite
python evaluate_rag.py

# Run evaluation module directly
python -m src.evaluation.evaluator

# Run specific test scripts
python test_system.py         # End-to-end system test
python test_generator.py      # Generator/LLM test
python test_llm.py            # LLM model loading test
python test_evaluation.py     # Evaluation metrics test

# View saved evaluation results
cat src/evaluation/results.json
```

---

## 🖥️ 7. Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run dashboard.py

# Open the static HTML dashboard in browser
xdg-open rag_dashboard.html        # Linux
open rag_dashboard.html             # macOS
start rag_dashboard.html            # Windows
```

---

## 📋 8. Logs

```bash
# View live pipeline logs
tail -f logs/pipeline.log

# View last 50 log lines
tail -50 logs/pipeline.log

# Search logs for errors
grep -i "error" logs/pipeline.log
grep -i "warning" logs/pipeline.log

# Clear logs before a fresh run
> logs/pipeline.log
```

---

## 🔧 9. Config Management

```bash
# View current config
cat src/config/config.yaml

# Edit config (chunk size, model, paths)
nano src/config/config.yaml

# Validate config loads correctly
python -c "
from src.config import config
print(config)
"
```

---

## 🧹 10. Cleanup & Reset

```bash
# Remove FAISS index (force re-index on next run)
rm src/vectorstore/index.faiss
rm src/vectorstore/index_metadata.pkl
rm src/vectorstore/customers_index.faiss
rm src/vectorstore/customers_index_metadata.pkl

# Remove all __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# Remove processed chunks (force re-chunk)
rm -rf src/data/chunks/
rm -rf src/data/embeddings/

# Full reset — clean everything except source code
rm -rf src/vectorstore/*.faiss
rm -rf src/vectorstore/*.pkl
rm -rf logs/*.log
rm -rf src/evaluation/results.json
```

---

## 🐛 11. Debugging Tips

```bash
# Check if embeddings model loads
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
vec = model.encode(['test sentence'])
print('Shape:', vec.shape)
"

# Check FAISS read/write
python -c "
import faiss, numpy as np
index = faiss.IndexFlatL2(384)
index.add(np.random.rand(10, 384).astype('float32'))
faiss.write_index(index, '/tmp/test.faiss')
print('FAISS write/read OK, vectors:', index.ntotal)
"

# Check LLM model loads
python test_llm.py

# Print all loaded chunks with metadata
python -c "
import pickle
with open('src/vectorstore/index_metadata.pkl', 'rb') as f:
    meta = pickle.load(f)
print(f'Total chunks: {len(meta)}')
print('Sample:', meta[0])
"
```
