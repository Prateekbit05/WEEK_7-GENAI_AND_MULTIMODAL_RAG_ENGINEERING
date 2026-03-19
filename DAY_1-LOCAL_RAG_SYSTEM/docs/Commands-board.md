# RAG Pipeline Dashboard — Commands

## 1. Install Dependencies (inside your venv)

```bash
# Activate your existing venv
source venv/bin/activate

# Core dashboard deps
pip install streamlit tiktoken faiss-cpu sentence-transformers

# For PDF loading (already have this)
pip install pypdf

# For DOCX loading (already have this)
pip install python-docx

# Vector store backends (choose what you need)
pip install chromadb           # Chroma
pip install qdrant-client      # Qdrant (also needs Docker)

# Optional: heavier embedding models
pip install InstructorEmbedding   # for Instructor-XL
```

---

## 2. Run the Dashboard

```bash
# From your DAY_1 project root
cd ~/Documents/HESTABIT_ALL_TASKS/WEEK_7-GENAI_AND_MULTIMODAL_RAG_ENGINEERING/DAY_1-LOCAL_RAG_SYSTEM

# Run Streamlit
streamlit run rag_streamlit_dashboard.py

# Custom port if 8501 is busy
streamlit run rag_streamlit_dashboard.py --server.port 8502
```

Opens at: http://localhost:8501

---

## 3. Qdrant (if using Qdrant backend)

```bash
# Pull and run Qdrant container
docker pull qdrant/qdrant
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/src/vectorstore/qdrant_data:/qdrant/storage \
  qdrant/qdrant

# Verify it's running
curl http://localhost:6333/healthz
```

---

## 4. Pipeline Flow (inside the dashboard)

```
Tab 1 → Upload files OR scan src/data/raw/
Tab 2 → Clean text → Chunk (500–800 tokens, configurable)
Tab 3 → Generate embeddings (BGE / GTE / Instructor)
Tab 4 → Build vector index (FAISS / Chroma / Qdrant)
Tab 5 → Query the retriever
Tab 6 → View stats & logs
```

---

## 5. File Placement

Place the dashboard file in your project root:

```
DAY_1-LOCAL_RAG_SYSTEM/
├── rag_streamlit_dashboard.py   ← this file
├── src/
│   ├── pipelines/
│   ├── embeddings/
│   ├── vectorstore/
│   └── ...
└── requirements.txt
```

---

## 6. Quick one-liner (full pipeline test)

```bash
# Run the original CLI pipeline (still works)
python run_pipeline.py

# Then open dashboard for visual inspection
streamlit run rag_streamlit_dashboard.py
```