# 🧠 INFO-MODEL.md — Day 1: Model & Component Reference

> Detailed reference for every model, embedding, vector store, and config choice used in the Day 1 Local RAG pipeline.

---

## 📐 System Architecture Overview

```
Raw Documents (PDF / TXT / DOCX / CSV)
         │
         ▼
  ┌─────────────────┐
  │   data_loader   │  ← Loads files, detects type
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │  text_cleaner   │  ← Strips noise, normalises text
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │    chunker      │  ← Splits into 500–800 token chunks with overlap
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │    embedder     │  ← Generates dense vector per chunk
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │   faiss_store   │  ← Stores vectors + metadata index
  └────────┬────────┘
           │
      Query Time
           │
           ▼
  ┌─────────────────┐
  │  query_engine   │  ← Embeds query → searches FAISS → returns top-k chunks
  └────────┬────────┘
           │
           ▼
  ┌──────────────────────┐
  │  response_generator  │  ← Builds prompt → calls LLM → returns answer
  └──────────────────────┘
```

---

## 🤖 LLM Model

| Property | Value |
|---|---|
| Module | `src/models/llm_model.py` |
| Path (API mode) | `src/generator/response_generator.py` |
| Config key | `config.yaml → model.provider` |

### Supported Providers

| Provider | Model Examples | Config Value |
|---|---|---|
| **Local (default)** | Mistral-7B, LLaMA-3, Phi-3, Qwen2 | `provider: local` |
| OpenAI API | GPT-4o, GPT-4o-mini | `provider: openai` |
| Anthropic API | Claude 3.5 Sonnet | `provider: anthropic` |
| Google Gemini API | Gemini 1.5 Pro, Flash | `provider: gemini` |

### Local Model Loading (HuggingFace)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    torch_dtype="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
```

### API Model Loading (Anthropic example)

```python
import anthropic
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": prompt}]
)
```

---

## 🔢 Embedding Model

| Property | Value |
|---|---|
| Module | `src/embeddings/embedder.py` |
| Default Model | `all-MiniLM-L6-v2` |
| Library | `sentence-transformers` |
| Vector Dimension | 384 |
| Output | L2-normalised float32 numpy array |

### Model Comparison

| Model | Dim | Speed | Quality | Use Case |
|---|---|---|---|---|
| `all-MiniLM-L6-v2` ✅ | 384 | Fast | Good | Default, CPU-friendly |
| `all-mpnet-base-v2` | 768 | Medium | Better | Higher quality retrieval |
| `BGE-small-en-v1.5` | 384 | Fast | Better | Enterprise RAG |
| `BGE-base-en-v1.5` | 768 | Medium | Best open | Production RAG |
| `instructor-xl` | 768 | Slow | Excellent | Instruction-aware embeddings |
| `CLIP ViT-B/32` | 512 | Fast | Good | Image + text (Day 3) |

### How Embeddings Are Generated

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Single chunk
vector = model.encode("This is a document chunk.", normalize_embeddings=True)
# → shape: (384,)

# Batch of chunks
vectors = model.encode(list_of_chunks, batch_size=64, normalize_embeddings=True)
# → shape: (N, 384)
```

---

## 🗄️ Vector Store — FAISS

| Property | Value |
|---|---|
| Module | `src/vectorstore/faiss_store.py` |
| Index file | `src/vectorstore/index.faiss` |
| Metadata file | `src/vectorstore/index_metadata.pkl` |
| Library | `faiss-cpu` |
| Index Type | `IndexFlatL2` (exact search) |
| Distance Metric | L2 (Euclidean) |

### Index Types Explained

| Index | Search Type | Speed | Accuracy | When to Use |
|---|---|---|---|---|
| `IndexFlatL2` ✅ | Exact brute-force | Slow at scale | 100% | < 100k vectors, dev/testing |
| `IndexFlatIP` | Exact inner product | Slow at scale | 100% | Cosine similarity (normalised vecs) |
| `IndexIVFFlat` | Approximate | Fast | ~95–99% | 100k–10M vectors |
| `IndexHNSWFlat` | Approximate (graph) | Very fast | ~98% | Production, low latency |
| `IndexIVFPQ` | Compressed approx | Fastest | ~90–95% | Very large scale, memory limited |

### FAISS Operations

```python
import faiss, numpy as np, pickle

# --- Build index ---
dim = 384
index = faiss.IndexFlatL2(dim)
index.add(vectors.astype("float32"))           # Add N vectors
faiss.write_index(index, "index.faiss")
with open("index_metadata.pkl", "wb") as f:
    pickle.dump(metadata_list, f)

# --- Query index ---
index = faiss.read_index("index.faiss")
with open("index_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

query_vec = model.encode([query]).astype("float32")
distances, indices = index.search(query_vec, k=5)  # top-5

results = [
    {"text": metadata[i]["text"], "score": float(distances[0][j]), "source": metadata[i]["source"]}
    for j, i in enumerate(indices[0]) if i != -1
]
```

---

## 🧩 Chunking Strategy

| Property | Value |
|---|---|
| Module | `src/pipelines/chunker.py` |
| Default chunk size | 600 tokens |
| Overlap | 100 tokens |
| Splitter | `RecursiveCharacterTextSplitter` (LangChain) |

### Why These Numbers?

| Parameter | Value | Reason |
|---|---|---|
| Chunk size | 500–800 tokens | Fits in LLM context without losing topic coherence |
| Overlap | 100 tokens | Prevents information loss at chunk boundaries |
| Recursive split | `\n\n → \n → . → ` | Tries to preserve paragraph and sentence boundaries |

### Chunk Metadata Schema

```python
{
    "text":       "The chunk content as a string...",
    "source":     "src/data/raw/policy_manual.pdf",
    "page":       3,
    "chunk_id":   "policy_manual_p3_c2",
    "tags":       ["policy", "finance"],
    "char_count": 421,
    "token_count": 97
}
```

---

## 📄 Document Loaders

| Module | `src/pipelines/data_loader.py` |
|---|---|
| `src/pipelines/image_loader.py` | Image file support |

### Supported File Types

| Format | Loader | Library |
|---|---|---|
| `.pdf` | `PyPDFLoader` | `pypdf` / `pdfplumber` |
| `.txt` | `TextLoader` | built-in |
| `.docx` | `Docx2txtLoader` | `docx2txt` |
| `.csv` | `CSVLoader` | `pandas` |
| `.png / .jpg` | `image_loader.py` | `PIL`, `pytesseract` |

---

## 📝 Prompt Template

| Module | `src/prompts/prompt.py` |
|---|---|

```
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided documents."
Do not make up or infer information beyond what is given.

Context:
{context}

Question: {question}

Answer:
```

### Prompt Design Rules

- **Grounded** — LLM is instructed to answer from context only
- **Refusal clause** — explicit "I don't know" instruction reduces hallucination
- **No system injection** — context is injected at query time, not baked into system prompt
- **Traceable** — source metadata is attached alongside context for citation

---

## ⚙️ Config Reference (`src/config/config.yaml`)

```yaml
model:
  provider: local             # local | openai | anthropic | gemini
  model_name: mistral-7b-instruct
  api_key_env: ANTHROPIC_API_KEY

embeddings:
  model_name: all-MiniLM-L6-v2
  batch_size: 64
  normalize: true

chunking:
  chunk_size: 600
  chunk_overlap: 100
  splitter: recursive

vectorstore:
  type: faiss
  index_path: src/vectorstore/index.faiss
  metadata_path: src/vectorstore/index_metadata.pkl
  index_type: IndexFlatL2

retrieval:
  top_k: 5

data:
  raw_path: src/data/raw/
  chunks_path: src/data/chunks/
  embeddings_path: src/data/embeddings/

logging:
  log_file: logs/pipeline.log
  level: INFO
```

---

## 📊 Evaluation Metrics

| Module | `src/evaluation/evaluator.py` |
|---|---|

| Metric | What It Measures | Formula |
|---|---|---|
| **Context Recall** | Were all relevant chunks retrieved? | Retrieved ∩ Relevant / Relevant |
| **Context Precision** | Were retrieved chunks actually relevant? | Relevant ∩ Retrieved / Retrieved |
| **Answer Faithfulness** | Does the answer stay grounded in context? | Supported claims / Total claims |
| **Answer Relevance** | Does the answer address the question? | Semantic similarity: answer ↔ question |
| **Latency** | End-to-end query response time | milliseconds |

Results are saved to `src/evaluation/results.json` after each evaluation run.
