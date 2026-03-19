# 📚 THEORY.md — Day 1: Local RAG System + Pipeline Architecture

> Deep-dive theory reference covering every concept behind the Day 1 Local RAG system — from first principles to production considerations.

---

## Table of Contents

1. [What Is RAG?](#1-what-is-rag)
2. [Why RAG Instead of Fine-Tuning?](#2-why-rag-instead-of-fine-tuning)
3. [RAG Architecture — Full Breakdown](#3-rag-architecture--full-breakdown)
4. [Document Loading & Preprocessing](#4-document-loading--preprocessing)
5. [Text Chunking Strategies](#5-text-chunking-strategies)
6. [Embeddings — Theory & Intuition](#6-embeddings--theory--intuition)
7. [Vector Databases & Indexing](#7-vector-databases--indexing)
8. [FAISS — Deep Dive](#8-faiss--deep-dive)
9. [Retrieval — How Semantic Search Works](#9-retrieval--how-semantic-search-works)
10. [The Generator — LLM Prompting](#10-the-generator--llm-prompting)
11. [Hallucination — Causes & Controls](#11-hallucination--causes--controls)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Local vs API LLMs — Tradeoffs](#13-local-vs-api-llms--tradeoffs)
14. [Production Considerations](#14-production-considerations)
15. [Glossary](#15-glossary)

---

## 1. What Is RAG?

**Retrieval-Augmented Generation (RAG)** is a pattern for grounding an LLM's responses in a specific, curated knowledge base rather than relying solely on its parametric (baked-in training) memory.

### The Core Problem RAG Solves

LLMs like GPT-4 or Mistral-7B are trained on massive internet corpora. Their knowledge has two fundamental limits:

1. **Knowledge cutoff** — They don't know what happened after their training ended.
2. **No private data** — They have never seen your company's internal documents, policies, or databases.

RAG solves both problems by retrieving relevant context at query time and injecting it into the LLM's prompt. The LLM then reasons over fresh, private, up-to-date information it was never trained on.

### The Three-Step Mental Model

```
Step 1 — Retrieve:   Find the most relevant chunks from your knowledge base
Step 2 — Augment:    Inject those chunks into the LLM's context window as "grounding"
Step 3 — Generate:   The LLM answers using only that grounded context
```

### RAG Is Not Search

Search returns documents. RAG synthesises an answer from documents. The LLM acts as a reasoning engine that reads retrieved evidence and generates a coherent, natural language response — it does not just rank or return the source text.

---

## 2. Why RAG Instead of Fine-Tuning?

Fine-tuning is often suggested as an alternative to RAG. Understanding why RAG is preferred for most enterprise use cases is fundamental.

### Side-by-Side Comparison

| Dimension | Fine-Tuning | RAG |
|---|---|---|
| Knowledge update | Requires retraining ($$$) | Update vector DB in minutes |
| Private documents | Encoded into weights (non-auditable) | Stored externally, fully auditable |
| Hallucination risk | High (model "memorises" patterns) | Lower (answers grounded in retrieved text) |
| Traceable sources | No | Yes — every answer cites its chunks |
| Data freshness | Stale until next fine-tune | Always fresh |
| Infrastructure cost | High (GPU training) | Low (embedding + vector search) |
| When it's better | Changing tone, style, format | Changing facts, knowledge, data |

### The Key Insight

Fine-tuning changes **how** the model responds. RAG changes **what** the model knows. For enterprise knowledge systems — policies, manuals, databases, product catalogs — you need to change what the model knows. That is RAG's domain.

---

## 3. RAG Architecture — Full Breakdown

A complete RAG system has two distinct phases: **Ingestion (offline)** and **Querying (online)**.

### Phase 1: Ingestion Pipeline (offline, run once or on schedule)

```
Raw Files
    │
    ▼
[1] Document Loader
    │  Reads PDF, DOCX, TXT, CSV
    │  Extracts raw text + page numbers
    ▼
[2] Text Cleaner
    │  Removes headers/footers, artifacts
    │  Normalises whitespace, encoding
    ▼
[3] Chunker
    │  Splits text into fixed-size overlapping windows
    │  Adds metadata: source, page, chunk_id, tags
    ▼
[4] Embedder
    │  Converts each chunk to a dense vector
    │  e.g., all-MiniLM-L6-v2 → 384-dim float32 vector
    ▼
[5] Vector Store
       Stores vectors + metadata in FAISS (or Qdrant/Chroma)
       index.faiss + index_metadata.pkl
```

### Phase 2: Query Pipeline (online, per user query)

```
User Query (natural language)
    │
    ▼
[1] Query Embedder
    │  Same embedding model as ingestion
    │  Query → 384-dim vector
    ▼
[2] Vector Search
    │  Compute similarity: query vec vs all stored vecs
    │  Return top-k most similar chunks
    ▼
[3] Context Builder
    │  Assembles retrieved chunks into a context block
    │  Deduplicates, ranks, trims to fit context window
    ▼
[4] Prompt Builder
    │  Injects context + question into prompt template
    ▼
[5] LLM Generator
    │  Reads the prompt
    │  Generates answer grounded in context
    ▼
[6] Response + Sources
       Returns answer + cited chunk metadata to user
```

---

## 4. Document Loading & Preprocessing

### Why Preprocessing Matters

Raw documents are messy. PDFs contain headers, footers, page numbers, watermarks, and OCR artifacts. Feeding this noise into the chunker creates polluted chunks that degrade retrieval quality. A clean text cleaner is essential before chunking.

### Common Preprocessing Steps

**1. Whitespace normalisation**
```python
import re
text = re.sub(r'\s+', ' ', text).strip()
```
Multiple spaces, tabs, and newlines become single spaces. This prevents the chunker from creating meaninglessly short "chunks" from blank lines.

**2. Encoding normalisation**
PDFs often produce non-UTF-8 characters — smart quotes (`""`), em-dashes (`—`), ligatures (`ﬁ`). These must be decoded or replaced to avoid tokenisation errors.

**3. Header/Footer removal**
Page headers like `CONFIDENTIAL — Page 3 of 42` and footers like `© 2024 Acme Corp` repeat across every page. If left in, they pollute retrieval — a query about "3" might match every page header.

**4. Table handling**
Tables in PDFs lose their structure on extraction. They become a sequence of raw strings without row/column context. Advanced pipelines convert tables to markdown or JSON before chunking. Basic pipelines skip them.

**5. Metadata extraction**
Every document gets tagged at load time:
```python
{
    "source": "policy_manual_v2.pdf",
    "page": 5,
    "file_type": "pdf",
    "ingested_at": "2026-03-13T10:00:00Z"
}
```
This metadata travels with every chunk and enables filtered retrieval ("only search in policy documents from 2024").

---

## 5. Text Chunking Strategies

Chunking is one of the most impactful decisions in a RAG pipeline. The chunk size determines what the LLM sees as context, and poor chunking is a leading cause of both missed answers and hallucinations.

### Why Chunking Is Necessary

LLMs have a **context window limit** — the maximum number of tokens they can process in a single prompt. Models like Mistral-7B support 8k–32k tokens. A full 200-page policy manual might be 150,000 tokens. You cannot feed the entire document into the prompt. You must select the most relevant portions: that is what chunking + retrieval does.

### Chunking Strategies Compared

**1. Fixed-Size Chunking (most common)**
```
Document → split every N tokens with M tokens overlap
```
- Simple, deterministic, fast
- Risk: splits in the middle of sentences or ideas
- Mitigation: use `RecursiveCharacterTextSplitter` which tries paragraph/sentence boundaries first

**2. Sentence-Based Chunking**
```
Document → split on sentence boundaries → group into windows of K sentences
```
- Cleaner semantic boundaries
- Risk: sentences vary wildly in length; some chunks may be 10 tokens, others 400

**3. Paragraph-Based Chunking**
```
Document → split on double newlines → each paragraph = one chunk
```
- Best semantic coherence per chunk
- Risk: some paragraphs are 20 words (too short) and some are 800 words (too long)

**4. Semantic Chunking (advanced)**
```
Embed every sentence → measure cosine similarity between consecutive sentences
→ split where similarity drops sharply (topic boundary)
```
- Best quality; chunks align with actual topic shifts
- Expensive: requires embedding every sentence twice (before and after deciding split points)

**5. Hierarchical Chunking (advanced)**
```
Document → parent chunks (large, ~1500 tokens) + child chunks (small, ~300 tokens)
Retrieve by child chunks → return parent chunk to LLM for more context
```
- Balances precision (child for retrieval) with completeness (parent for generation)
- Used in LangChain `ParentDocumentRetriever`

### Overlap — Why It's Critical

```
Chunk 1: "...The policy requires annual review. All departments must submit..."
Chunk 2: "All departments must submit compliance reports by December 31..."
```

Without overlap, a question about "who submits compliance reports?" might only partially match Chunk 1 (missing the subject) or Chunk 2 (missing the verb). With overlap of 100 tokens, both chunks contain the full sentence, so either will retrieve correctly.

### Choosing Chunk Size — Rules of Thumb

| Document Type | Recommended Chunk Size | Reason |
|---|---|---|
| Dense technical docs (manuals, policies) | 500–700 tokens | High info density; smaller = more precise |
| Narrative text (reports, articles) | 700–1000 tokens | Context flows across paragraphs |
| FAQs, Q&A documents | 200–400 tokens | Each Q&A pair is a natural unit |
| CSV rows | 1 row per chunk | Each row is semantically atomic |

---

## 6. Embeddings — Theory & Intuition

### What Is an Embedding?

An embedding is a dense, fixed-size numerical vector that encodes the **semantic meaning** of a piece of text. Two texts with similar meaning will have vectors that are geometrically close to each other in the embedding space — regardless of whether they share the same words.

```
"How do I reset my password?"  → [0.12, -0.34, 0.87, ...]  (384 numbers)
"Steps to change my login key" → [0.14, -0.31, 0.89, ...]  (384 numbers)
                                  ↑ Very similar vectors ↑
```

This is the foundation of semantic search. Unlike keyword search (which requires exact word matches), embedding-based search finds conceptually related text even when the vocabulary is entirely different.

### How Embedding Models Are Trained

Most embedding models used in RAG are based on **BERT-style transformers** fine-tuned with **contrastive learning**:

1. Pairs of semantically similar sentences are fed to the model.
2. The model is trained to produce vectors that are **close** for similar pairs and **far apart** for dissimilar pairs.
3. This training objective is called **Siamese Network / Triplet Loss**.

The `sentence-transformers` library provides pre-trained models optimised for this task. The default `all-MiniLM-L6-v2` is a 6-layer transformer distilled from a larger model for CPU-friendly inference with strong retrieval quality.

### Cosine Similarity vs L2 Distance

When comparing two vectors, you need a distance metric:

**Cosine Similarity** measures the angle between vectors (direction, not magnitude):
```
cos(θ) = (A · B) / (‖A‖ × ‖B‖)
Range: -1 (opposite) to +1 (identical)
```

**L2 (Euclidean) Distance** measures straight-line distance:
```
d = √(Σ(Aᵢ - Bᵢ)²)
Range: 0 (identical) to ∞
```

For normalised vectors (unit length), cosine similarity and L2 distance are mathematically equivalent — a higher cosine similarity corresponds to a smaller L2 distance. `sentence-transformers` normalises embeddings by default, which is why FAISS's `IndexFlatL2` still finds the correct semantic nearest neighbours.

### The Curse of Dimensionality

As vector dimensions increase, the difference between the nearest and farthest neighbour shrinks — all points become approximately equidistant. This is why approximate nearest neighbour (ANN) algorithms like HNSW are necessary at scale: exact L2 search becomes unreliable in very high dimensions with very large corpora.

---

## 7. Vector Databases & Indexing

### What Is a Vector Database?

A vector database is a specialised storage system designed to efficiently store, index, and search over high-dimensional dense vectors. Unlike relational databases (which index integers, strings, dates), vector DBs index semantic meaning.

### Core Operations

| Operation | Description |
|---|---|
| `add(vectors, metadata)` | Insert N vectors with associated metadata |
| `search(query_vector, k)` | Find k nearest vectors to the query |
| `delete(id)` | Remove a specific vector |
| `update(id, new_vector)` | Replace a vector (re-index) |
| `filter(search + metadata_condition)` | Semantic search constrained by metadata |

### Vector DB Comparison

| System | Type | Scale | Filtering | Persistence | Best For |
|---|---|---|---|---|---|
| **FAISS** ✅ | Library | Millions | Manual | Files | Fast local prototyping |
| **Chroma** | Embedded DB | 100k–1M | Native | SQLite | Simple local RAG |
| **Qdrant** | Server/embedded | Billions | Native | Disk/cloud | Production enterprise |
| **Weaviate** | Server | Billions | Native | Cloud-native | Multi-modal |
| **Pinecone** | Managed cloud | Billions | Native | Fully managed | Zero-infra cloud |
| **pgvector** | Postgres ext. | Millions | Full SQL | Postgres | If already using Postgres |

FAISS is used in Day 1 because it has zero infrastructure overhead — it's a library, not a server. For production Day 5 deployment, Qdrant or Chroma would be appropriate.

---

## 8. FAISS — Deep Dive

FAISS (Facebook AI Similarity Search) is a C++ library with Python bindings for efficient similarity search over dense vectors. It is the most widely used vector search library in research and production.

### Index Types You Need To Know

**IndexFlatL2 — Exact Brute Force**
```
Search method: Compute L2 distance between query and every stored vector
Complexity: O(N × D) per query
Accuracy: 100% (exact)
```
This is what Day 1 uses. Fine for up to ~100k vectors. Beyond that, latency grows linearly.

**IndexIVFFlat — Inverted File Index (Approximate)**
```
Build: Cluster all vectors into C clusters (using k-means)
Search: Only search vectors in the top nprobe closest clusters
Complexity: O(nprobe × N/C × D)
```
Example: With 1M vectors, C=1024 clusters, nprobe=32 → you search only 3.1% of the corpus per query.

**IndexHNSWFlat — Hierarchical Navigable Small World (Approximate)**
```
Build: Construct a multi-layer proximity graph
Search: Navigate the graph starting from random entry points
Complexity: O(log N)
```
HNSW offers the best query speed with high accuracy (~98%). It is the default index type in most production RAG systems. Memory-intensive but CPU-efficient.

**IndexIVFPQ — Product Quantization (Compressed)**
```
Adds vector compression to IVFFlat
Each vector is compressed from 32-bit floats to 8-bit codes
Memory: 96× reduction vs FlatL2 for dim=768
Accuracy: ~90–95%
```
Used when the corpus is so large it doesn't fit in RAM.

### The FAISS Workflow in This Project

```python
# Ingestion
index = faiss.IndexFlatL2(384)        # Exact search, 384-dim
index.add(vectors)                     # Add all chunk vectors
faiss.write_index(index, "index.faiss")

# Query
index = faiss.read_index("index.faiss")
query_vec = embedder.encode([query]).astype("float32")
distances, indices = index.search(query_vec, k=5)
# distances: shape (1, 5) — L2 distances
# indices: shape (1, 5) — positions in metadata list
```

### Metadata Is Separate

FAISS only stores vectors. It does not store text, filenames, or any other metadata. The `index_metadata.pkl` file stores a Python list where `metadata[i]` corresponds to the vector at position `i` in the FAISS index. You must keep them in sync.

---

## 9. Retrieval — How Semantic Search Works

### The Retrieval Problem

Given a user's question and a corpus of N chunks, find the K chunks most likely to contain the answer. This must be fast (milliseconds), accurate (relevant chunks), and complete (don't miss the answer).

### Semantic Search Steps

```
1. User types: "What is the policy for employee leave?"

2. Embed the query:
   query_vector = embedder.encode("What is the policy for employee leave?")
   # → [0.21, -0.44, 0.83, ...]

3. Search FAISS:
   distances, indices = index.search(query_vector, k=5)
   # Returns 5 most similar chunk positions

4. Fetch metadata:
   results = [metadata[i] for i in indices[0]]

5. Return top-k chunks:
   [
     {"text": "The leave policy states employees are entitled to...", "source": "hr_policy.pdf", "page": 12},
     {"text": "Annual leave accrues at a rate of 1.5 days per...", "source": "hr_policy.pdf", "page": 13},
     ...
   ]
```

### Why Semantic Search Beats Keyword Search

Consider a user asking: **"Can I take time off for illness?"**

- **Keyword search** looks for: `time`, `off`, `illness` → misses chunks that say "sick leave", "medical absence", "health-related leave"
- **Semantic search** embeds the concept of illness-related absence → finds chunks about "sick leave", "medical leave", "health policy" even with zero keyword overlap

This is the defining advantage of embedding-based retrieval.

### Top-K Selection

`top_k=5` is a common default. Increasing k:
- **Pros:** Higher recall (less likely to miss the answer chunk)
- **Cons:** More tokens in the LLM's context window, potentially more noise, higher cost/latency

Typical production ranges:
- Simple factual Q&A: k=3
- Analytical or multi-part questions: k=5–8
- Document summarisation: k=10+

---

## 10. The Generator — LLM Prompting

### The Prompt Is The Product

In RAG, the quality of the final answer depends almost entirely on two things: the quality of the retrieved chunks, and the quality of the prompt that instructs the LLM what to do with them.

### Anatomy of a RAG Prompt

```
[System Instruction]
You are a helpful assistant. Answer questions strictly based on the provided context.
If the answer is not in the context, say "I don't know based on the available documents."

[Context Block]
--- Document 1 (hr_policy.pdf, page 12) ---
The leave policy states employees are entitled to 20 days of annual leave...

--- Document 2 (hr_policy.pdf, page 13) ---
Sick leave accrues at 10 days per year. Medical certification is required for...

[Question]
Can employees carry over unused annual leave to the next year?

[Answer]
```

### Prompt Engineering Principles for RAG

**1. Explicit grounding instruction**
Always tell the LLM to use ONLY the provided context. Without this, the LLM will blend retrieval context with parametric knowledge — producing confident, unfounded answers.

**2. Explicit refusal instruction**
"If the answer is not in the context, say I don't know." Without this, LLMs will invent plausible-sounding answers. This is hallucination.

**3. Source attribution**
Include source filenames and page numbers in the context block. Instruct the LLM to cite them. This makes answers auditable.

**4. Context ordering**
Place the most relevant (highest-scoring) chunks closest to the question. LLMs exhibit "recency bias" — they pay more attention to content near the end of their context window.

**5. Context window management**
Context windows have hard limits. For Mistral-7B: 8192 tokens. For GPT-4o: 128k tokens. Calculate: `max_context_tokens = context_window - system_prompt_tokens - question_tokens - answer_budget_tokens`. Truncate the context block if needed.

---

## 11. Hallucination — Causes & Controls

### What Is Hallucination?

Hallucination in LLMs refers to the model generating confident, fluent, syntactically correct statements that are factually incorrect or entirely fabricated. In a RAG system, hallucination typically manifests in two forms:

1. **Faithful hallucination** — the answer looks like it comes from the context but contains subtle distortions (paraphrased incorrectly, numbers changed)
2. **Parametric hallucination** — the LLM ignores the retrieved context and answers from its training data, which may be outdated, wrong, or irrelevant

### Root Causes

| Cause | Description |
|---|---|
| No grounding instruction | LLM defaults to parametric memory |
| Irrelevant retrieved chunks | Context doesn't contain the answer; LLM invents one |
| Chunk boundary splits | Key facts split across chunks; neither chunk is fully informative |
| Over-long context | LLM loses focus in a 10,000-token context; misses key sentences |
| High temperature | Higher temperature = more creative = more hallucination |
| Ambiguous question | LLM guesses at intent and answers a different question |

### Controls Used in This Pipeline

| Control | Implementation |
|---|---|
| Strict grounding prompt | "Use ONLY the context below" |
| Explicit refusal clause | "Say I don't know if not in context" |
| Low temperature | `temperature=0.1–0.3` for factual tasks |
| Source citation | Context chunks include source + page metadata |
| Faithfulness scoring | Evaluator checks answer tokens against context tokens |
| Top-k tuning | Retrieve enough chunks to actually contain the answer |

---

## 12. Evaluation Metrics

### Why Evaluation Matters

You cannot improve what you cannot measure. RAG evaluation measures whether the system is retrieving the right content and generating faithful, accurate answers.

### Retrieval Metrics

**Context Recall**
```
= (Retrieved chunks that are relevant) / (Total relevant chunks)
```
Measures: "Did we find all the evidence we needed?"
High recall = low miss rate. The opposite of recall is missing the answer entirely.

**Context Precision**
```
= (Retrieved chunks that are relevant) / (Total retrieved chunks)
```
Measures: "Were the chunks we retrieved actually useful?"
High precision = low noise in the context. More noise = LLM more likely to hallucinate.

### Generation Metrics

**Answer Faithfulness**
```
= (Claims in answer supported by context) / (Total claims in answer)
```
This is the primary anti-hallucination metric. A score of 1.0 means every statement in the answer can be directly traced to the retrieved context.

**Answer Relevance**
```
= Cosine similarity between (question embedding) and (answer embedding)
```
Measures: "Did the answer actually address the question asked?"
An answer can be faithful (grounded in context) but irrelevant (answers a tangential point instead of the actual question).

### The Evaluation Stack

For production RAG evaluation, the RAGAS framework provides automated scoring across all four metrics using an LLM-as-judge approach. For Day 1, a simpler overlap-based faithfulness score and manual evaluation are sufficient.

---

## 13. Local vs API LLMs — Tradeoffs

### When to Use Local Models

| Scenario | Reason |
|---|---|
| Data must never leave the building | GDPR, HIPAA, financial regulations |
| No internet access (air-gapped systems) | Government, defence, critical infrastructure |
| Low per-query cost at high volume | API calls at 10M queries/month become expensive |
| Customisation required | Fine-tune on proprietary style or domain |

**Practical constraint:** Local models (Mistral-7B, LLaMA-3) require at minimum a machine with 16GB RAM for CPU inference, or a GPU with 8GB VRAM for reasonable speed. Inference on CPU is functional but slow (~5–15 seconds per query for 7B models).

### When to Use API Models

| Scenario | Reason |
|---|---|
| No GPU available | API handles all compute |
| Best quality required | GPT-4o, Claude 3.7 outperform local 7B models significantly |
| Fast prototyping | No model download or quantisation setup |
| Low internal traffic | API cost is affordable at moderate query volumes |

### Quality Comparison (Practical)

| Task | Mistral-7B | GPT-4o | Claude 3.7 Sonnet |
|---|---|---|---|
| Simple factual Q&A | Good | Excellent | Excellent |
| Complex multi-document reasoning | Moderate | Excellent | Excellent |
| Instruction following | Good | Excellent | Excellent |
| Refusal to hallucinate | Moderate | Strong | Strong |
| Cost per 1M tokens | ~$0 (local) | ~$15 | ~$15 |

### The Provider-Switch Pattern

This project's architecture is provider-agnostic. The only file that changes between local and API mode is `src/models/llm_model.py`. All upstream pipeline code (chunking, embedding, retrieval, prompting) is identical regardless of the chosen provider. This is intentional — it allows the same RAG system to run in an air-gapped factory (local Mistral) or a cloud-hosted enterprise portal (GPT-4o) without changing anything else.

---

## 14. Production Considerations

### Scalability

**The bottleneck shifts as scale increases:**

- At < 10k chunks: FAISS IndexFlatL2 is fine. Embedding is the bottleneck.
- At 100k–1M chunks: Switch to HNSW or IVFFlat. Retrieval becomes the bottleneck.
- At > 10M chunks: Distributed vector DB required (Qdrant cluster, Weaviate). Chunking and embedding must be parallelised.

### Freshness

RAG systems go stale when the underlying documents change. Production pipelines need:
- **Change detection:** Hash documents on ingest; detect when a source file is modified
- **Incremental re-ingestion:** Re-chunk, re-embed, and replace only the changed documents in the index (not a full re-index)
- **Metadata timestamps:** Tag every chunk with ingestion date so staleness can be audited

### Security

- Never log raw query text containing PII without redaction
- Chunk metadata (source filenames, page numbers) may be sensitive — apply access control before returning sources to users
- In API mode, ensure `api_key_env` variables are never committed to git (use `.env.example` + `.env` pattern)

### Latency Budget (Typical per-query)

| Step | Typical Latency |
|---|---|
| Query embedding | 20–80ms (CPU) |
| FAISS search (100k vectors) | 5–20ms |
| Prompt assembly | < 1ms |
| LLM generation (local 7B, CPU) | 5,000–15,000ms |
| LLM generation (GPT-4o API) | 800–2,000ms |
| **Total (local)** | **~6–16 seconds** |
| **Total (API)** | **~1–2.5 seconds** |

---

## 15. Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation — grounding LLM responses in retrieved documents |
| **Embedding** | A dense numerical vector encoding the semantic meaning of text |
| **Chunk** | A fixed-size segment of a document with overlap and metadata |
| **Vector Store** | A database optimised for storing and searching high-dimensional vectors |
| **FAISS** | Facebook AI Similarity Search — C++ library for fast vector search |
| **HNSW** | Hierarchical Navigable Small World — graph-based approximate nearest neighbour index |
| **IVF** | Inverted File Index — cluster-based approximate nearest neighbour index |
| **Top-K** | The K most similar vectors returned by a similarity search |
| **Cosine Similarity** | Angle-based distance metric for comparing vector direction (semantic meaning) |
| **L2 Distance** | Euclidean distance between two vectors in N-dimensional space |
| **Hallucination** | LLM generating confident but factually incorrect or unsupported statements |
| **Faithfulness** | Whether every claim in an LLM answer is directly supported by retrieved context |
| **Context Window** | Maximum token capacity of an LLM in a single prompt+completion turn |
| **Parametric Memory** | Knowledge encoded in an LLM's weights during training |
| **Semantic Search** | Finding similar content based on meaning rather than keyword matching |
| **BM25** | Best Match 25 — a classical keyword-frequency ranking algorithm (used in hybrid retrieval Day 2) |
| **Reranker** | A cross-encoder model that re-scores retrieved chunks for higher precision (Day 2) |
| **MMR** | Maximal Marginal Relevance — selects diverse, non-redundant chunks from retrieval results |
| **CLIP** | Contrastive Language–Image Pretraining — OpenAI model that embeds images and text into a shared vector space (Day 3) |
| **OCR** | Optical Character Recognition — extracting text from scanned images (Day 3) |
| **PSI** | Population Stability Index — measures data distribution shift (used in ML monitoring Week 6) |
| **Temperature** | LLM sampling parameter: 0 = deterministic, 1 = creative. Use low values (0.1–0.3) for factual RAG |
| **Token** | Approximate unit of text processed by LLMs. ~1 token ≈ 0.75 English words |
| **Context Recall** | Retrieval metric: fraction of relevant chunks that were actually retrieved |
| **Context Precision** | Retrieval metric: fraction of retrieved chunks that were actually relevant |
