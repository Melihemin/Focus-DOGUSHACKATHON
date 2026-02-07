# FocusFlow

An AI-powered education platform for students with **ADHD**, featuring a carbon-aware RAG architecture that minimizes compute waste while delivering high-quality, personalized explanations.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![License](https://img.shields.io/badge/license-Unspecified-lightgrey.svg)

> **Core Principle:** *"The greenest token is the one never generated."*

---

## What Problem Does This Solve?

Traditional e-learning platforms deliver one-size-fits-all content that overwhelms ADHD learners. Generic LLM assistants hallucinate or drift from course materials, and large cloud models waste energy generating verbose responses no one reads.

**FocusFlow addresses this by:**

- Grounding all answers in your actual course content (DB + PDFs)
- Using efficient local inference (Qwen3:4B via Ollama) instead of large cloud models
- Maximizing retrieval precision to minimize downstream token generation
- Tracking learner attention via eye/mouse tracking for future adaptive response shaping
- Displaying carbon savings to build awareness of AI compute costs

---

## Key Features

| Feature | Status | Description |
|---------|--------|-------------|
| **Hybrid Retrieval** | ✅ Implemented | Dense (FAISS) + Sparse (BM25) with RRF fusion |
| **Query Rewriting** | ✅ Implemented | LLM-based query expansion for better recall |
| **Cross-Encoder Reranking** | ✅ Implemented | Prunes to top-N high-signal chunks |
| **Context Compression** | ✅ Implemented | Extractive compression maximizes info density |
| **Local-First Inference** | ✅ Implemented | Qwen3:4B via Ollama — low watt-per-token |
| **Carbon Estimation** | ✅ Implemented | Client-side CO₂ savings calculation + toast UI |
| **Eye/Mouse Tracking** | ✅ Implemented | Collects focus duration, tab switches, gaze data |
| **ADHD-Optimized Prompts** | ✅ Implemented | Step-by-step explanations with examples |
| **Attention-Adaptive Budgets** | 🔜 Planned | Dynamic token limits based on attention score |
| **Grid-Aware Throttling** | 🔜 Planned | Reduce generation during high-carbon periods |

---

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, SQLAlchemy
- **LLM:** Ollama (local) with Qwen3:4B (configurable)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`
- **Vector Store:** FAISS (persisted to `faiss_index/`)
- **Sparse Retrieval:** BM25 via LangChain
- **Reranker:** CrossEncoder `ms-marco-MiniLM-L-6-v2`
- **Frontend:** Jinja2 templates + Tailwind CSS
- **Database:** SQLite (default) via SQLAlchemy

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai) running locally with a model pulled:
  ```bash
  ollama pull qwen3:4b
  ```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/focusflow.git
cd focusflow

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
python server.py
# Or with uvicorn directly:
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000 in your browser.

### Minimal API Test

```bash
curl -X POST http://localhost:8000/api/rag/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What is cognitive load?"}'
```

---

## Project Structure

```
Focus/
├── server.py              # FastAPI app entry point
├── requirements.txt       # Python dependencies
├── models/
│   └── main.py            # RAG pipeline: retrieval, rerank, compression, LLM
├── routers/
│   └── edu.py             # Course CRUD, eye tracking, AI evaluation endpoints
├── database/
│   ├── models.py          # SQLAlchemy models (Courses, User, EyeTrackingSession)
│   └── settings.py        # DB connection settings
├── templates/             # Jinja2 HTML templates
├── static/                # CSS, JS, images
├── pdfs/                  # Drop PDFs here for ingestion
└── faiss_index/           # Persisted FAISS vector index
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Homepage |
| `POST` | `/api/rag/answer` | Generate RAG answer `{"question": "..."}` |
| `GET` | `/edu/courses` | List all courses |
| `POST` | `/edu/courses` | Add a new course |
| `GET` | `/edu/courses/{id}` | Get course detail + RAG explanation |
| `POST` | `/edu/save-eye-tracking` | Save eye tracking session data |
| `GET` | `/edu/statistics` | User statistics dashboard |

---

## Configuration

Set via `.env` file or environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3:4b` | Model to use for generation |
| `OLLAMA_TIMEOUT` | `60` | Request timeout (seconds) |
| `OLLAMA_MAX_TOKENS` | `512` | Max tokens per response |
| `ENABLE_QUERY_REWRITE` | `true` | Enable query rewriting |
| `ENABLE_HYBRID_RETRIEVAL` | `true` | Enable dense + sparse retrieval |
| `ENABLE_RERANK` | `true` | Enable cross-encoder reranking |
| `ENABLE_COMPRESSION` | `true` | Enable context compression |
| `DENSE_K` | `8` | Number of dense retrieval results |
| `SPARSE_K` | `12` | Number of BM25 results |
| `RERANK_TOP_N` | `4` | Top N after reranking |
| `MAX_CONTEXT_CHARS` | `6000` | Max context size for compression |

---

## Architecture Deep Dive

### Carbon-Aware Adaptive RAG Pipeline

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Rewriting │  ← LRU cached, intent normalization
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│      Hybrid Retrieval           │
│  ┌─────────┐    ┌─────────┐     │
│  │  FAISS  │    │  BM25   │     │  ← Parallel execution
│  │ (dense) │    │(sparse) │     │
│  └────┬────┘    └────┬────┘     │
│       └──────┬───────┘          │
│              ▼                  │
│     RRF Fusion + Dedup          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Cross-Encoder Reranking       │  ← Cached, batched scoring
│   (top-4 high-signal chunks)    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│    Context Compression          │  ← TF-IDF + query overlap
│    (preserve definitions,       │
│     drop low-signal text)       │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Local LLM (Ollama Qwen3:4B)   │  ← Low watt-per-token
│   ADHD-optimized prompt         │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Carbon Estimation (client)    │  ← Toast notification
│   baseline vs local savings     │
└─────────────────────────────────┘
```

### Why Each Layer Matters

| Layer | Token Impact | Engineering Choice |
|-------|--------------|-------------------|
| Query Rewrite | Improves recall → fewer irrelevant chunks | LRU cache avoids repeated LLM calls |
| Hybrid Retrieval | Dense catches semantics, sparse catches exact terms | Parallel ThreadPoolExecutor |
| Reranking | Prunes mediocre context that degrades LLM quality | CrossEncoder with result caching |
| Compression | Maximizes info/token ratio | Sentence-level TF-IDF scoring |
| Local Inference | ~6x lower energy than cloud GPT-4 class models | Ollama with keep_alive for warmth |

### Carbon Estimation Formula

```
Carbon (gCO₂) = Energy (kWh) × Carbon Intensity (gCO₂/kWh)

Baseline (cloud): 0.0009 kWh per 1k tokens
Local (Qwen3:4B): 0.00015 kWh per 1k tokens
Carbon intensity: 0.42 kg CO₂/kWh (global average)

Savings = (Baseline - Local) × tokens × carbon_intensity × 1000
```

---

## Eye Tracking & Attention Data

FocusFlow collects behavioral telemetry to understand learner engagement:

- **Focus duration** (good / warning / alert states)
- **Tab switches** (attention breaks)
- **Session duration**
- **Tracking mode** (webcam or mouse fallback)

Data is stored in `EyeTrackingSession` table for per-user analysis.

**Planned:** Use attention scores to dynamically adjust response length — short summaries for distracted users, detailed explanations for engaged users.

---

## AWS Architecture (Production Reference)

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/f77b95ca-7147-4ce1-9d32-ac7978e17de0" />


**Key decisions:**
- ALB is the only public ingress; all compute in private subnets
- GPU nodes use spot instances with on-demand fallback
- SQS decouples heavy RAG jobs from API latency
- Multi-AZ for RDS and OpenSearch

---

## Roadmap

- [ ] **Attention-adaptive token budgets** — use eye tracking scores to control response length
- [ ] **Grid-aware throttling** — reduce generation during high carbon intensity periods
- [ ] **Dockerfile + docker-compose** — containerized deployment
- [ ] **CI/CD pipeline** — GitHub Actions for lint, test, deploy
- [ ] **Admin UI** — reindex FAISS, manage courses, view analytics
- [ ] **User authentication** — per-user progress and personalization
- [ ] **Benchmark suite** — measure latency, token efficiency, carbon savings

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes with tests where applicable
4. Submit a PR with a clear description

Please keep PRs focused and small. For large changes, open an issue first to discuss.

---

## License

License not yet specified. Contact maintainers before redistribution.

---

## Acknowledgments

- [LangChain](https://langchain.com) for RAG primitives
- [Ollama](https://ollama.ai) for local LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Sentence Transformers](https://sbert.net) for embeddings and reranking
# Focus-DOGUSHACKATHON
