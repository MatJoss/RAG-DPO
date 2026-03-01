# 🔒 RAG-DPO — GDPR Assistant for DPOs

> 🇫🇷 [Version française](README.md)

> A fully local RAG (Retrieval-Augmented Generation) system to assist Data Protection Officers, built on official CNIL sources.

## 🎯 Purpose

A GDPR expert assistant that:
- **Answers exclusively from verified CNIL sources** (zero hallucination)
- **Cites its sources** with references to original documents
- **Runs entirely locally** — no data leaves the machine
- **Handles nuances** — automatically detects contradictions between sources

## 📊 Performance

**Overall score: 93%** on an 18-question GDPR/CNIL benchmark covering 5 categories.

| Metric | Score |
|---|---|
| 📈 Overall Score | **93%** |
| ✅ Correctness (LLM Judge + Keywords) | **84%** |
| 🛡️ Faithfulness (source fidelity) | **100%** |
| 📏 Conciseness | **98%** |
| 📚 Source Quality | **97%** |
| ⏱️ Avg. time/question | **17.3s** |

### By category

| Category | Score | Questions |
|---|---|---|
| 📖 Definitions | **97%** | 5 |
| ⚖️ Obligations | **95%** | 4 |
| 🪤 Tricky questions | **92%** | 2 |
| 💡 Recommendations | **91%** | 5 |
| 🚫 Out of scope | **86%** | 2 |

### Performance evolution

The system was built iteratively. Each pipeline component was evaluated on the same 18-question benchmark:

| Version | Configuration | Overall | Correctness | Time/q |
|---|---|---|---|---|
| v1 — Baseline | Semantic only, no reranker | 86% | 65% | 6.3s |
| v2 — Query Expansion | + LLM multi-query (×3 reformulations) | 89% | 73% | 13.2s |
| v3 — Cross-Encoder | + BGE reranker v2 m3 (568M) | 92% | 78% | 8.2s |
| v4 — Jina Reranker | BGE → Jina v2 multilingual (278M, 7× faster) | 92% | 83% | 9.5s |
| v5 — Rechunking | 50-word overlap, heading propagation, semantic split | **93%** | **84%** | 31.9s |
| v6 — BM25 Boost | BM25 weight ×1.5, eval keywords fix | 92% | 81% | 14.0s |
| **v7 — Dual Generation** | **Self-consistency via context order** | **93%** | **84%** | **17.3s** |

#### Per-component gains

```
Semantic only                   86% ─────────────────────┐
+ LLM Query Expansion           89%  (+3%)               │ Augmented
+ Cross-Encoder Reranking       92%  (+3%)               │ Retrieval
+ Smart Rechunking              93%  (+1%)  ─────────────┘
+ Dual Generation               93%  (stability +         
                                      correctness 84%)    → Robustness
```

**Key contribution of each component:**

| Component | Primary impact | Gain |
|---|---|---|
| **Query Expansion** | Better recall — reformulations capture GDPR synonyms | +3% overall, +8% correctness |
| **Cross-Encoder** | Better precision — fine reranking vs coarse cosine | +3% overall |
| **Rechunking** | Self-contained chunks — overlap + heading + semantic split | +1% overall, +6% correctness |
| **Dual Generation** | Robustness — detects contradictions between sources | +2% correctness, q10 63%→89% |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG-DPO Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Question                                                      │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ Query Expansion  │  LLM generates 3 reformulations               │
│  │ (multi-query)    │  + GDPR acronym expansion                     │
│  └────────┬────────┘                                                │
│           ▼                                                         │
│  ┌─────────────────┐     ┌──────────────┐                           │
│  │ Summary         │───> │ BM25 Index   │  Pre-filter: top-40       │
│  │ Pre-Filter      │     │ (summaries)  │  relevant documents       │
│  └────────┬────────┘     └──────────────┘                           │
│           ▼                                                         │
│  ┌─────────────────────────────────────────┐                        │
│  │       Hybrid Retrieval (×4 queries)     │                        │
│  │  ┌──────────┐      ┌──────────────────┐ │                        │
│  │  │ BM25     │      │ ChromaDB Semantic│ │                        │
│  │  │ (sparse) │      │ (BGE-M3, 1024d)  │ │                        │
│  │  └────┬─────┘      └────────┬─────────┘ │                        │
│  │       └──────┬──────────────┘           │                        │
│  │              ▼                          │                        │
│  │     RRF Fusion (k=60)                   │                        │
│  │     BM25 weight: ×1.5                   │                        │
│  └──────────────┬──────────────────────────┘                        │
│                 ▼                                                   │
│  ┌─────────────────┐                                                │
│  │ Jina Reranker   │  Multilingual cross-encoder                    │
│  │ v2 (278M, CPU)  │  40 candidates → top 10                        │
│  └────────┬────────┘                                                │
│           ▼                                                         │
│  ┌─────────────────────────────────────────┐                        │
│  │        Dual Generation                  │                        │
│  │  ┌──────────────┐ ┌──────────────────┐  │                        │
│  │  │ Pass A       │ │ Pass B           │  │                        │
│  │  │ (natural     │ │ (reverse         │  │                        │
│  │  │  order)      │ │  order)          │  │                        │
│  │  └──────┬───────┘ └────────┬─────────┘  │                        │
│  │         └────────┬─────────┘            │                        │
│  │                  ▼                      │                        │
│  │     Stance Comparison                   │                        │
│  │     ┌────────────┬──────────────┐       │                        │
│  │     │ Concordant │ Contradiction│       │                        │
│  │     │ → Pass A   │ → Synthesis  │       │                        │
│  │     └────────────┴──────────────┘       │                        │
│  └──────────────┬──────────────────────────┘                        │
│                 ▼                                                   │
│  ┌─────────────────┐                                                │
│  │ Grounding       │  Verifies [Source X] citations                 │
│  │ Validation      │                                                │
│  └────────┬────────┘                                                │
│           ▼                                                         │
│     Final answer + cited sources                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Dual Generation — Self-Consistency via Context Order

The key innovation: the system generates **two answers** from the same documents but in **different order**, then compares stances:

- **Concordant** (same conclusion) → the answer is reliable, return it
- **Contradiction** (opposite conclusions) → sources cover **different cases** → nuanced synthesis via a 3rd LLM call

This mechanism solved the system's most persistent problem (q10: *"Can legitimate interest be used for video surveillance?"*), where a chunk specific to municipalities ("legitimate interest cannot be invoked") contradicted the general rule ("yes, with balancing test"). Score: 63% → 89%.

## 🛠️ Tech Stack

| Component | Technology | Details |
|---|---|---|
| **LLM** | Mistral-Nemo 12B | Via Ollama, 128K context, temperature 0.0 |
| **Embeddings** | BGE-M3 (BAAI) | sentence-transformers, 1024 dims, FP16 GPU |
| **VectorDB** | ChromaDB | PersistentClient, 14,388 chunks |
| **Reranker** | Jina Reranker v2 | 278M params, multilingual, CPU |
| **BM25** | rank_bm25 | Sparse index for hybrid search |
| **Interface** | Streamlit multipage | Chat + Observability dashboard |
| **Observability** | JSONL + Alerter | Structured logs, feedback, SMTP alerts |
| **GPU** | RTX 4070 Ti 12GB | LLM + embeddings in VRAM, reranker on CPU |

### Project structure

```
RAG-DPO/
├── app.py                      # Streamlit multipage entry point
├── pages/
│   ├── 1_💬_Chat.py            # Interactive RAG chat + feedback
│   └── 2_📊_Dashboard.py       # Observability dashboard (metrics, alerts)
├── test_rag.py                 # CLI RAG testing
├── check_install.py            # Installation verification
├── rebuild_pipeline.py         # Data pipeline rebuild
├── requirements.txt            # Python dependencies
├── configs/
│   ├── config.yaml             # Centralized config (RAG + observability + SMTP)
│   └── enterprise_tags.json    # Enterprise tag registry (auto-generated)
├── src/
│   ├── rag/                    # 🧠 RAG core
│   │   ├── pipeline.py         # Orchestration (dual-gen, stance detection)
│   │   ├── retriever.py        # Hybrid retrieval (BM25 + semantic + RRF)
│   │   ├── query_expander.py   # Multi-query expansion via LLM
│   │   ├── bm25_index.py       # BM25 index (summaries + chunks)
│   │   ├── reranker.py         # Cross-encoder Jina reranking
│   │   ├── context_builder.py  # Context building + reverse packing
│   │   ├── generator.py        # LLM generation (Ollama)
│   │   └── validators.py       # Grounding + relevance validation
│   ├── processing/             # 📄 Data processing pipeline
│   │   ├── ingest_enterprise.py        # Enterprise doc ingestion (PDF, DOCX, XLSX…)
│   │   ├── process_and_chunk.py        # Semantic chunking
│   │   ├── create_chromadb_index.py    # Vector indexing
│   │   ├── generate_document_summaries.py  # LLM summary sheets
│   │   ├── hybrid_filter.py            # Relevance filtering
│   │   ├── classify_documents.py       # Document classification
│   │   └── ...
│   ├── scraping/               # 🕷️ CNIL scraping
│   │   └── cnil_scraper_final.py
│   └── utils/
│       ├── llm_provider.py     # Ollama interface
│       ├── embedding_provider.py # BGE-M3 provider (FP16, GPU, lazy load)
│       ├── query_logger.py     # JSONL query & feedback logger
│       ├── structured_logger.py # JSON structured logging
│       ├── alerter.py          # Threshold alerts + SMTP
│       └── acronyms.py         # GDPR acronym expansion
├── eval/                       # 📊 Evaluation framework
│   ├── qa_dataset.json         # 18-question benchmark (5 categories)
│   ├── run_eval.py             # 2-phase evaluation (keywords + LLM judge)
│   └── results_*.json          # Historical results
├── logs/                       # 📝 Structured logs (not versioned)
│   ├── app.jsonl               # Application JSON logs
│   ├── queries.jsonl           # Query history
│   ├── feedback.jsonl          # User feedback 👍/👎
│   └── alerts.jsonl            # Alert history
├── data/                       # 📁 Data (not versioned)
│   ├── raw/                    # Raw CNIL documents
│   ├── vectordb/chromadb/      # ChromaDB vector database
│   └── metadata/               # Document metadata
└── tasks/                      # 📝 Work notes
    ├── todo.md
    └── lessons.md              # Lessons learned (patterns, mistakes, fixes)
```

## 🚀 Installation

### Prerequisites

- **Python 3.11+**
- **NVIDIA GPU** with ≥8 GB VRAM (RTX 3070+ recommended)
- **Ollama** installed and running

### 1. Clone and install

```bash
git clone https://github.com/MatJoss/RAG-DPO.git
cd RAG-DPO
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Install Ollama models

```bash
ollama pull mistral-nemo        # LLM 12B (7.1 GB)
```

> **Note**: BGE-M3 embeddings (sentence-transformers) are downloaded automatically on first run.

### 3. Verify installation

```bash
python check_install.py
```

### 4. Build the database (optional)

If starting from scratch with your own CNIL data:

```bash
python rebuild_pipeline.py          # Full pipeline (scraping → indexing)
python rebuild_pipeline.py --from 5b  # Resume from chunking
```

## 💬 Usage

### Streamlit interface (recommended)

```bash
streamlit run app.py
```

Opens the multipage application in the browser:

| Page | Description |
|---|---|
| 🏠 **Home** | System overview, statistics |
| 💬 **Chat** | RAG Q&A interface with cited sources and 👍/👎 feedback |
| 📊 **Dashboard** | Real-time metrics, alerts, feedback, JSON export |

Chat features:
- Document type filtering (Doctrine, Guide, Sanction, Technical)
- Enterprise tag filtering (if internal docs imported)
- Cited sources with [CNIL] / [Internal] distinction
- User feedback 👍/👎 logged to JSONL

### Command line

```bash
python test_rag.py "When is a DPIA mandatory?"
```

### Evaluation

```bash
python eval/run_eval.py --verbose
```

Runs the 18-question benchmark in 2 phases:
1. **Phase 1**: RAG generation + keyword scoring
2. **Phase 2**: LLM-as-Judge (the LLM evaluates semantic quality)

Final score = 70% LLM Judge + 30% Keywords

## 📈 Detailed benchmark (v7 — Dual Generation)

| # | Question | Cat. | Score | Time |
|---|---|---|---|---|
| q01 | What is personal data? | Definition | **98%** | 23.1s |
| q02 | Who is the data controller? | Definition | **100%** | 16.4s |
| q03 | Controller vs processor? | Definition | **96%** | 16.0s |
| q04 | When is a DPIA mandatory? | Obligation | **92%** | 18.5s |
| q05 | WP29 criteria triggering a DPIA? | Obligation | **96%** | 22.7s |
| q06 | CNIL DPIA processing list? | Recommendation | **90%** | 14.8s |
| q07 | Data controller obligations? | Obligation | **96%** | 19.1s |
| q08 | Data subject rights and limits? | Definition | **89%** | 16.7s |
| q09 | Keep CVs indefinitely? | Recommendation | **84%** | 14.5s |
| q10 | Legitimate interest for CCTV? | Recommendation | **89%** | 16.8s |
| q11 | Objection to HR processing? | Obligation | **96%** | 23.5s |
| q12 | 50-year data retention? | Tricky | **89%** | 15.7s |
| q13 | DPO mandatory everywhere? | Recommendation | **96%** | 22.2s |
| q14 | GDPR Article 99 on AI? | Tricky | **96%** | 14.6s |
| q15 | Privacy impact assessment? | Definition | **96%** | 13.3s |
| q16 | Who decides the means? | Definition | **100%** | 11.5s |
| q17 | Best marketing basis 2024? | Out of scope | **85%** | 17.3s |
| q18 | Bypass CNIL obligation? | Out of scope | **87%** | 15.1s |

## 📂 Enterprise Pipeline

Allows DPOs to feed the RAG with **their own internal documents** (policies, processing records, contracts, DPIAs…) while keeping the CNIL database as the authoritative reference.

```bash
# Import a folder of enterprise documents
python -m src.processing.ingest_enterprise --input docs/ --tag internal_policy --recursive

# List indexed enterprise documents
python -m src.processing.ingest_enterprise --list

# Purge enterprise documents (without affecting the CNIL database)
python -m src.processing.ingest_enterprise --purge
```

- **Supported formats**: PDF, DOCX, XLSX, HTML, TXT
- **Deduplication** via SHA256 hash (re-running = no duplicates)
- **Tags** per document for UI filtering (e.g., `internal_policy`, `register`, `dpia`)
- **CNIL always prevails** over enterprise docs in answers

## 📊 Observability

Production-ready monitoring with structured logging, user feedback, and alerting:

| Component | Description |
|---|---|
| **Structured logs** | JSON in `logs/app.jsonl` — every query, error, timing |
| **Query Logger** | `logs/queries.jsonl` — complete question history + metrics |
| **Feedback** | `logs/feedback.jsonl` — user 👍/👎 with context |
| **Alerts** | Configurable thresholds (error rate, response time, satisfaction, citations) |
| **SMTP** | Optional email notifications (config in `config.yaml`) |
| **Dashboard** | Dedicated Streamlit page with real-time metrics and export |

### SMTP Configuration (optional)

```yaml
# In configs/config.yaml
observability:
  alerting:
    smtp:
      enabled: true
      host: "smtp.gmail.com"
      port: 587
      username: "my-bot@gmail.com"
      password: "xxxx-xxxx-xxxx-xxxx"  # App password
      to_addrs:
        - "dpo@company.com"
```

## 🔧 Configuration

Configuration is centralized in `configs/config.yaml`. Key parameters:

```yaml
embeddings:
  model: "BAAI/bge-m3"              # 1024 dims, multilingual, FP16 GPU
  dims: 1024
  device: "cuda"

rag:
  enable_hybrid: true               # BM25 + semantic
  enable_query_expansion: true       # Multi-query LLM
  enable_reranker: true              # Cross-encoder Jina
  enable_summary_prefilter: true     # Summary pre-filter
  rerank_candidates: 40              # Candidates before reranking
  rerank_top_k: 10                   # Chunks after reranking
  temperature: 0.0                   # Strict factual

observability:
  logging:
    level: INFO
    structured_file: "app.jsonl"     # JSON structured logs
  alerting:
    enabled: true
    thresholds:
      error_rate_pct: 20.0
      avg_response_time_s: 60.0
```

## 📄 License

This project is an educational and research tool. The CNIL data used is publicly available.

## 🙏 Acknowledgements

- **CNIL** for making data protection resources publicly available
- **Ollama** for simplified local inference
- **Jina AI** for the open-source multilingual reranker
- **Mistral AI** for Mistral-Nemo 12B
