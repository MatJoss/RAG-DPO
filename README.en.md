# 🔒 RAG-DPO — GDPR Assistant for DPOs

**RAG (Retrieval-Augmented Generation) system specialized in personal data protection**, designed to assist DPOs in their daily tasks. Fully local, no data sent to third parties.

> **Benchmark score: 89.7% ± 0.4%** on 42 questions × 3 runs (scoring v7, no discretization)
> One question below 80% — raw scoring, no inflated scores.

---

## 📋 Table of Contents

- [Benchmark Results](#-benchmark-results)
- [Before / After — Chunking Impact](#-before--after--chunking-impact)
- [Architecture](#️-architecture)
- [Data Pipeline](#-data-pipeline)
- [Tech Stack](#️-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [CNIL Maintenance](#-maintenance--cnil-database-updates)
- [Enterprise Pipeline](#-enterprise-pipeline)
- [Observability](#-observability)
- [Configuration](#-configuration)
- [System Evolution](#-system-evolution)
- [License](#-license)

---

## 📊 Benchmark Results

**Benchmark v8** — 42 questions, 3 runs, scoring v7 (55% Correctness + 25% Faithfulness + 20% Sources, raw LLM score 0-100)

### Global Score

| Metric | Value |
|---|---|
| **Global score** | **89.7% ± 0.4%** |
| Individual runs | 89.8%, 90.0%, 89.1% |
| Questions < 80% | **1** (q38, 80.0%) |
| Unstable questions (spread > 10%) | **3** (q10, q11, q38) |
| Average spread per question | 0.029 |
| Max spread | 0.32 |

### By Category

| Category | Questions | Score |
|---|---|---|
| **Definition** | 12 | **90.6%** |
| **Obligation** | 10 | **90.5%** |
| **Tricky** | 4 | **90.4%** |
| **Out of scope** | 4 | **88.2%** |
| **Recommendation** | 12 | **88.4%** |

### Per-Question Scores

| # | Question | Cat. | Score |
|---|---|---|---|
| q01 | What is personal data? | Definition | **94.0%** |
| q02 | Who is the data controller? | Definition | **84.0%** |
| q03 | Controller vs processor? | Definition | **82.7%** |
| q04 | When is a DPIA mandatory? | Obligation | **91.0%** |
| q05 | WP29 criteria triggering a DPIA? | Recommendation | **93.0%** |
| q06 | CNIL DPIA processing list? | Recommendation | **96.0%** |
| q07 | Data controller obligations? | Obligation | **91.0%** |
| q08 | Data subject rights and limits? | Definition | **91.0%** |
| q09 | Keep CVs indefinitely? | Recommendation | **91.7%** |
| q10 | Legitimate interest for CCTV? | Recommendation | **88.7%** |
| q11 | Objection to HR processing? | Recommendation | **87.0%** |
| q12 | 50-year data retention? | Tricky | **84.7%** |
| q13 | DPO mandatory everywhere? | Obligation | **94.3%** |
| q14 | GDPR Article 99 on AI? | Tricky | **95.0%** |
| q15 | When is a privacy impact assessment needed? | Obligation | **90.3%** |
| q16 | Who decides the processing means? | Definition | **93.0%** |
| q17 | Best marketing database 2024? | Out of scope | **89.7%** |
| q18 | Bypass CNIL obligation? | Out of scope | **93.3%** |
| q19 | Consent validity conditions? | Definition | **91.7%** |
| q20 | Contract as legal basis? | Definition | **95.0%** |
| q21 | Data breach notification deadline? | Obligation | **91.0%** |
| q22 | When to inform data subjects of a breach? | Obligation | **90.7%** |
| q23 | Is the processing register mandatory? | Obligation | **89.0%** |
| q24 | What is pseudonymization? | Definition | **93.0%** |
| q25 | Is anonymized data still personal data? | Definition | **95.0%** |
| q26 | Conditions for transfers outside the EU? | Obligation | **91.3%** |
| q27 | What sanctions can CNIL impose? | Definition | **84.7%** |
| q28 | Special categories of data? | Definition | **92.3%** |
| q29 | What is profiling? | Definition | **90.3%** |
| q30 | Fully automated decisions? | Obligation | **86.0%** |
| q31 | Payslip retention period? | Recommendation | **87.0%** |
| q32 | Consent required for all cookies? | Recommendation | **91.0%** |
| q33 | Commercial prospecting of a client? | Recommendation | **86.0%** |
| q34 | Cloud subprocessor outside EU? | Recommendation | **87.0%** |
| q35 | Does GDPR prohibit sensitive data processing? | Tricky | **91.7%** |
| q36 | Can you refuse all access requests? | Tricky | **90.3%** |
| q37 | Which antivirus does CNIL recommend? | Out of scope | **89.7%** |
| q38 | How to hack a website? | Out of scope | **80.0%** |
| q39 | Client data retention for prospecting? | Recommendation | **88.0%** |
| q40 | Transfer sensitive data to non-EU cloud? | Recommendation | **85.3%** |
| q41 | Tracking cookies without consent? | Recommendation | **81.7%** |
| q42 | Minimum age for child consent online? | Recommendation | **87.3%** |

---

## 🔄 Before / After — Chunking Impact

The transition from v6b to v7 (March 2026) introduced two major changes:
1. **Content-based table detection**: tables in HTML, PDF and DOCX are now extracted and converted to natural text via LLM, instead of being ignored or flattened
2. **Guided GDPR tags**: 25 normalized categories replace the ~7,500 anarchic free-form tags

### Measured Gains

> **Note**: v6b (Before) scores used discretized scoring that inflated results. v7 (After) scores use raw scoring v7 — deltas are therefore not directly comparable.

| Metric | Before (v6b, discretized scoring) | After (v7+v8, raw scoring) | Δ |
|---|---|---|---|
| **Global score** | 89.2% ± 1.1%¹ | **89.7% ± 0.4%** | **+0.5 pts** (real ~+2.9 pts) |
| Weighted score | 89.6%¹ | 89.6% | = |
| Questions < 80% | 4 | **1** | -3 |
| Unstable questions (spread > 10%) | 6 | **3** | -3 |
| Average spread per question | 0.049 | **0.029** | ÷1.7 |
| Max spread | 0.47 | **0.32** | ÷1.5 |
| Chunks in ChromaDB | ~14,400 | **16,919** | +2,519 |

¹ *Discretized scoring (v6): LLM scores were rounded to 6 tiers (0/25/50/75/90/100), inflating results by ~+2.4 pts.*

### By Category

| Category | Before¹ | After (raw) | Δ |
|---|---|---|---|
| Definition | 93.3% | **90.6%** | -2.7¹ |
| Obligation | 92.0% | **90.5%** | -1.5¹ |
| Recommendation | 83.1% | **88.4%** | **+5.3** |
| Tricky | 91.2% | **90.4%** | -0.8¹ |
| Out of scope | 88.3% | **88.2%** | -0.1¹ |

¹ *Apparent decreases are due to the switch to raw (non-discretized) scoring. The “recommendation” category genuinely improved thanks to table extraction.*

> **The “recommendation” category (+5.3 pts)** benefits most from the new chunking. CNIL tables containing retention periods, prospecting rules, and cookie recommendations were precisely in the `<table>` HTML elements ignored by the old chunker.

### Top 3 Improvements

| Question | Before¹ | After (raw) | Gain |
|---|---|---|---|
| q33 — Commercial prospecting | 48% | **86.0%** | **+38 pts** |
| q40 — Sensitive data to non-EU cloud | 58% | **85.3%** | **+27.3 pts** |
| q31 — Payslip retention period | 65.7% | **87.0%** | **+21.3 pts** |

These three questions relied on information contained in **HTML tables** from CNIL. The old chunker (`<h2>, <h3>, <p>, <ul>` only) completely ignored `<table>` elements, making this data invisible to the retriever.

### Key Takeaway

> **Chunking is the foundation of RAG.** No amount of pipeline tuning (top-k, reranking, prompts) can compensate for poorly extracted data at the source. When the correct information isn't in the chunks, no reformulation will surface it.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG-DPO Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User question                                                      │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ Intent Classif.  │  7 intents: factual, methodological,          │
│  │ (Phase 0)        │  organizational, comparison, case_study,      │
│  │                  │  exhaustive_list, refusal                     │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ├─── intent = "refusal" ──→ Direct refusal response       │
│           │                                                         │
│           ▼                                                         │
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
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                 LangGraph Agent                           │       │
│  │  ┌──────────────────────────────────────────────────────┐ │       │
│  │  │ rewrite → classify → enrich → retrieve →             │ │       │
│  │  │ generate → validate → check_completeness → respond   │ │       │
│  │  └──────────────────────────────────────────────────────┘ │       │
│  │  5 tools: DateCalculator, ArticleLookup,                  │       │
│  │  TopicSearch, QuestionDecomposer, CompletenessChecker     │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│  ┌─────────────────┐                                                │
│  │ Grounding       │  Verifies [Source X] citations                 │
│  │ Validation      │                                                │
│  └────────┬────────┘                                                │
│           ▼                                                         │
│     Final answer + cited sources                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Intent Classification — Smart Routing

Before any retrieval, the LLM classifies the question into **7 intents** via a structured prompt:

| Intent | Description | Specialized Prompt |
|---|---|---|
| `factual` | Definition/fact questions | Concise answer, direct references |
| `methodological` | "How to" questions | Numbered steps, practical approach |
| `organizational` | Organization/roles | Actors, responsibilities, org chart |
| `comparison` | Comparing concepts | Comparison table, commonalities/differences |
| `case_study` | Concrete situation | Case analysis, applicable rules |
| `exhaustive_list` | Complete enumeration | Structured, exhaustive list |
| `refusal` | Out-of-scope / circumvention | Firm refusal in 1-3 sentences, sanction reminder |

### LangGraph Agent — Intelligent Orchestration

The agent pipeline uses a LangGraph graph with **8 nodes** and **5 local tools** (no external calls):

1. **rewrite**: Multi-turn resolution — reformulates follow-up questions into standalone queries
2. **classify**: Intent classification (7 types)
3. **enrich**: Decomposition into sub-questions + date calculations + anti-confusion guards
4. **retrieve**: Hybrid retrieval + reranking (40 candidates → top 10)
5. **generate**: LLM generation with intent-specialized prompt
6. **validate**: Citation verification (grounding), automatic retry on failure
7. **check_completeness**: Ensures all sub-questions are covered, re-retrieves if not
8. **respond**: Final answer formatting + metadata

---

## 📄 Data Pipeline

The processing pipeline transforms raw CNIL documents into vectorized chunks in ChromaDB.

### Pipeline Steps

```
1. CNIL Scraping           → raw HTML pages (1,829 documents)
2. Classification           → document type (doctrine, guide, sanction, technical)
3. Relevance filtering      → non-GDPR documents excluded
4. LLM Summaries           → summary sheet per document (BM25 pre-filter)
5. Semantic chunking        → 16,919 chunks (50-word overlap, semantic split, heading propagation)
6. ChromaDB indexing        → BGE-M3 1024d embeddings
7. GDPR tagging             → 25 normalized categories per chunk
```

### Content-Based Chunking

The chunker detects and processes tables **by content**, not by file extension:

| Format | Detection | Method | Documents Affected |
|---|---|---|---|
| **HTML** | `<table>` elements in DOM | `_convert_html_table()` | 39 documents |
| **PDF** | PyMuPDF `find_tables()` | `_extract_pdf_tables()` | 264 documents (54%) |
| **DOCX** | `w:tbl` in python-docx DOM | `doc.tables` | 1 document |
| **XLSX/ODS** | Always (native spreadsheet) | `_chunk_spreadsheet()` | 27 documents |

All converge to **`_convert_table_rows()`** — a common pipeline:
1. Zone splitting (heading + data rows)
2. Split if zone > 500 words
3. Pipe-delimited text conversion
4. LLM rewrite (Mistral-Nemo) to natural text
5. Mechanical fallback if LLM fails

### Guided GDPR Tags

Each chunk is tagged among **25 normalized GDPR categories**:

```
data subject rights, consent, data security,
data retention period, subprocessing, legal basis,
sensitive data, transfers outside EU, cookies,
data breach, transparency, DPO,
video surveillance, purpose of processing, processing records,
DPIA, anonymization, minimization, data controller,
commercial prospecting, GDPR compliance, profiling,
CNIL sanctions, health data, informing data subjects
```

The LLM prompt guides the model toward this controlled vocabulary, eliminating anarchic tags (7,500 → ~25).

---

## 🛠️ Tech Stack

| Component | Technology | Details |
|---|---|---|
| **LLM** | Mistral-Nemo 12B | Via Ollama, 128K context, temperature 0.0 |
| **Embeddings** | BGE-M3 (BAAI) | sentence-transformers, 1024 dims, FP16 GPU |
| **VectorDB** | ChromaDB | PersistentClient, 16,919 chunks |
| **Reranker** | Jina Reranker v2 | 278M params, multilingual, CPU |
| **BM25** | rank_bm25 | Sparse index for hybrid search |
| **Agent** | LangGraph 1.0 | 8-node graph, 5 local tools, state management |
| **Intent** | LLM Classification | 7 intents, structured prompt, JSON output |
| **Interface** | Streamlit multipage | Chat + Dashboard + Enterprise documents |
| **Observability** | JSONL + Alerter | Structured logs, feedback, SMTP alerts |
| **GPU** | RTX 4070 Ti 12GB | LLM + embeddings in VRAM, reranker on CPU |

### Project Structure

```
RAG-DPO/
├── app.py                      # Streamlit multipage entry point
├── update_cnil.py              # Incremental CNIL database update (~1x/month)
├── rebuild_pipeline.py         # Data pipeline rebuild
├── tag_all_chunks.py           # Guided GDPR tagging (25 categories)
├── pages/
│   ├── 1_💬_Chat.py            # Interactive RAG chat + feedback + Agent/Native toggle
│   ├── 2_📊_Dashboard.py       # Observability dashboard (metrics, alerts)
│   └── 3_📂_Documents.py       # Enterprise document management
├── test_rag.py                 # CLI RAG testing
├── check_install.py            # Installation verification
├── configs/
│   ├── config.yaml             # Centralized config (RAG + observability + SMTP)
│   └── enterprise_tags.json    # Enterprise tag registry (auto-generated)
├── src/
│   ├── rag/                    # 🧠 RAG core
│   │   ├── pipeline.py         # Orchestration (intent-aware, dual-gen)
│   │   ├── intent_classifier.py # 7-intent classification (Phase 0)
│   │   ├── retriever.py        # Hybrid retrieval (BM25 + semantic + RRF)
│   │   ├── query_expander.py   # Multi-query expansion via LLM
│   │   ├── bm25_index.py       # BM25 index (summaries + chunks)
│   │   ├── reranker.py         # Cross-encoder Jina reranking + topic boost
│   │   ├── context_builder.py  # Context building + 7 specialized prompts
│   │   ├── generator.py        # LLM generation (Ollama)
│   │   ├── validators.py       # Grounding + relevance validation
│   │   └── agent/              # 🤖 LangGraph Agent Pipeline
│   │       ├── graph.py        # LangGraph graph (8 nodes)
│   │       ├── nodes.py        # Node functions
│   │       ├── tools.py        # 5 local tools
│   │       └── state.py        # RAGState TypedDict
│   ├── processing/             # 📄 Data processing pipeline
│   │   ├── process_and_chunk.py        # Semantic chunking + table detection
│   │   ├── create_chromadb_index.py    # BGE-M3 vector indexing
│   │   ├── generate_document_summaries.py  # LLM summary sheets
│   │   ├── hybrid_filter.py            # Relevance filtering
│   │   ├── classify_documents.py       # Document classification
│   │   └── ingest_enterprise.py        # Enterprise doc ingestion
│   ├── scraping/               # 🕷️ CNIL scraping
│   │   └── cnil_scraper_final.py
│   └── utils/
│       ├── llm_provider.py     # Ollama interface
│       ├── embedding_provider.py # BGE-M3 provider (FP16, GPU, lazy load)
│       ├── rgpd_topics.py      # 25 GDPR categories + TopicMatcher + prompts
│       ├── query_logger.py     # JSONL query & feedback logger
│       ├── structured_logger.py # JSON structured logging
│       ├── alerter.py          # Threshold alerts + SMTP
│       └── acronyms.py         # GDPR acronym expansion
├── eval/                       # 📊 Evaluation framework
│   ├── qa_dataset.json         # 42-question benchmark (5 categories)
│   ├── run_eval.py             # 4-axis evaluation + multi-run (--runs N)
│   └── results_*.json          # Historical results
├── logs/                       # 📝 Structured logs (not versioned)
├── data/                       # 📁 Data (not versioned)
│   ├── raw/                    # Raw CNIL documents
│   ├── vectordb/chromadb/      # ChromaDB vector database
│   └── metadata/               # Document metadata
└── tasks/                      # 📝 Work notes
    ├── todo.md
    └── lessons.md              # Lessons learned (patterns, mistakes, fixes)
```

---

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
ollama pull llava:7b             # Vision — image text extraction (4.7 GB)
```

> **Note**: BGE-M3 embeddings (sentence-transformers) are downloaded automatically on first run.

### 3. Verify installation

```bash
python check_install.py
```

### 4. Build the database (optional)

If starting from scratch with your own CNIL data:

```bash
python rebuild_pipeline.py              # Full pipeline (scraping → indexing)
python rebuild_pipeline.py --from 5b    # Resume from chunking
python rebuild_pipeline.py --fresh      # Force reprocessing of all documents
```

---

## 💬 Usage

### Streamlit interface (recommended)

```bash
streamlit run app.py
```

Opens the multipage application in the browser:

| Page | Description |
|---|---|
| 🏠 **Home** | System overview, statistics |
| 💬 **Chat** | RAG Q&A interface with cited sources, 👍/👎 feedback, Agent/Native toggle |
| 📊 **Dashboard** | Real-time metrics, alerts, feedback, JSON export |
| 📂 **Documents** | Enterprise document management (import, list, purge) |

Chat features:
- **Agent / Native toggle** in sidebar (agent recommended)
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
# Multi-run recommended (3 runs for statistical averaging)
python eval/run_eval.py --agent --runs 3 --verbose

# Single run
python eval/run_eval.py --verbose
```

Runs the 42-question benchmark in 2 phases:
1. **Phase 1**: RAG generation + keyword scoring + semantic similarity (BGE-M3 cosine)
2. **Phase 2**: LLM-as-Judge — free score 0-100 in JSON, used directly (no discretization)

Final score = **55% Correctness** (60% LLM-Judge + 40% Semantic) + **25% Faithfulness** + **20% Sources**

---

## 🔄 Maintenance — CNIL Database Updates

The CNIL database evolves regularly (new sanctions, guides, recommendations). A dedicated script handles incremental updates (~1x/month):

```bash
python update_cnil.py --status          # Current database state
python update_cnil.py                   # Full update
python update_cnil.py --dry-run         # Preview without executing
python update_cnil.py --scrape-only     # Check for CNIL-side changes
python update_cnil.py --force-reindex   # Full ChromaDB reindexation
```

Scraping uses conditional requests (`If-Modified-Since` → `304 Not Modified`) to only re-download modified pages. Subsequent steps (classification, chunking, summaries, tagging) automatically detect already-processed documents.

---

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

- **Supported formats**: PDF, DOCX, XLSX, ODS, HTML, TXT
- **Table detection**: automatic for all formats (content-based)
- **Deduplication** via SHA256 hash (re-running = no duplicates)
- **Tags** per document for UI filtering
- **CNIL always prevails** over enterprise docs in answers

---

## 📊 Observability

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

---

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

---

## 📈 System Evolution

| Version | Component | Impact |
|---|---|---|
| v1 | Semantic search + ChromaDB | Baseline 70% |
| v2 | Nomic embeddings + BM25 | +8% |
| v3 | LLM Query Expansion | +3% (recall) |
| v4 | Cross-Encoder Jina v2 | +3% (precision) |
| v5 | Smart rechunking (overlap, heading, semantic split) | +1% |
| v6 | Intent Classification (7 intents, specialized prompts) | Targeted prompts |
| v6b | LangGraph Agent (8 nodes, 5 tools) | +1.5% + robustness |
| v6c | Eval v5 → v6 (free score JSON, 42 questions) | Reliable thermometer |
| v6d | BGE-M3 migration (replaces nomic, 1024d) | Native FR embeddings |
| **v7** | **Content-based table detection + guided GDPR tags** | **+0.5%** (89.2%¹ → 89.7%) |
| **v8** | **Removed LLM-Judge discretization (raw scoring v7)** | Real scores (not inflated) |

### Scoring (eval v3 → v7)

Scores are **not directly comparable** across evaluation generations:

| | Eval v3 (v1–v5) | Eval v4 (v6–v6b) | Eval v6 (v6c–v7) | Eval v7 (v8+) |
|---|---|---|---|---|
| Final score | 70% LLM-Judge + 30% Keywords | 55% Correctness + 25% Faithfulness + 20% Sources | Same as v4 | Same as v4 |
| LLM-Judge | Free score 0-100 (text) | Free score 0-100 (text) | Free score 0-100 (JSON, discretized) | **Raw score 0-100 (JSON, no discretization)** |
| Dataset | 18 questions | 18 questions | **42 questions** (5 categories) | **42 questions** (5 categories) |
| Multi-run | No | Yes (3 runs) | Yes (3 runs) | Yes (3 runs) |
| Calibration | — | — | Discretized → inflated scores ~+2.4 pts | **Raw scores, not inflated** |

### Gains by Component

```
Semantic only                   70% ─────────────────────┐
+ BM25 hybrid                   78%  (+8%)               │ Augmented
+ LLM Query Expansion           81%  (+3%)               │ retrieval
+ Cross-Encoder Reranking       84%  (+3%)               │
+ Smart rechunking              85%  (+1%)  ─────────────┘
+ Intent Classification         86%  (targeted prompts)    → Precision
+ LangGraph Agent               89%  (+1.5%)               → Tools + control
+ Content-based tables          89.7% (+0.5%)¹            → Complete data
¹ Real chunking gain is ~+2.9% but masked by the removal of
  discretization that inflated previous scores by ~2.4 pts.
```

---

## 📄 License

This project is an educational and research tool. The CNIL data used is publicly available.

## 🙏 Acknowledgements

- **CNIL** for making data protection resources publicly available
- **Ollama** for simplified local inference
- **Jina AI** for the open-source multilingual reranker
- **Mistral AI** for Mistral-Nemo 12B
- **BAAI** for BGE-M3 (multilingual embeddings)
