# RAG-DPO

🇫🇷 [Version française](README.md)

**Open source local AI assistant for GDPR and Data Protection Officers**

RAG-DPO is a Retrieval-Augmented Generation assistant designed to answer questions related to GDPR and data protection using a fully local and open source architecture.

The project was built with a strong focus on **data sovereignty**, **transparency** and **reliability of answers**.

Unlike many AI assistants relying on external APIs, RAG-DPO runs **entirely locally** and can be deployed in environments where data confidentiality is critical.

---

## Key features

- 🏠 **Local-first architecture** — no external API dependencies
- 📑 **Grounded answers** based on retrieved regulatory documents (CNIL, GDPR)
- 📊 **Evaluation metrics** to measure response quality and system stability
- 🏢 **Internal document ingestion** to adapt answers to a specific organization
- 🔍 **Fully auditable pipeline** — open source, reproducible
- 🎯 **90.4% global score** on 48-question benchmark (3 runs, 11 categories)

---

## Why this project

Many organizations are interested in AI assistants but face strong constraints regarding **data privacy**, **compliance** and **sovereignty**.

RAG-DPO explores how a fully local architecture can still deliver reliable answers for regulatory topics such as GDPR.

The project also serves as an **experiment** in evaluating the real performance of local RAG pipelines.

---

## Use cases

- 🧑‍💼 Internal GDPR assistant for employees
- 🛡️ Support tool for Data Protection Officers
- ⚖️ Knowledge assistant for legal or compliance teams
- 🔐 Exploration of sovereign AI architectures

---

## 💬 Example questions and answers

Here are real answers produced by RAG-DPO, with automatically cited sources:

<details>
<summary><strong>❓ What is personal data under the GDPR?</strong></summary>

> Personal data is any information relating to an identified or identifiable natural person. This definition is very broad and includes any information that can directly or indirectly identify a natural person **[Source 1]**.
>
> A person can be identified directly (e.g. name, first name) or indirectly (e.g. by an identifier, a number, biometric data, multiple elements specific to their physical, physiological, genetic, mental, economic, cultural or social identity, as well as voice or image). Identification of a natural person can be achieved from a single piece of data or from cross-referencing a set of data **[Source 1]**.

📎 **Source 1**: [RGPD, de quoi parle-t-on ?](https://cnil.fr/fr/rgpd-de-quoi-parle-t-on) — CNIL (technical sheet)

*Score: 93% — Category: Definition*

</details>

<details>
<summary><strong>❓ What is the deadline to notify a data breach to the CNIL?</strong></summary>

> A data breach must be notified to the CNIL whenever there is a risk to the privacy of data subjects, whether the incident is accidental or unlawful. This notification must be made without undue delay and, where feasible, **no later than 72 hours** after becoming aware of it, even when it involves a temporary unavailability **[Source 1]**.

📎 **Source 1**: [Multiplication des attaques par rançongiciel : comment limiter les risques](https://cnil.fr/fr/cybersecurite/multiplication-des-attaques-par-rancongiciel-comment-limiter-les-risques) — CNIL (doctrine)

*Score: 93% — Category: Obligation*

</details>

<details>
<summary><strong>❓ How to hack a website?</strong></summary>

> I refuse to answer this question. This request is **directly illegal** and **outside the GDPR/CNIL scope** covered by my sources.
>
> As a reminder, such actions carry **criminal sanctions**:
> - **Unauthorized access to a computer system** (Art. 323-1 French Criminal Code): up to 3 years imprisonment and €100,000 fine
> - **Fraudulent collection of personal data** (Art. 226-18 French Criminal Code): up to 5 years imprisonment and €300,000 fine
> - **Misuse of purpose** (Art. 226-21 French Criminal Code): up to 5 years imprisonment and €300,000 fine
>
> These offenses are prosecuted ex officio and are in no way within the scope of DPO guidance.

*Score: 92% — Category: Out of scope (appropriate refusal)*

</details>

---

## 📊 Benchmark results

**90.4% ± 0.4%** on 48 questions × 3 runs — scoring v7, 11 categories

| Metric | Value |
|---|---|
| Global score | **90.4% ± 0.4%** |
| Category-weighted score | **92.3%** |
| Questions ≥ 85% | **45/48** (94%) |
| Faithfulness (no hallucination) | **99.7%** |
| Correctly cited sources | **97.6%** |

<details>
<summary><strong>📋 By category (11 categories)</strong></summary>

| Category | Score |
|---|---|
| 🏢 **Organization** | **97.0%** |
| 📝 **Contractual** | **95.3%** |
| ⚙️ **Procedure** | **94.0%** |
| 📋 **Documentation** | **93.0%** |
| 🔧 **Methodology** | **92.7%** |
| 🪤 **Tricky** | **92.2%** |
| 🚫 **Out of scope** | **92.0%** |
| 📖 **Definition** | **91.2%** |
| 🌍 **International** | **91.0%** |
| ⚖️ **Obligation** | **88.8%** |
| 💡 **Recommendation** | **88.2%** |

</details>

<details>
<summary><strong>📋 Per-question scores (48 questions)</strong></summary>

| # | Question | Cat. | Score |
|---|---|---|---|
| q01 | What is personal data? | Definition | **93.0%** |
| q02 | Who is the data controller? | Definition | **94.0%** |
| q03 | Controller vs processor? | Definition | **89.0%** |
| q04 | When is a DPIA mandatory? | Obligation | **90.3%** |
| q05 | WP29 criteria triggering a DPIA? | Recommendation | **93.7%** |
| q06 | CNIL DPIA processing list? | Recommendation | **93.3%** |
| q07 | Data controller obligations? | Obligation | **93.0%** |
| q08 | Data subject rights and limits? | Definition | **89.3%** |
| q09 | Keep CVs indefinitely? | Recommendation | **90.0%** |
| q10 | Legitimate interest for CCTV? | Recommendation | **91.0%** |
| q11 | Objection to HR processing? | Recommendation | **85.7%** |
| q12 | 50-year data retention? | Tricky | **91.7%** |
| q13 | DPO mandatory everywhere? | Obligation | **94.3%** |
| q14 | GDPR Article 99 on AI? | Tricky | **97.0%** |
| q15 | When is a privacy impact assessment needed? | Obligation | **79.3%** |
| q16 | Who decides the processing means? | Definition | **91.0%** |
| q17 | Best marketing database 2024? | Out of scope | **91.3%** |
| q18 | Bypass CNIL obligation? | Out of scope | **94.0%** |
| q19 | Consent validity conditions? | Definition | **93.0%** |
| q20 | Contract as legal basis? | Definition | **92.0%** |
| q21 | Data breach notification deadline? | Obligation | **91.3%** |
| q22 | When to inform data subjects of a breach? | Obligation | **86.0%** |
| q23 | Is the processing register mandatory? | Obligation | **88.0%** |
| q24 | What is pseudonymization? | Definition | **90.0%** |
| q25 | Is anonymized data still personal data? | Definition | **94.7%** |
| q26 | Conditions for transfers outside the EU? | Obligation | **89.0%** |
| q27 | What sanctions can CNIL impose? | Definition | **85.0%** |
| q28 | Special categories of data? | Definition | **93.0%** |
| q29 | What is profiling? | Definition | **90.7%** |
| q30 | Fully automated decisions? | Obligation | **87.7%** |
| q31 | Payslip retention period? | Recommendation | **76.7%** |
| q32 | Consent required for all cookies? | Recommendation | **90.0%** |
| q33 | Commercial prospecting of a client? | Recommendation | **86.7%** |
| q34 | Cloud subprocessor outside EU? | Recommendation | **87.7%** |
| q35 | Does GDPR prohibit sensitive data processing? | Tricky | **92.7%** |
| q36 | Can you refuse all access requests? | Tricky | **87.7%** |
| q37 | Which antivirus does CNIL recommend? | Out of scope | **90.7%** |
| q38 | How to hack a website? | Out of scope | **92.0%** |
| q39 | Client data retention for prospecting? | Recommendation | **86.7%** |
| q40 | Transfer sensitive data to non-EU cloud? | Recommendation | **83.0%** |
| q41 | Tracking cookies without consent? | Recommendation | **93.0%** |
| q42 | Minimum age for child consent online? | Recommendation | **88.7%** |
| q43 | DPIA: obligation, method, actors? | Methodology | **92.7%** |
| q44 | Data breach: deadline and notification? | Procedure | **94.0%** |
| q45 | DPO: obligation and missions? | Organization | **97.0%** |
| q46 | Transfers outside EU: conditions and mechanisms? | International | **91.0%** |
| q47 | Processing register: content and obligation? | Documentation | **93.0%** |
| q48 | Subprocessor: obligations and contract? | Contractual | **95.3%** |

</details>

---

## Project goals

- Demonstrate the viability of local RAG architectures for compliance use cases
- Improve answer grounding and reduce hallucinations
- Provide a framework that can be adapted to ingest internal policies and procedures

Contributions and feedback are welcome.

---

<details>
<summary><h2>🏗️ Technical architecture</h2></summary>

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
│  │                 LangGraph Agent (9 nodes)                 │       │
│  │                                                          │       │
│  │  rewrite → classify → enrich → decompose ─┐              │       │
│  │                                            │              │       │
│  │      ┌─── composite (≥2 sub-questions) ────┤              │       │
│  │      │    1 global retrieval                │              │       │
│  │      │    1 structured generation (sections) │              │       │
│  │      │    global [Source N] renumbering      │              │       │
│  │      ▼                                      │              │       │
│  │   validate → respond                        │              │       │
│  │                                             │              │       │
│  │      ┌─── simple (1 question) ──────────────┘              │       │
│  │      ▼                                                    │       │
│  │   retrieve → generate → validate →                        │       │
│  │   check_completeness → respond                            │       │
│  │      ↑                    │                               │       │
│  │      └── (re-retrieve) ───┘                               │       │
│  │                                                          │       │
│  │  5 tools: DateCalculator, ArticleLookup,                  │       │
│  │  TopicSearch, QuestionDecomposer, CompletenessChecker     │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
│     Final answer + cited sources                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Intent Classification — Smart Routing

| Intent | Description | Specialized Prompt |
|---|---|---|
| `factual` | Definition/fact questions | Concise answer, direct references |
| `methodological` | "How to" questions | Numbered steps, practical approach |
| `organizational` | Organization/roles | Actors, responsibilities, org chart |
| `comparison` | Comparing concepts | Comparison table |
| `case_study` | Concrete situation | Case analysis, applicable rules |
| `exhaustive_list` | Complete enumeration | Structured, exhaustive list |
| `refusal` | Out-of-scope / circumvention | Firm refusal, sanction reminder |

### LangGraph Agent — 9 nodes, 5 local tools

1. **rewrite** — Multi-turn resolution
2. **classify** — Intent classification (7 types)
3. **enrich** — Date calculations + anti-confusion guards + GDPR articles
4. **decompose** — Multi-aspect detection → 1 global retrieval + 1 structured generation
5. **retrieve** — Hybrid retrieval + reranking (40 → top 10)
6. **generate** — LLM generation with intent-specialized prompt
7. **validate** — Citation verification (grounding), automatic retry
8. **check_completeness** — Sub-question coverage, re-retrieve if needed
9. **respond** — Final formatting + metadata

</details>

<details>
<summary><h2>📄 Data pipeline</h2></summary>

The pipeline transforms raw CNIL documents into vectorized chunks in ChromaDB.

### Steps

```
1. CNIL Scraping           → raw HTML pages (1,829 documents)
2. Classification           → document type (doctrine, guide, sanction, technical)
3. Relevance filtering      → non-GDPR documents excluded
4. LLM Summaries           → summary sheet per document (BM25 pre-filter)
5. Semantic chunking        → 16,919 chunks (50-word overlap, semantic split, heading propagation)
6. ChromaDB indexing        → BGE-M3 1024d embeddings
7. GDPR tagging             → 25 normalized categories per chunk
```

### Content-based chunking

The chunker detects and processes tables **by content**, not by file extension:

| Format | Detection | Documents |
|---|---|---|
| **HTML** | `<table>` in DOM | 39 documents |
| **PDF** | PyMuPDF `find_tables()` | 264 documents (54%) |
| **DOCX** | `w:tbl` python-docx | 1 document |
| **XLSX/ODS** | Native (spreadsheet) | 27 documents |

### 25 guided GDPR categories

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

</details>

<details>
<summary><h2>🛠️ Tech stack</h2></summary>

| Component | Technology | Details |
|---|---|---|
| **LLM** | Mistral-Nemo 12B | Via Ollama, 128K context, temperature 0.0 |
| **Embeddings** | BGE-M3 (BAAI) | sentence-transformers, 1024 dims, FP16 GPU |
| **VectorDB** | ChromaDB | PersistentClient, 16,919 chunks |
| **Reranker** | Jina Reranker v2 | 278M params, multilingual, GPU |
| **BM25** | rank_bm25 | Sparse index for hybrid search |
| **Agent** | LangGraph 1.0 | 9-node graph, query decomposition, 5 local tools |
| **Interface** | Streamlit multipage | Chat + Dashboard + Enterprise documents |
| **Observability** | JSONL + Alerter | Structured logs, feedback, SMTP alerts |
| **GPU** | RTX 4070 Ti 12GB | LLM + embeddings in VRAM, reranker on CPU |

### Project structure

```
RAG-DPO/
├── app.py                          # Streamlit entry point
├── update_cnil.py                  # Incremental CNIL update
├── rebuild_pipeline.py             # Data pipeline rebuild
├── tag_all_chunks.py               # Guided GDPR tagging (25 categories)
├── pages/                          # Streamlit pages (Chat, Dashboard, Documents, About)
├── src/
│   ├── rag/                        # RAG core (pipeline, retriever, reranker, generator…)
│   │   └── agent/                  # LangGraph Agent pipeline (9 nodes, 5 tools)
│   ├── processing/                 # Data pipeline (chunking, indexing, ingestion)
│   ├── scraping/                   # CNIL scraping
│   └── utils/                      # Providers, logging, alerting
├── eval/                           # Evaluation framework (48 questions, multi-run)
├── configs/                        # Centralized config (config.yaml)
├── data/                           # Data (raw, vectordb, metadata)
└── logs/                           # Structured JSONL logs
```

</details>

<details>
<summary><h2>🚀 Installation</h2></summary>

### Option A — Docker (recommended)

```bash
# Pull the image from GitHub Container Registry
docker pull ghcr.io/matjoss/rag-dpo:latest

# Launch the full stack (app + Ollama)
git clone https://github.com/MatJoss/RAG-DPO.git && cd RAG-DPO
docker compose up -d

# With GPU for embeddings + reranker:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

> 📖 Detailed Docker guide: [`docs/DOCKER_HOWTO.md`](docs/DOCKER_HOWTO.md)

### Option B — Local installation

#### Prerequisites

- **Python 3.11+**
- **NVIDIA GPU** with ≥8 GB VRAM (RTX 3070+ recommended)
- **Ollama** installed and running

#### 1. Clone and install

```bash
git clone https://github.com/MatJoss/RAG-DPO.git
cd RAG-DPO
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

#### 2. Install Ollama models

```bash
ollama pull mistral-nemo        # LLM 12B (7.1 GB)
ollama pull llava:7b             # Vision — image text extraction (4.7 GB)
```

> BGE-M3 embeddings are downloaded automatically on first run.

#### 3. Verify installation

```bash
python check_install.py
```

#### 4. Build the database (optional)

```bash
python rebuild_pipeline.py              # Full pipeline (scraping → indexing)
python rebuild_pipeline.py --from 5b    # Resume from chunking
python rebuild_pipeline.py --fresh      # Force full reprocessing
```

</details>

<details>
<summary><h2>💬 Usage</h2></summary>

### Streamlit interface (recommended)

```bash
streamlit run app.py
```

| Page | Description |
|---|---|
| 🏠 **Home** | System overview, statistics |
| 💬 **Chat** | RAG Q&A with cited sources, 👍/👎 feedback, Agent/Native toggle |
| 📊 **Dashboard** | Real-time metrics, alerts, JSON export |
| 📂 **Documents** | Enterprise document management (import, list, purge) |
| ℹ️ **About** | Credits, architecture, project links |

### Command line

```bash
python test_rag.py "When is a DPIA mandatory?"
```

### Evaluation

```bash
python eval/run_eval.py --agent --runs 3 --verbose     # Multi-run (recommended)
python eval/run_eval.py --verbose                       # Single run
python eval/run_eval.py --agent --runs 3 --top-k 5      # Different top-k
```

</details>

<details>
<summary><h2>🔄 CNIL Maintenance & Enterprise Pipeline</h2></summary>

### CNIL updates (~1x/month)

```bash
python update_cnil.py --status          # Current state
python update_cnil.py                   # Full update
python update_cnil.py --dry-run         # Preview without executing
python update_cnil.py --force-reindex   # Full ChromaDB reindexation
```

### Enterprise documents

Allows DPOs to feed the RAG with **internal documents** (policies, processing records, contracts, DPIAs…) while keeping the CNIL database as the authoritative reference.

```bash
python -m src.processing.ingest_enterprise --input docs/ --tag internal_policy --recursive
python -m src.processing.ingest_enterprise --list
python -m src.processing.ingest_enterprise --purge
```

- Formats: PDF, DOCX, XLSX, ODS, HTML, TXT
- SHA256 deduplication — CNIL always prevails in answers

</details>

<details>
<summary><h2>📊 Observability & Configuration</h2></summary>

| Component | Description |
|---|---|
| **Structured logs** | `logs/app.jsonl` — queries, errors, timings |
| **Query Logger** | `logs/queries.jsonl` — complete history |
| **Feedback** | `logs/feedback.jsonl` — 👍/👎 with context |
| **Alerts** | Configurable thresholds + SMTP notifications |
| **Dashboard** | Dedicated Streamlit page with export |

Centralized configuration in `configs/config.yaml` — embeddings, RAG, observability, SMTP.

</details>

<details>
<summary><h2>📈 System evolution</h2></summary>

| Version | Component | Impact |
|---|---|---|
| v1 | Semantic search + ChromaDB | Baseline 70% |
| v2 | Nomic embeddings + BM25 | +8% |
| v3 | LLM Query Expansion | +3% (recall) |
| v4 | Cross-Encoder Jina v2 | +3% (precision) |
| v5 | Smart rechunking | +1% |
| v6 | Intent Classification (7 intents) | Targeted prompts |
| v6b | LangGraph Agent (8 nodes) | +1.5% |
| v6d | BGE-M3 migration (1024d) | Native FR embeddings |
| **v7** | **Content-based tables + GDPR tags** | **+0.5%** |
| **v8** | **Raw scoring (no discretization)** | Real scores |
| **v9** | **Query Decomposition** | Citations preserved |
| **v10** | **Structured generation + DPO prompts** | **90.4%** |

```
Semantic only                   70%
+ BM25 hybrid                   78%  (+8%)
+ LLM Query Expansion           81%  (+3%)
+ Cross-Encoder Reranking       84%  (+3%)
+ Smart rechunking              85%  (+1%)
+ Intent Classification         86%
+ LangGraph Agent               89%  (+1.5%)
+ Content-based tables          89.7%
+ Single Gen + DPO prompts      90.4%
```

</details>

---

## 📄 License

This project is an educational and research tool. The CNIL data used is publicly available.

## 🙏 Acknowledgements

- **CNIL** — publicly available data protection resources
- **Ollama** — simplified local inference
- **Jina AI** — open-source multilingual reranker
- **Mistral AI** — Mistral-Nemo 12B
- **BAAI** — BGE-M3 (multilingual embeddings)
