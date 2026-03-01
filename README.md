# 🔒 RAG-DPO — Assistant RGPD pour DPO

> 🇬🇧 [English version](README.en.md)

> Système RAG (Retrieval-Augmented Generation) 100% local pour assister les Délégués à la Protection des Données, basé sur les sources officielles CNIL.

<!-- Captures à ajouter -->

## 🎯 Objectif

Un assistant expert RGPD qui :
- **Répond uniquement à partir de sources CNIL vérifiées** (zéro hallucination)
- **Cite ses sources** avec renvoi vers les documents originaux
- **Tourne entièrement en local** — aucune donnée ne sort de la machine
- **Gère les nuances** — détecte automatiquement les contradictions entre sources

## 📊 Performances

**Score global : 93%** sur un benchmark de 18 questions RGPD/CNIL couvrant 5 catégories.

| Métrique | Score |
|---|---|
| 📈 Score Global | **93%** |
| ✅ Correctness (LLM Judge + Keywords) | **84%** |
| 🛡️ Faithfulness (fidélité aux sources) | **100%** |
| 📏 Conciseness | **98%** |
| 📚 Source Quality | **97%** |
| ⏱️ Temps moyen/question | **17.3s** |

### Par catégorie

| Catégorie | Score | Questions |
|---|---|---|
| 📖 Définitions | **97%** | 5 |
| ⚖️ Obligations | **95%** | 4 |
| 🪤 Pièges | **92%** | 2 |
| 💡 Recommandations | **91%** | 5 |
| 🚫 Hors périmètre | **86%** | 2 |

### Évolution des performances

Le système a été construit de manière itérative. Chaque composant du pipeline a été évalué sur le même benchmark de 18 questions :

| Version | Configuration | Global | Correctness | Temps/q |
|---|---|---|---|---|
| v1 — Baseline | Semantic seul, pas de reranker | 86% | 65% | 6.3s |
| v2 — Query Expansion | + LLM multi-query (×3 reformulations) | 89% | 73% | 13.2s |
| v3 — Cross-Encoder | + BGE reranker v2 m3 (568M) | 92% | 78% | 8.2s |
| v4 — Jina Reranker | BGE → Jina v2 multilingual (278M, 7× plus rapide) | 92% | 83% | 9.5s |
| v5 — Rechunking | Overlap 50w, heading propagé, split sémantique | **93%** | **84%** | 31.9s |
| v6 — BM25 Boost | BM25 weight ×1.5, fix éval keywords | 92% | 81% | 14.0s |
| **v7 — Dual Generation** | **Self-consistency via context order** | **93%** | **84%** | **17.3s** |

#### Gains par composant

```
Semantic seul                   86% ─────────────────────┐
+ Query Expansion LLM           89%  (+3%)               │ Retrieval
+ Cross-Encoder Reranking       92%  (+3%)               │ augmenté
+ Rechunking intelligent        93%  (+1%)  ─────────────┘
+ Dual Generation               93%  (stabilité +         
                                      correctness 84%)    → Robustesse
```

**Contribution clé de chaque composant :**

| Composant | Impact principal | Gain |
|---|---|---|
| **Query Expansion** | Meilleur recall — reformulations capturent les synonymes RGPD | +3% global, +8% correctness |
| **Cross-Encoder** | Meilleure précision — reranking fin vs cosine grossier | +3% global |
| **Rechunking** | Chunks auto-suffisants — overlap + heading + split sémantique | +1% global, +6% correctness |
| **Dual Generation** | Robustesse — détecte les contradictions entre sources | +2% correctness, q10 63%→89% |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAG-DPO Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Question utilisateur                                               │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ Query Expansion  │  LLM génère 3 reformulations                  │
│  │ (multi-query)    │  + expansion acronymes RGPD                   │
│  └────────┬────────┘                                                │
│           ▼                                                         │
│  ┌─────────────────┐     ┌──────────────┐                           │
│  │ Summary         │───> │ BM25 Index   │  Pré-filtre : top-40      │
│  │ Pre-Filter      │     │ (summaries)  │  documents pertinents     │
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
│  │ Jina Reranker   │  Cross-encoder multilingual                    │
│  │ v2 (278M, CPU)  │  40 candidats → top 10                         │
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
│  │ Grounding       │  Vérifie les citations [Source X]              │
│  │ Validation      │                                                │
│  └────────┬────────┘                                                │
│           ▼                                                         │
│     Réponse finale + sources citées                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Dual Generation — Self-Consistency via Context Order

Le mécanisme clé d'innovation : le système génère **deux réponses** avec les mêmes documents mais dans un **ordre différent**, puis compare les positions (stance) :

- **Concordant** (même conclusion) → la réponse est fiable, on la retourne
- **Contradiction** (conclusions opposées) → les sources couvrent des **cas différents** → synthèse nuancée via un 3ème appel LLM

Ce mécanisme a résolu le problème le plus tenace du système (q10 : *"Peut-on utiliser l'intérêt légitime pour la vidéosurveillance ?"*), où un chunk spécifique aux communes ("l'intérêt légitime n'est pas mobilisable") contredisait la règle générale ("oui, avec mise en balance"). Score : 63% → 89%.

## 🛠️ Stack Technique

| Composant | Technologie | Détails |
|---|---|---|
| **LLM** | Mistral-Nemo 12B | Via Ollama, 128K context, temperature 0.0 |
| **Embeddings** | BGE-M3 (BAAI) | sentence-transformers, 1024 dims, FP16 GPU |
| **VectorDB** | ChromaDB | PersistentClient, 14 388 chunks |
| **Reranker** | Jina Reranker v2 | 278M params, multilingual, CPU |
| **BM25** | rank_bm25 | Index sparse pour recherche hybride |
| **Interface** | Streamlit multipage | Chat + Dashboard observabilité |
| **Observabilité** | JSONL + Alerter | Logs structurés, feedback, alertes SMTP |
| **GPU** | RTX 4070 Ti 12GB | LLM + embeddings en VRAM, reranker sur CPU |

### Structure du projet

```
RAG-DPO/
├── app.py                      # Point d'entrée Streamlit multipage
├── pages/
│   ├── 1_💬_Chat.py            # Chat RAG interactif + feedback
│   └── 2_📊_Dashboard.py       # Dashboard observabilité (métriques, alertes)
├── test_rag.py                 # Test RAG en ligne de commande
├── check_install.py            # Vérification de l'installation
├── rebuild_pipeline.py         # Reconstruction du pipeline de données
├── requirements.txt            # Dépendances Python
├── configs/
│   ├── config.yaml             # Configuration centralisée (RAG + observabilité + SMTP)
│   └── enterprise_tags.json    # Registre des tags entreprise (auto-généré)
├── src/
│   ├── rag/                    # 🧠 Cœur du système RAG
│   │   ├── pipeline.py         # Orchestration (dual-gen, stance detection)
│   │   ├── retriever.py        # Hybrid retrieval (BM25 + semantic + RRF)
│   │   ├── query_expander.py   # Multi-query expansion via LLM
│   │   ├── bm25_index.py       # Index BM25 (summaries + chunks)
│   │   ├── reranker.py         # Cross-encoder Jina reranking
│   │   ├── context_builder.py  # Construction contexte + reverse packing
│   │   ├── generator.py        # Génération LLM (Ollama)
│   │   └── validators.py       # Grounding + relevance validation
│   ├── processing/             # 📄 Pipeline de traitement des données
│   │   ├── ingest_enterprise.py        # Ingestion docs entreprise (PDF, DOCX, XLSX…)
│   │   ├── process_and_chunk.py        # Chunking sémantique
│   │   ├── create_chromadb_index.py    # Indexation vectorielle
│   │   ├── generate_document_summaries.py  # Fiches synthétiques LLM
│   │   ├── hybrid_filter.py            # Filtrage pertinence
│   │   ├── classify_documents.py       # Classification par nature
│   │   └── ...
│   ├── scraping/               # 🕷️ Scraping CNIL
│   │   └── cnil_scraper_final.py
│   └── utils/
│       ├── llm_provider.py     # Interface Ollama
│       ├── embedding_provider.py # Provider BGE-M3 (FP16, GPU, lazy load)
│       ├── query_logger.py     # Logger JSONL queries + feedback
│       ├── structured_logger.py # Logging JSON structuré
│       ├── alerter.py          # Alertes seuils + SMTP
│       └── acronyms.py         # Expansion acronymes RGPD
├── eval/                       # 📊 Framework d'évaluation
│   ├── qa_dataset.json         # 18 questions benchmark (5 catégories)
│   ├── run_eval.py             # Évaluation 2 phases (keywords + LLM judge)
│   └── results_*.json          # Résultats historiques
├── logs/                       # 📝 Logs structurés (non versionné)
│   ├── app.jsonl               # Logs applicatifs JSON
│   ├── queries.jsonl           # Historique requêtes
│   ├── feedback.jsonl          # Feedback utilisateurs 👍/👎
│   └── alerts.jsonl            # Historique alertes
├── data/                       # 📁 Données (non versionné)
│   ├── raw/                    # Documents bruts CNIL
│   ├── vectordb/chromadb/      # Base vectorielle ChromaDB
│   └── metadata/               # Métadonnées documents
└── tasks/                      # 📝 Notes de travail
    ├── todo.md
    └── lessons.md              # Leçons apprises (patterns, erreurs, fixes)
```

## 🚀 Installation

### Prérequis

- **Python 3.11+**
- **NVIDIA GPU** avec ≥8 Go VRAM (RTX 3070+ recommandé)
- **Ollama** installé et lancé

### 1. Cloner et installer

```bash
git clone https://github.com/<user>/RAG-DPO.git
cd RAG-DPO
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Installer les modèles Ollama

```bash
ollama pull mistral-nemo        # LLM 12B (7.1 GB)
```

> **Note** : Les embeddings BGE-M3 (sentence-transformers) sont téléchargés automatiquement au premier lancement.

### 3. Vérifier l'installation

```bash
python check_install.py
```

### 4. Construire la base de données (optionnel)

Si vous partez de zéro avec vos propres données CNIL :

```bash
python rebuild_pipeline.py          # Pipeline complet (scraping → indexation)
python rebuild_pipeline.py --from 5b  # Reprendre depuis le chunking
```

## 💬 Utilisation

### Interface Streamlit (recommandé)

```bash
streamlit run app.py
```

Ouvre l'application multipage dans le navigateur :

| Page | Description |
|---|---|
| 🏠 **Accueil** | Vue d'ensemble du système, statistiques |
| 💬 **Chat** | Interface Q&R RAG avec sources citées et feedback 👍/👎 |
| 📊 **Dashboard** | Métriques temps réel, alertes, feedback, export JSON |

Fonctionnalités du chat :
- Filtrage par nature de document (Doctrine, Guide, Sanction, Technique)
- Filtrage par tags entreprise (si docs internes importés)
- Sources citées avec distinction [CNIL] / [Interne]
- Feedback utilisateur 👍/👎 enregistré dans les logs

### Ligne de commande

```bash
python test_rag.py "Quand une AIPD est-elle obligatoire ?"
```

### Évaluation

```bash
python eval/run_eval.py --verbose
```

Exécute le benchmark 18 questions en 2 phases :
1. **Phase 1** : Génération RAG + scoring par mots-clés
2. **Phase 2** : LLM-as-Judge (le LLM évalue la qualité sémantique)

Score final = 70% LLM Judge + 30% Keywords

## 📈 Benchmark détaillé (v7 — Dual Generation)

| # | Question | Cat. | Score | Temps |
|---|---|---|---|---|
| q01 | Qu'est-ce qu'une donnée personnelle ? | Définition | **98%** | 23.1s |
| q02 | Qui est responsable de traitement ? | Définition | **100%** | 16.4s |
| q03 | RT vs sous-traitant ? | Définition | **96%** | 16.0s |
| q04 | Quand une AIPD est obligatoire ? | Obligation | **92%** | 18.5s |
| q05 | Critères G29 déclenchant une AIPD ? | Obligation | **96%** | 22.7s |
| q06 | Liste traitements AIPD CNIL ? | Recommandation | **90%** | 14.8s |
| q07 | Obligations responsable traitement ? | Obligation | **96%** | 19.1s |
| q08 | Droits des personnes et limites ? | Définition | **89%** | 16.7s |
| q09 | Conserver des CV indéfiniment ? | Recommandation | **84%** | 14.5s |
| q10 | Intérêt légitime vidéosurveillance ? | Recommandation | **89%** | 16.8s |
| q11 | Opposition traitement RH ? | Obligation | **96%** | 23.5s |
| q12 | Conservation données 50 ans ? | Piège | **89%** | 15.7s |
| q13 | DPO obligatoire partout ? | Recommandation | **96%** | 22.2s |
| q14 | Article 99 RGPD sur l'IA ? | Piège | **96%** | 14.6s |
| q15 | Étude d'impact vie privée ? | Définition | **96%** | 13.3s |
| q16 | Qui décide des moyens ? | Définition | **100%** | 11.5s |
| q17 | Meilleure base marketing 2024 ? | Hors périmètre | **85%** | 17.3s |
| q18 | Contourner obligation CNIL ? | Hors périmètre | **87%** | 15.1s |

## 📂 Pipeline Entreprise

Permet aux DPO d'alimenter le RAG avec **leurs propres documents internes** (politiques, registres, contrats, PIA…) tout en conservant la base CNIL comme référentiel autoritaire.

```bash
# Importer un dossier de documents entreprise
python -m src.processing.ingest_enterprise --input docs/ --tag politique_interne --recursive

# Lister les documents entreprise indexés
python -m src.processing.ingest_enterprise --list

# Purger les documents entreprise (sans toucher à la base CNIL)
python -m src.processing.ingest_enterprise --purge
```

- **Formats supportés** : PDF, DOCX, XLSX, HTML, TXT
- **Déduplication** par hash SHA256 (relancer = pas de doublons)
- **Tags** par document pour filtrage dans l'UI (ex: `politique_interne`, `registre`, `pia`)
- **CNIL prévaut toujours** sur les docs entreprise dans les réponses

## 📊 Observabilité

Système de monitoring production-ready avec logging structuré, feedback utilisateur et alerting :

| Composant | Description |
|---|---|
| **Logs structurés** | JSON dans `logs/app.jsonl` — chaque requête, erreur, timing |
| **Query Logger** | `logs/queries.jsonl` — historique complet des questions + métriques |
| **Feedback** | `logs/feedback.jsonl` — 👍/👎 utilisateur avec contexte |
| **Alertes** | Seuils configurables (taux erreur, temps, satisfaction, citations) |
| **SMTP** | Notifications email optionnelles (config dans `config.yaml`) |
| **Dashboard** | Page Streamlit dédiée avec métriques temps réel et export |

### Configuration SMTP (optionnelle)

```yaml
# Dans configs/config.yaml
observability:
  alerting:
    smtp:
      enabled: true
      host: "smtp.gmail.com"
      port: 587
      username: "mon-bot@gmail.com"
      password: "xxxx-xxxx-xxxx-xxxx"  # App password
      to_addrs:
        - "dpo@entreprise.com"
```

## 🔧 Configuration

La configuration est centralisée dans `configs/config.yaml`. Paramètres clés :

```yaml
embeddings:
  model: "BAAI/bge-m3"              # 1024 dims, multilingue, FP16 GPU
  dims: 1024
  device: "cuda"

rag:
  enable_hybrid: true               # BM25 + semantic
  enable_query_expansion: true       # Multi-query LLM
  enable_reranker: true              # Cross-encoder Jina
  enable_summary_prefilter: true     # Pré-filtre par résumés
  rerank_candidates: 40              # Candidats avant reranking
  rerank_top_k: 10                   # Chunks après reranking
  temperature: 0.0                   # Factuel strict

observability:
  logging:
    level: INFO
    structured_file: "app.jsonl"     # Logs JSON structurés
  alerting:
    enabled: true
    thresholds:
      error_rate_pct: 20.0
      avg_response_time_s: 60.0
```

## 📄 Licence

Ce projet est un outil éducatif et de recherche. Les données CNIL utilisées sont publiques.

## 🙏 Remerciements

- **CNIL** pour la mise à disposition des ressources sur la protection des données
- **Ollama** pour l'inférence locale simplifiée
- **Jina AI** pour le reranker multilingue open-source
- **Mistral AI** pour Mistral-Nemo 12B
