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

Résultats sur un benchmark de **18 questions RGPD/CNIL** couvrant 6 catégories, moyennés sur **3 runs** pour fiabilité statistique (temp=0 ne garantit pas le déterminisme GPU).

### Pipeline Agent LangGraph (recommandé)

| Métrique | Score |
|---|---|
| 📈 **Score Global** | **88.9% ± 1.8%** |
| ✅ Correctness (50% LLM-Judge + 35% Semantic + 15% Keywords) | **81.4%** |
| 🛡️ Faithfulness (fidélité aux sources) | **96.3%** |
| 📚 Source Quality | **100%** |
| 🔄 Stabilité inter-runs (σ) | **1.8%** |
| ⏱️ Temps moyen/question | **24.8s** |

### Pipeline Natif (intent-aware)

| Métrique | Score |
|---|---|
| 📈 **Score Global** | **89.1% ± 0.4%** |
| ✅ Correctness | **80.4%** |
| 🛡️ Faithfulness | **100%** |
| 📚 Source Quality | **99.1%** |
| 🔄 Stabilité inter-runs (σ) | **0.4%** |
| ⏱️ Temps moyen/question | **21.0s** |

### Comparaison Agent vs Natif (moyenne 3 runs)

| Question | Cat. | Agent | Natif | Δ | Gagnant |
|---|---|---|---|---|---|
| q01 Donnée personnelle | Définition | 93.7% | 94.0% | -0.3 | ≈ |
| q02 Responsable de traitement | Définition | 95.3% | 92.3% | +3.0 | 🤖 Agent |
| q03 RT vs sous-traitant | Définition | 87.0% | 87.7% | -0.7 | ≈ |
| q04 AIPD obligatoire | Obligation | 87.7% | 87.3% | +0.4 | ≈ |
| q05 Critères G29 AIPD | Recommandation | 93.0% | 93.0% | 0.0 | ≈ |
| q06 Liste traitements AIPD | Recommandation | 92.0% | 92.0% | 0.0 | ≈ |
| q07 Obligations RT | Obligation | 93.0% | 94.0% | -1.0 | ≈ |
| q08 Droits personnes | Définition | 91.0% | 90.0% | +1.0 | ≈ |
| q09 Conservation CV | Recommandation | 91.3% | 85.3% | +6.0 | 🤖 Agent |
| q10 Vidéosurveillance | Recommandation | 87.0% | 89.0% | -2.0 | ≈ |
| q11 Opposition RH | Recommandation | 92.0% | 91.0% | +1.0 | ≈ |
| q12 Conservation 50 ans | Piège | 80.0% | 78.7% | +1.3 | ≈ |
| q13 DPO obligatoire | Obligation | 92.0% | 92.0% | 0.0 | ≈ |
| q14 Article 99 IA | Piège | 85.7% | 91.7% | -6.0 | Natif |
| q15 Étude d'impact | Obligation | 88.0% | 88.0% | 0.0 | ≈ |
| q16 Qui décide moyens | Définition | 94.3% | 92.7% | +1.6 | 🤖 Agent |
| q17 Marketing hors périmètre | Hors périmètre | **96.0%** | **96.0%** | 0.0 | ≈ |
| q18 Contourner CNIL | Hors périmètre | **92.3%** | 69.0% | **+23.3** | 🤖 Agent |

**Verdict** : L'agent est meilleur sur 4 questions clés (q02, q09, q16, q18) et n'est inférieur que sur q14 (article piège). L'écart le plus marquant est sur q18 ("contourner une obligation CNIL") où le pipeline agent, grâce à l'intent "refus", produit une réponse ferme et bien cotée (+23 pts).

### Analyse de la variance (3 runs, temp=0)

La température à 0 ne garantit **pas** des réponses déterministes — l'arithmétique GPU en virgule flottante introduit des variations d'arrondi qui peuvent inverser le choix du prochain token.

| Indicateur | Agent | Natif |
|---|---|---|
| σ global | 0.3% | 0.4% |
| Spread moyen/question | 3.4% | 3.7% |
| Spread max | 12% (q10) | 13% (q03) |
| Questions à σ=0 (parfaitement stables) | 5/18 | 6/18 |

**Conclusion** : les deux pipelines sont stables au global (σ < 0.5%), mais les scores par question peuvent varier de ±5-12% entre runs. L'agent est légèrement plus stable globalement.

### Évolution des performances

Le système a été construit de manière itérative. Chaque composant du pipeline a été évalué sur le même benchmark de 18 questions :

| Version | Configuration | Global | Correctness | Temps/q |
|---|---|---|---|---|
| v1 — Baseline | Semantic seul, pas de reranker | 86% | 65% | 6.3s |
| v2 — Query Expansion | + LLM multi-query (×3 reformulations) | 89% | 73% | 13.2s |
| v3 — Cross-Encoder | + BGE reranker v2 m3 (568M) | 92% | 78% | 8.2s |
| v4 — Jina Reranker | BGE → Jina v2 multilingual (278M, 7× plus rapide) | 92% | 83% | 9.5s |
| v5 — Rechunking | Overlap 50w, heading propagé, split sémantique | 93% | 84% | 31.9s |
| v6 — BM25 Boost | BM25 weight ×1.5, fix éval keywords | 92% | 81% | 14.0s |
| v7 — Dual Generation | Self-consistency via context order | 93% | 84% | 17.3s |
| v8 — Intent Classification | 7 intents (dont "refus"), prompts spécialisés | 89.1%* | 80.4% | 21.0s |
| v9 — Agent LangGraph | Graph agent + 5 outils locaux | 90.6%* | 83.6% | 24.8s |
| **v10 — Eval v5 (juge JSON)** | **Juge JSON structuré + paliers fixes + calibration 12B** | **88.9%**** | **81.4%** | **24.8s** |

> \* v8–v9 : scores moyennés sur 3 runs (eval v4). Non directement comparables aux v1–v7 (eval v3).  
> \*\* v10 : même pipeline que v9, seul le juge d'évaluation a changé (eval v5 — juge JSON calibré). Scores non comparables aux v8–v9.

#### Changement de scoring (eval v3 → v4 → v5)

Les scores ne sont **pas directement comparables** entre générations d'évaluation :

| | Eval v3 (v1–v7) | Eval v4 (v8–v9) | Eval v5 (v10+) |
|---|---|---|---|
| Score final | 70% LLM-Judge + 30% Keywords | 55% Correctness + 25% Faithfulness + 20% Sources | Idem v4 |
| LLM-Judge | Score libre 0-100 (texte) | Score libre 0-100 (texte) | Paliers fixes 0/30/50/70/85/100 (JSON) |
| Format juge | Parsing texte `SCORE: X` | Parsing texte `SCORE: X` | JSON structuré (`format='json'` Ollama) |
| Calibration | — | — | Biais positif contrôlé pour modèles 12B |
| Conciseness | 10% du score | Supprimé | Supprimé |
| Multi-run | Non | Oui (3 runs) | Oui (3 runs) |

#### Gains par composant

```
Semantic seul                   86% ─────────────────────┐
+ Query Expansion LLM           89%  (+3%)               │ Retrieval
+ Cross-Encoder Reranking       92%  (+3%)               │ augmenté
+ Rechunking intelligent        93%  (+1%)  ─────────────┘
+ Dual Generation               93%  (stabilité)          → Robustesse
+ Intent Classification         89%  (scoring v4 strict)  → Prompts ciblés
+ Agent LangGraph               91%  (+1.5% vs natif)     → Outils + contrôle
```

**Contribution clé de chaque composant :**

| Composant | Impact principal | Gain |
|---|---|---|
| **Query Expansion** | Meilleur recall — reformulations capturent les synonymes RGPD | +3% global, +8% correctness |
| **Cross-Encoder** | Meilleure précision — reranking fin vs cosine grossier | +3% global |
| **Rechunking** | Chunks auto-suffisants — overlap + heading + split sémantique | +1% global, +6% correctness |
| **Dual Generation** | Robustesse — détecte les contradictions entre sources | +2% correctness, q10 63%→89% |
| **Intent Classification** | Prompts spécialisés par type de question + refus hors-périmètre | q18 69%→92% (agent) |
| **Agent LangGraph** | Enrichissement, décomposition, vérification complétude | +1.5% global, +23 pts q18 vs natif |
| **Eval v5 (juge JSON)** | Paliers fixes + JSON structuré + calibration 12B — scoring plus fiable | Thermomètre recalibré, scores non comparables |

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
│  │ Intent Classif.  │  7 intents : factuel, méthodologique,         │
│  │ (Phase 0)        │  organisationnel, comparaison, cas_pratique,  │
│  │                  │  liste_exhaustive, refus                      │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ├─── intent = "refus" ──→ Réponse de refus directe        │
│           │                                                         │
│           ▼                                                         │
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
│           │                                                         │
│           ├─── Pipeline Natif ──────┐   ┌── Pipeline Agent ────┐    │
│           │                         │   │                      │    │
│           ▼                         │   ▼                      │    │
│  ┌──────────────────┐     ┌──────────────────────────────────┐ │    │
│  │ Prompt spécialisé│     │ LangGraph Agent                  │ │    │
│  │ par intent       │     │  ┌────────────────────────────┐  │ │    │
│  │ + Génération LLM │     │  │ rewrite → classify →       │  │ │    │
│  │                  │     │  │ enrich → retrieve →        │  │ │    │
│  └────────┬─────────┘     │  │ generate → validate →      │  │ │    │
│           │               │  │ check_completeness →       │  │ │    │
│           │               │  │ respond                    │  │ │    │
│           │               │  └────────────────────────────┘  │ │    │
│           │               │  5 outils : DateCalculator,      │ │    │
│           │               │  ArticleLookup, TopicSearch,     │ │    │
│           │               │  QuestionDecomposer,             │ │    │
│           │               │  CompletenessChecker             │ │    │
│           │               └──────────────┬───────────────────┘ │    │
│           │                              │                     │    │
│           └──────────────┬───────────────┘                     │    │
│                          ▼                                          │
│  ┌─────────────────┐                                                │
│  │ Grounding       │  Vérifie les citations [Source X]              │
│  │ Validation      │                                                │
│  └────────┬────────┘                                                │
│           ▼                                                         │
│     Réponse finale + sources citées                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Intent Classification — Routage intelligent

Avant toute retrieval, le LLM classifie la question en **7 intents** via un prompt structuré :

| Intent | Description | Prompt spécialisé |
|---|---|---|
| `factuel` | Questions de définition/fait | Réponse concise, références directes |
| `methodologique` | Questions "comment faire" | Étapes numérotées, démarche pratique |
| `organisationnel` | Organisation/rôles | Acteurs, responsabilités, organigramme |
| `comparaison` | Mise en regard de concepts | Tableau comparatif, points communs/différences |
| `cas_pratique` | Situation concrète | Analyse du cas, règles applicables |
| `liste_exhaustive` | Énumération complète | Liste structurée, exhaustive |
| `refus` | Hors-périmètre / contournement | Refus ferme en 1-3 phrases, rappel sanctions |

La classification suit des règles strictes : **REFUS D'ABORD** (contournement → refus), **FACTUEL PAR DÉFAUT** ("Qui est X?" → factuel), jamais de faux refus sur les questions légitimes.

### Agent LangGraph — Orchestration intelligente

Le pipeline agent utilise un graphe LangGraph avec 8 nœuds et 5 outils locaux (aucun appel externe) :

1. **rewrite** : Résolution multi-turn — reformule les questions de suivi ("et pour eux ?") en questions autonomes
2. **classify** : Classification d'intent (7 types : factuel, méthodologique, organisationnel, etc.)
3. **enrich** : Décomposition en sous-questions + calculs de dates + guards anti-confusion
4. **retrieve** : Retrieval hybride + reranking (40 candidats → top 10)
5. **generate** : Génération LLM avec prompt spécialisé par intent
6. **validate** : Vérification des citations (grounding), retry automatique si échec
7. **check_completeness** : Vérifie que toutes les sous-questions sont couvertes, re-retrieve si non
8. **respond** : Formatage de la réponse finale + métadonnées

## 🛠️ Stack Technique

| Composant | Technologie | Détails |
|---|---|---|
| **LLM** | Mistral-Nemo 12B | Via Ollama, 128K context, temperature 0.0 |
| **Embeddings** | BGE-M3 (BAAI) | sentence-transformers, 1024 dims, FP16 GPU |
| **VectorDB** | ChromaDB | PersistentClient, 14 388 chunks |
| **Reranker** | Jina Reranker v2 | 278M params, multilingual, CPU |
| **BM25** | rank_bm25 | Index sparse pour recherche hybride |
| **Agent** | LangGraph 1.0 | Graphe de nœuds, 5 outils locaux, state management |
| **Intent** | Classification LLM | 7 intents, prompt structuré, JSON output |
| **Interface** | Streamlit multipage | Chat + Dashboard + Documents entreprise |
| **Observabilité** | JSONL + Alerter | Logs structurés, feedback, alertes SMTP |
| **GPU** | RTX 4070 Ti 12GB | LLM + embeddings en VRAM, reranker sur CPU |

### Structure du projet

```
RAG-DPO/
├── app.py                      # Point d'entrée Streamlit multipage
├── update_cnil.py              # Mise à jour incrémentale base CNIL (~1x/mois)
├── pages/
│   ├── 1_💬_Chat.py            # Chat RAG interactif + feedback + toggle Agent/Natif
│   ├── 2_📊_Dashboard.py       # Dashboard observabilité (métriques, alertes)
│   └── 3_📄_Documents.py       # Gestion documents entreprise
├── test_rag.py                 # Test RAG en ligne de commande
├── check_install.py            # Vérification de l'installation
├── rebuild_pipeline.py         # Reconstruction du pipeline de données
├── requirements.txt            # Dépendances Python
├── configs/
│   ├── config.yaml             # Configuration centralisée (RAG + observabilité + SMTP)
│   └── enterprise_tags.json    # Registre des tags entreprise (auto-généré)
├── src/
│   ├── rag/                    # 🧠 Cœur du système RAG
│   │   ├── pipeline.py         # Orchestration (intent-aware, dual-gen)
│   │   ├── intent_classifier.py # Classification 7 intents (Phase 0)
│   │   ├── retriever.py        # Hybrid retrieval (BM25 + semantic + RRF)
│   │   ├── query_expander.py   # Multi-query expansion via LLM
│   │   ├── bm25_index.py       # Index BM25 (summaries + chunks)
│   │   ├── reranker.py         # Cross-encoder Jina reranking
│   │   ├── context_builder.py  # Construction contexte + 7 prompts spécialisés
│   │   ├── generator.py        # Génération LLM (Ollama)
│   │   ├── validators.py       # Grounding + relevance validation
│   │   └── agent/              # 🤖 Pipeline Agent LangGraph
│   │       ├── __init__.py     # Export run_agent_pipeline()
│   │       ├── state.py        # RAGState TypedDict
│   │       ├── graph.py        # Graphe LangGraph (7 nœuds)
│   │       ├── nodes.py        # Fonctions de nœuds (classify, enrich, retrieve…)
│   │       └── tools.py        # 5 outils locaux (dates, articles, topics…)
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
│   ├── qa_dataset.json         # 18 questions benchmark (6 catégories)
│   ├── run_eval.py             # Évaluation 4 axes + multi-run (--runs N)
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
git clone https://github.com/MatJoss/RAG-DPO.git
cd RAG-DPO
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Installer les modèles Ollama

```bash
ollama pull mistral-nemo        # LLM 12B (7.1 GB)
ollama pull llava:7b             # Vision — extraction texte images (4.7 GB)
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
| 💬 **Chat** | Interface Q&R RAG avec sources citées, feedback 👍/👎 et toggle Agent/Natif |
| 📊 **Dashboard** | Métriques temps réel, alertes, feedback, export JSON |
| 📄 **Documents** | Gestion des documents entreprise (import, liste, purge) |

Fonctionnalités du chat :
- **Toggle Agent / Natif** dans la sidebar (agent recommandé)
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
# Run simple (1 run)
python eval/run_eval.py --verbose

# Multi-run pour moyennage statistique (recommandé)
python eval/run_eval.py --runs 3 --verbose

# Pipeline agent
python eval/run_eval.py --agent --runs 3 --verbose

# Pipeline natif uniquement (pas de dual generation)
python eval/run_eval.py --no-dual --verbose
```

Exécute le benchmark 18 questions en 2 phases :
1. **Phase 1** : Génération RAG + scoring par mots-clés + similarité sémantique (BGE-M3 cosine)
2. **Phase 2** : LLM-as-Judge (le LLM évalue la qualité sémantique)

Score final = **55% Correctness** (50% LLM-Judge + 35% Semantic + 15% Keywords) + **25% Faithfulness** + **20% Sources**

## 🔄 Maintenance — Mise à jour de la base CNIL

La base CNIL évolue régulièrement (nouvelles sanctions, guides, recommandations). Un script dédié permet la mise à jour incrémentale (~1x/mois) :

```bash
# Voir l'état actuel de la base
python update_cnil.py --status

# Mise à jour complète (scraping → classification → chunking → indexation)
python update_cnil.py

# Voir ce qui serait fait sans rien exécuter
python update_cnil.py --dry-run

# Seulement vérifier les modifications côté CNIL
python update_cnil.py --scrape-only

# Forcer une réindexation complète de ChromaDB
python update_cnil.py --force-reindex
```

Le scraping utilise des requêtes conditionnelles (`If-Modified-Since` → `304 Not Modified`) pour ne re-télécharger que les pages modifiées. Les étapes suivantes (classification, chunking, résumés) détectent automatiquement les documents déjà traités.

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
