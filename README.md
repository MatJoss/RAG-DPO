# 🔒 RAG-DPO — Assistant RGPD pour DPO

**Système RAG (Retrieval-Augmented Generation) spécialisé en protection des données personnelles**, conçu pour assister les DPO dans leurs missions quotidiennes. Entièrement local, sans envoi de données à un tiers.

> **Score benchmark : 90.4% ± 0.4%** sur 48 questions × 3 runs (scoring v7, 11 catégories)
> Pondéré par catégorie : **92.3%** — 94% des questions ≥ 85%, faithfulness 99.7%.

---

## 📋 Table des matières

- [Résultats benchmark](#-résultats-benchmark)
- [Avant / Après — Impact du chunking](#-avant--après--impact-du-chunking)
- [Architecture](#️-architecture)
- [Pipeline de données](#-pipeline-de-données)
- [Stack technique](#️-stack-technique)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Maintenance CNIL](#-maintenance--mise-à-jour-de-la-base-cnil)
- [Pipeline entreprise](#-pipeline-entreprise)
- [Observabilité](#-observabilité)
- [Configuration](#-configuration)
- [Évolution du système](#-évolution-du-système)
- [Licence](#-licence)

---

## 📊 Résultats benchmark

**Benchmark v10** — 48 questions, 3 runs, scoring v7 (55% Correctness + 25% Faithfulness + 20% Sources, score LLM brut 0-100)

### Score global

| Métrique | Valeur |
|---|---|
| **Score global** | **90.4% ± 0.4%** |
| **Score pondéré (par catégorie)** | **92.3%** |
| Runs individuels | 90.3%, 90.1%, 90.8% |
| Questions ≥ 90% | **32/48** (67%) |
| Questions ≥ 85% | **45/48** (94%) |
| Questions < 80% | **2** (q15, q31) |
| Écart moyen par question | 0.033 |
| Écart max | 0.40 |

### Par catégorie (11 catégories)

| Catégorie | Score |
|---|---|
| 🏢 **Organisation** | **97.0%** |
| 📝 **Contractuel** | **95.3%** |
| ⚙️ **Procédure** | **94.0%** |
| 📋 **Documentation** | **93.0%** |
| 🔧 **Méthodologie** | **92.7%** |
| 🪤 **Piège** | **92.2%** |
| 🚫 **Hors périmètre** | **92.0%** |
| 📖 **Définition** | **91.2%** |
| 🌍 **International** | **91.0%** |
| ⚖️ **Obligation** | **88.8%** |
| 💡 **Recommandation** | **88.2%** |

### Sous-scores moyens

| Métrique | Score | Description |
|---|---|---|
| **Correctness** | 83.7% | Exactitude factuelle vs réponse de référence |
| **Faithfulness** | 99.7% | Fidélité aux sources (pas d'hallucination) |
| **Sources** | 97.6% | Citations correctes [Source N] |
| **LLM-Judge** | 90.9% | Évaluation qualitative par LLM |
| **Semantic Sim** | 72.7% | Similarité sémantique avec la référence |
| **Keywords** | 86.2% | Présence des mots-clés attendus |

### Scores par question (48 questions)

| # | Question | Cat. | Score |
|---|---|---|---|
| q01 | Qu'est-ce qu'une donnée personnelle ? | Définition | **93.0%** |
| q02 | Qui est responsable de traitement ? | Définition | **94.0%** |
| q03 | RT vs sous-traitant ? | Définition | **89.0%** |
| q04 | Quand une AIPD est-elle obligatoire ? | Obligation | **90.3%** |
| q05 | Critères WP29 déclenchant une AIPD ? | Recommandation | **93.7%** |
| q06 | Liste AIPD CNIL ? | Recommandation | **93.3%** |
| q07 | Obligations du responsable de traitement ? | Obligation | **93.0%** |
| q08 | Droits des personnes et limites ? | Définition | **89.3%** |
| q09 | Conserver des CV indéfiniment ? | Recommandation | **90.0%** |
| q10 | Intérêt légitime pour la vidéosurveillance ? | Recommandation | **91.0%** |
| q11 | Opposition à un traitement RH ? | Recommandation | **85.7%** |
| q12 | Conservation de données 50 ans ? | Piège | **91.7%** |
| q13 | DPO obligatoire partout ? | Obligation | **94.3%** |
| q14 | Article 99 du RGPD sur l'IA ? | Piège | **97.0%** |
| q15 | Quand faire une étude d'impact ? | Obligation | **79.3%** |
| q16 | Qui décide des moyens du traitement ? | Définition | **91.0%** |
| q17 | Meilleure base marketing 2024 ? | Hors périmètre | **91.3%** |
| q18 | Contourner une obligation CNIL ? | Hors périmètre | **94.0%** |
| q19 | Conditions de validité du consentement ? | Définition | **93.0%** |
| q20 | Fondement contractuel d'un traitement ? | Définition | **92.0%** |
| q21 | Délai notification violation CNIL ? | Obligation | **91.3%** |
| q22 | Informer les personnes d'une violation ? | Obligation | **86.0%** |
| q23 | Registre des traitements obligatoire ? | Obligation | **88.0%** |
| q24 | Qu'est-ce que la pseudonymisation ? | Définition | **90.0%** |
| q25 | Donnée anonymisée = donnée personnelle ? | Définition | **94.7%** |
| q26 | Conditions transfert hors UE ? | Obligation | **89.0%** |
| q27 | Sanctions possibles de la CNIL ? | Définition | **85.0%** |
| q28 | Catégories particulières de données ? | Définition | **93.0%** |
| q29 | Qu'est-ce que le profilage ? | Définition | **90.7%** |
| q30 | Décision entièrement automatisée ? | Obligation | **87.7%** |
| q31 | Durée conservation bulletins de paie ? | Recommandation | **76.7%** |
| q32 | Consentement pour tous les cookies ? | Recommandation | **90.0%** |
| q33 | Prospection commerciale d'un client ? | Recommandation | **86.7%** |
| q34 | Sous-traitant cloud hors UE ? | Recommandation | **87.7%** |
| q35 | RGPD interdit données sensibles ? | Piège | **92.7%** |
| q36 | Refuser une demande d'accès ? | Piège | **87.7%** |
| q37 | Antivirus recommandé par la CNIL ? | Hors périmètre | **90.7%** |
| q38 | Comment pirater un site web ? | Hors périmètre | **92.0%** |
| q39 | Durée conservation données clients ? | Recommandation | **86.7%** |
| q40 | Transférer données sensibles cloud hors UE ? | Recommandation | **83.0%** |
| q41 | Cookies de tracking sans consentement ? | Recommandation | **93.0%** |
| q42 | Âge minimum consentement enfant en ligne ? | Recommandation | **88.7%** |
| q43 | AIPD : obligation, méthode, acteurs ? | Méthodologie | **92.7%** |
| q44 | Violation de données : délai et notification ? | Procédure | **94.0%** |
| q45 | DPO : obligation et missions ? | Organisation | **97.0%** |
| q46 | Transfert hors UE : conditions et mécanismes ? | International | **91.0%** |
| q47 | Registre des traitements : contenu et obligation ? | Documentation | **93.0%** |
| q48 | Sous-traitant : obligations et contrat ? | Contractuel | **95.3%** |

---

## 🔄 Avant / Après — Impact du chunking

Le passage de la version v6b à v7 (mars 2026) a introduit deux changements majeurs :
1. **Détection de tableaux content-based** : les tableaux dans les HTML, PDF et DOCX sont désormais extraits et convertis en texte naturel via LLM, au lieu d'être ignorés ou aplatis
2. **Tags RGPD guidés** : 25 catégories normalisées remplacent les ~7 500 tags libres anarchiques

### Gains mesurés

> **Note** : les scores v6b (Avant) utilisaient un scoring discrétisé qui gonflait les résultats. Les scores v10 (Après) utilisent le scoring brut v7 — les deltas ne sont donc pas directement comparables.

| Métrique | Avant (v6b, scoring discrétisé) | Après (v10, scoring brut) | Δ |
|---|---|---|---|
| **Score global** | 89.2% ± 1.1%¹ | **90.4% ± 0.4%** | **+1.2 pts** (réel ~+3.6 pts) |
| Score pondéré | 89.6%¹ | **92.3%** | **+2.7** |
| Questions ≥ 85% | — | **45/48** (94%) | — |
| Questions < 80% | 4 | **2** | -2 |
| Std (stabilité) | 0.011 | **0.004** | ÷2.75 |
| Dataset | 42 questions (5 cat.) | **48 questions (11 cat.)** | +6 composites |
| Chunks dans ChromaDB | ~14 400 | **16 919** | +2 519 |

¹ *Scoring discrétisé (v6) : les scores LLM étaient arrondis à 6 paliers (0/25/50/75/90/100), gonflant les résultats d'environ +2.4 pts.*

### Par catégorie (vs baseline v6b)

| Catégorie | Avant¹ | Après (brut) | Δ |
|---|---|---|---|
| Définition | 93.3% | **91.2%** | -2.1¹ |
| Obligation | 92.0% | **88.8%** | -3.2¹ |
| Recommandation | 83.1% | **88.2%** | **+5.1** |
| Piège | 91.2% | **92.2%** | **+1.0** |
| Hors périmètre | 88.3% | **92.0%** | **+3.7** |

¹ *Les baisses apparentes en définition/obligation s'expliquent par le passage au scoring brut (non discrétisé). Les catégories « recommandation » et « hors périmètre » progressent réellement.*

**6 nouvelles catégories composites** (questions multi-aspects, v10) :

| Catégorie | Score | Question type |
|---|---|---|
| Organisation | **97.0%** | DPO : obligation et missions |
| Contractuel | **95.3%** | Sous-traitant : obligations et contrat |
| Procédure | **94.0%** | Violation de données : délai et notification |
| Documentation | **93.0%** | Registre des traitements : contenu et obligation |
| Méthodologie | **92.7%** | AIPD : obligation, méthode, acteurs |
| International | **91.0%** | Transfert hors UE : conditions et mécanismes |

> Ces 6 questions composites testent la capacité de **query decomposition** : le système décompose en sous-questions, fait une seule retrieval globale, puis génère une réponse structurée en sections. Score moyen : **93.8%**.

### Top 3 des améliorations (vs baseline)

| Question | Avant¹ | Après (brut) | Gain |
|---|---|---|---|
| q41 — Cookies de tracking | 74.0% | **93.0%** | **+19.0 pts** |
| q12 — Conservation 50 ans (piège) | 82.7% | **91.7%** | **+9.0 pts** |
| q40 — Données sensibles cloud hors UE | 85.0% | **83.0%** | -2.0 pts |

> **La catégorie « recommandation »** est celle qui progresse le plus (+5.1 pts réels), grâce aux tableaux CNIL correctement extraits et aux prompts optimisés.

### Leçon clé

> **Le chunking est la fondation du RAG.** Aucun réglage de pipeline (top-k, reranking, prompt) ne peut compenser des données mal extraites à la source. Quand l'information correcte n'est pas dans les chunks, aucune reformulation ne la fera apparaître.

---

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
│           ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                 LangGraph Agent (9 nœuds)                 │       │
│  │                                                          │       │
│  │  rewrite → classify → enrich → decompose ─┐              │       │
│  │                                            │              │       │
│  │      ┌─── composite (≥2 sous-questions) ───┤              │       │
│  │      │    1 retrieval global (question orig.)│              │       │
│  │      │    1 génération structurée (sections) │              │       │
│  │      │    renumbering [Source N] global      │              │       │
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
│  │  5 outils : DateCalculator, ArticleLookup,                │       │
│  │  TopicSearch, QuestionDecomposer, CompletenessChecker     │       │
│  └──────────────────────┬───────────────────────────────────┘       │
│                         ▼                                           │
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

### Agent LangGraph — Orchestration intelligente

Le pipeline agent utilise un graphe LangGraph avec **9 nœuds** et **5 outils locaux** (aucun appel externe) :

1. **rewrite** : Résolution multi-turn — reformule les questions de suivi en questions autonomes
2. **classify** : Classification d'intent (7 types)
3. **enrich** : Calculs de dates + guards anti-confusion + injection d'articles RGPD
4. **decompose** : Détecte les questions multi-aspects → retrieve+generate par sous-question → fusion programmatique (voir ci-dessous)
5. **retrieve** : Retrieval hybride + reranking (40 candidats → top 10)
6. **generate** : Génération LLM avec prompt spécialisé par intent
7. **validate** : Vérification des citations (grounding), retry automatique si échec
8. **check_completeness** : Vérifie que toutes les sous-questions sont couvertes, re-retrieve si non
9. **respond** : Formatage de la réponse finale + métadonnées

#### Query Decomposition (nœud `decompose`)

Les questions composites (ex: *"Dois-je faire une AIPD, comment, et qui contacter ?"*) sont automatiquement décomposées en sous-questions atomiques, chacune traitée indépendamment (retrieve + generate). Les sous-réponses sont ensuite fusionnées **programmatiquement** (pas par LLM) :

- **Pourquoi pas de fusion LLM ?** — Mistral-Nemo 12B supprime systématiquement les citations `[Source N]` quand on lui demande de fusionner/réécrire du texte. La fusion programmatique préserve 100% des citations.
- **Renumbering des sources** : chaque sous-question a ses sources numérotées localement (1-5). Après déduplications, les `[Source N]` sont renumérotées vers un index global cohérent.
- **Coût** : 0 appel LLM pour la fusion (vs ~20-30s avec le LLM), latence réduite de ~20%.
- **UI** : les sous-réponses individuelles sont visibles dans un expander dédié sous la réponse.

---

## 📄 Pipeline de données

Le pipeline de traitement transforme les documents CNIL bruts en chunks vectorisés dans ChromaDB.

### Étapes du pipeline

```
1. Scraping CNIL          → pages HTML brutes (1 829 documents)
2. Classification          → nature du document (doctrine, guide, sanction, technique)
3. Filtrage pertinence     → documents non-RGPD exclus
4. Résumés LLM            → fiche synthétique par document (BM25 pre-filter)
5. Chunking sémantique     → 16 919 chunks (overlap 50 mots, split sémantique, heading propagé)
6. Indexation ChromaDB     → embeddings BGE-M3 1024d
7. Tagging RGPD            → 25 catégories normalisées par chunk
```

### Chunking content-based

Le chunker détecte et traite les tableaux **par contenu**, pas par extension de fichier :

| Format | Détection | Méthode | Documents concernés |
|---|---|---|---|
| **HTML** | Éléments `<table>` dans le DOM | `_convert_html_table()` | 39 documents |
| **PDF** | `find_tables()` de PyMuPDF | `_extract_pdf_tables()` | 264 documents (54%) |
| **DOCX** | `w:tbl` dans le DOM python-docx | `doc.tables` | 1 document |
| **XLSX/ODS** | Toujours (spreadsheet natif) | `_chunk_spreadsheet()` | 27 documents |

Tous convergent vers **`_convert_table_rows()`** — un pipeline commun :
1. Découpage en zones (heading + lignes de données)
2. Split si zone > 500 mots
3. Conversion en texte pipe-delimited
4. Réécriture LLM (Mistral-Nemo) en texte naturel
5. Fallback mécanique si le LLM échoue

### Tags RGPD guidés

Chaque chunk est taggé parmi **25 catégories RGPD normalisées** :

```
droits des personnes, consentement, sécurité des données,
durée de conservation, sous-traitance, base légale,
données sensibles, transfert hors UE, cookies,
violation de données, transparence, DPO,
vidéosurveillance, finalité du traitement, registre des traitements,
AIPD, anonymisation, minimisation, responsable de traitement,
prospection commerciale, conformité RGPD, profilage,
sanctions CNIL, données de santé, information des personnes
```

Le prompt LLM guide le modèle vers ce vocabulaire contrôlé, éliminant les tags anarchiques (7 500 → ~25).

---

## 🛠️ Stack Technique

| Composant | Technologie | Détails |
|---|---|---|
| **LLM** | Mistral-Nemo 12B | Via Ollama, 128K context, temperature 0.0 |
| **Embeddings** | BGE-M3 (BAAI) | sentence-transformers, 1024 dims, FP16 GPU |
| **VectorDB** | ChromaDB | PersistentClient, 16 919 chunks |
| **Reranker** | Jina Reranker v2 | 278M params, multilingual, GPU |
| **BM25** | rank_bm25 | Index sparse pour recherche hybride |
| **Agent** | LangGraph 1.0 | Graphe 9 nœuds, query decomposition, génération structurée, 5 outils locaux |
| **Intent** | Classification LLM | 7 intents, prompt structuré, JSON output |
| **Interface** | Streamlit multipage | Chat + Dashboard + Documents entreprise |
| **Observabilité** | JSONL + Alerter | Logs structurés, feedback, alertes SMTP |
| **GPU** | RTX 4070 Ti 12GB | LLM + embeddings en VRAM, reranker sur CPU |

### Structure du projet

```
RAG-DPO/
├── app.py                      # Point d'entrée Streamlit multipage
├── update_cnil.py              # Mise à jour incrémentale base CNIL (~1x/mois)
├── rebuild_pipeline.py         # Reconstruction du pipeline de données
├── tag_all_chunks.py           # Tagging RGPD guidé (25 catégories)
├── pages/
│   ├── 1_💬_Chat.py            # Chat RAG interactif + feedback + toggle Agent/Natif
│   ├── 2_📊_Dashboard.py       # Dashboard observabilité (métriques, alertes)
│   └── 3_📂_Documents.py       # Gestion documents entreprise
├── test_rag.py                 # Test RAG en ligne de commande
├── check_install.py            # Vérification de l'installation
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
│   │   ├── reranker.py         # Cross-encoder Jina reranking + topic boost
│   │   ├── context_builder.py  # Construction contexte + 7 prompts spécialisés
│   │   ├── generator.py        # Génération LLM (Ollama)
│   │   ├── validators.py       # Grounding + relevance validation
│   │   └── agent/              # 🤖 Pipeline Agent LangGraph
│   │       ├── graph.py        # Graphe LangGraph (9 nœuds, decompose + routing)
│   │       ├── nodes.py        # Fonctions de nœuds (incl. fusion programmatique)
│   │       ├── tools.py        # 5 outils locaux
│   │       └── state.py        # RAGState TypedDict
│   ├── processing/             # 📄 Pipeline de traitement des données
│   │   ├── process_and_chunk.py        # Chunking sémantique + détection tableaux
│   │   ├── create_chromadb_index.py    # Indexation vectorielle BGE-M3
│   │   ├── generate_document_summaries.py  # Fiches synthétiques LLM
│   │   ├── hybrid_filter.py            # Filtrage pertinence
│   │   ├── classify_documents.py       # Classification par nature
│   │   └── ingest_enterprise.py        # Ingestion docs entreprise
│   ├── scraping/               # 🕷️ Scraping CNIL
│   │   └── cnil_scraper_final.py
│   └── utils/
│       ├── llm_provider.py     # Interface Ollama
│       ├── embedding_provider.py # Provider BGE-M3 (FP16, GPU, lazy load)
│       ├── rgpd_topics.py      # 25 catégories RGPD + TopicMatcher + prompts
│       ├── query_logger.py     # Logger JSONL queries + feedback
│       ├── structured_logger.py # Logging JSON structuré
│       ├── alerter.py          # Alertes seuils + SMTP
│       └── acronyms.py         # Expansion acronymes RGPD
├── eval/                       # 📊 Framework d'évaluation
│   ├── qa_dataset.json         # 48 questions benchmark (5 cat. + 6 composites)
│   ├── run_eval.py             # Évaluation 4 axes + multi-run (--runs N, --top-k K)
│   └── results_*.json          # Résultats historiques
├── logs/                       # 📝 Logs structurés (non versionné)
├── data/                       # 📁 Données (non versionné)
│   ├── raw/                    # Documents bruts CNIL
│   ├── vectordb/chromadb/      # Base vectorielle ChromaDB
│   └── metadata/               # Métadonnées documents
└── tasks/                      # 📝 Notes de travail
    ├── todo.md
    └── lessons.md              # Leçons apprises (patterns, erreurs, fixes)
```

---

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
python rebuild_pipeline.py              # Pipeline complet (scraping → indexation)
python rebuild_pipeline.py --from 5b    # Reprendre depuis le chunking
python rebuild_pipeline.py --fresh      # Forcer le retraitement de tous les docs
```

---

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
| 📂 **Documents** | Gestion des documents entreprise (import, liste, purge) |

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
# Multi-run recommandé (3 runs pour moyennage statistique)
python eval/run_eval.py --agent --runs 3 --verbose

# Run simple
python eval/run_eval.py --verbose

# Tester avec un top_k différent (défaut: 10)
python eval/run_eval.py --agent --runs 3 --top-k 5
```

Exécute le benchmark 48 questions (42 simples + 6 composites) en 2 phases :
1. **Phase 1** : Génération RAG + scoring par mots-clés + similarité sémantique (BGE-M3 cosine)
2. **Phase 2** : LLM-as-Judge — score libre 0-100 en JSON, utilisé directement (pas de discrétisation)

Score final = **55% Correctness** (60% LLM-Judge + 40% Semantic) + **25% Faithfulness** + **20% Sources**

---

## 🔄 Maintenance — Mise à jour de la base CNIL

La base CNIL évolue régulièrement (nouvelles sanctions, guides, recommandations). Un script dédié permet la mise à jour incrémentale (~1x/mois) :

```bash
python update_cnil.py --status          # État actuel de la base
python update_cnil.py                   # Mise à jour complète
python update_cnil.py --dry-run         # Aperçu sans exécution
python update_cnil.py --scrape-only     # Vérifier modifications côté CNIL
python update_cnil.py --force-reindex   # Réindexation complète ChromaDB
```

Le scraping utilise des requêtes conditionnelles (`If-Modified-Since` → `304 Not Modified`) pour ne re-télécharger que les pages modifiées. Les étapes suivantes (classification, chunking, résumés, tagging) détectent automatiquement les documents déjà traités.

---

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

- **Formats supportés** : PDF, DOCX, XLSX, ODS, HTML, TXT
- **Détection de tableaux** : automatique pour tous les formats (content-based)
- **Déduplication** par hash SHA256 (relancer = pas de doublons)
- **Tags** par document pour filtrage dans l'UI
- **CNIL prévaut toujours** sur les docs entreprise dans les réponses

---

## 📊 Observabilité

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

---

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

---

## 📈 Évolution du système

| Version | Composant | Impact |
|---|---|---|
| v1 | Semantic search + ChromaDB | Baseline 70% |
| v2 | Nomic embeddings + BM25 | +8% |
| v3 | Query Expansion LLM | +3% (recall) |
| v4 | Cross-Encoder Jina v2 | +3% (precision) |
| v5 | Rechunking intelligent (overlap, heading, split sémantique) | +1% |
| v6 | Intent Classification (7 intents, prompts spécialisés) | Prompts ciblés |
| v6b | Agent LangGraph (8 nœuds, 5 outils) | +1.5% + robustesse |
| v6c | Eval v5 → v6 (score libre JSON, 42 questions) | Thermomètre fiable |
| v6d | Migration BGE-M3 (remplace nomic, 1024d) | Embeddings FR natifs |
| **v7** | **Détection tableaux content-based + tags RGPD guidés** | **+0.5%** (89.2%¹ → 89.7%) |
| **v8** | **Suppression discrétisation LLM-Judge (scoring brut v7)** | Scores réels (non gonflés) |
| **v9** | **Query Decomposition + fusion programmatique** | Citations préservées sur questions composites |
| **v10** | **Génération unique structurée + prompts DPO-centric** | 90.4% — single retrieval+gen, ×3 plus rapide, 11 catégories, 48 questions |

### Scoring (eval v3 → v7)

Les scores ne sont **pas directement comparables** entre générations d'évaluation :

| | Eval v3 (v1–v5) | Eval v4 (v6–v6b) | Eval v6 (v6c–v7) | Eval v7 (v8+) |
|---|---|---|---|---|
| Score final | 70% LLM-Judge + 30% Keywords | 55% Correctness + 25% Faithfulness + 20% Sources | Idem v4 | Idem v4 |
| LLM-Judge | Score libre 0-100 (texte) | Score libre 0-100 (texte) | Score libre 0-100 (JSON, discrétisé) | **Score brut 0-100 (JSON, sans discrétisation)** |
| Dataset | 18 questions | 18 questions | **42 questions** (5 catégories) | **48 questions** (11 catégories) |
| Multi-run | Non | Oui (3 runs) | Oui (3 runs) | Oui (3 runs) |
| Calibration | — | — | Discrétisé → scores gonflés ~+2.4 pts | **Scores bruts, non gonflés** |

### Gains par composant

```
Semantic seul                   70% ─────────────────────┐
+ BM25 hybride                  78%  (+8%)               │ Retrieval
+ Query Expansion LLM           81%  (+3%)               │ augmenté
+ Cross-Encoder Reranking       84%  (+3%)               │
+ Rechunking intelligent        85%  (+1%)  ─────────────┘
+ Intent Classification         86%  (prompts ciblés)      → Précision
+ Agent LangGraph               89%  (+1.5%)               → Outils + contrôle
+ Tables content-based          89.7% (+0.5%)¹            → Données complètes
+ Query Decomposition           —     (qualité composites) → Citations préservées
+ Single Gen + prompts DPO      90.4% (+0.7%)              → Réponses détaillées
¹ Le gain réel du chunking est ~+2.9% mais masqué par la suppression
  de la discrétisation qui gonflait les scores précédents de ~2.4 pts.
```

---

## 📄 Licence

Ce projet est un outil éducatif et de recherche. Les données CNIL utilisées sont publiques.

## 🙏 Remerciements

- **CNIL** pour la mise à disposition des ressources sur la protection des données
- **Ollama** pour l'inférence locale simplifiée
- **Jina AI** pour le reranker multilingue open-source
- **Mistral AI** pour Mistral-Nemo 12B
- **BAAI** pour BGE-M3 (embeddings multilingues)
