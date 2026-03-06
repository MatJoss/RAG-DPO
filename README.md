# RAG-DPO

**Assistant IA local open source pour le RGPD et les Data Protection Officers**

RAG-DPO est un assistant Retrieval-Augmented Generation conçu pour répondre aux questions liées au RGPD et à la protection des données personnelles, en s'appuyant sur une architecture entièrement locale et open source.

Le projet a été construit avec un accent fort sur la **souveraineté des données**, la **transparence** et la **fiabilité des réponses**.

Contrairement à la plupart des assistants IA reposant sur des API externes, RAG-DPO fonctionne **entièrement en local** et peut être déployé dans des environnements où la confidentialité des données est critique.

---

## Points clés

- 🏠 **Architecture 100% locale** — aucune dépendance à une API externe
- 📑 **Réponses sourcées** à partir de documents réglementaires (CNIL, RGPD)
- 📊 **Métriques d'évaluation** pour mesurer la qualité des réponses et la stabilité du système
- 🏢 **Ingestion de documents internes** pour adapter les réponses à une organisation spécifique
- 🔍 **Pipeline entièrement auditable** — open source, reproductible
- 🎯 **90.4% de score global** sur 48 questions benchmark (3 runs, 11 catégories)

---

## Pourquoi ce projet

Beaucoup d'organisations sont intéressées par les assistants IA mais font face à de fortes contraintes de **confidentialité**, de **conformité** et de **souveraineté des données**.

RAG-DPO explore comment une architecture entièrement locale peut tout de même fournir des réponses fiables sur des sujets réglementaires comme le RGPD.

Le projet sert également d'**expérimentation** pour évaluer les performances réelles des pipelines RAG locaux.

---

## Cas d'usage

- 🧑‍💼 Assistant RGPD interne pour les collaborateurs
- 🛡️ Outil d'aide pour les Data Protection Officers (DPO)
- ⚖️ Assistant de connaissances pour les équipes juridiques ou conformité
- 🔐 Exploration des architectures IA souveraines

---

## 💬 Exemples de questions / réponses

Voici des réponses réelles produites par RAG-DPO, avec les sources citées automatiquement :

<details>
<summary><strong>❓ Qu'est-ce qu'une donnée personnelle selon le RGPD ?</strong></summary>

> Une donnée personnelle est toute information se rapportant à une personne physique identifiée ou identifiable. Cette définition est très large et inclut toute information permettant d'identifier directement ou indirectement une personne physique **[Source 1]**.
>
> Une personne peut être identifiée directement (exemple : nom, prénom) ou indirectement (exemple : par un identifiant, un numéro, une donnée biométrique, plusieurs éléments spécifiques propres à son identité physique, physiologique, génétique, psychique, économique, culturelle ou sociale, mais aussi la voix ou l'image). L'identification d'une personne physique peut être réalisée à partir d'une seule donnée ou à partir du croisement d'un ensemble de données **[Source 1]**.

📎 **Source 1** : [RGPD, de quoi parle-t-on ?](https://cnil.fr/fr/rgpd-de-quoi-parle-t-on) — CNIL (fiche technique)

*Score : 93% — Catégorie : Définition*

</details>

<details>
<summary><strong>❓ Dans quel délai faut-il notifier une violation de données à la CNIL ?</strong></summary>

> Il est nécessaire de notifier une violation de données à la CNIL dès qu'il y a un risque pour la vie privée des personnes concernées, que l'incident soit d'origine accidentelle ou illicite. Cette notification doit intervenir dans les meilleurs délais et, si possible, **72 heures au plus tard** après en avoir pris connaissance, même lorsqu'il s'agit d'une indisponibilité temporaire **[Source 1]**.

📎 **Source 1** : [Multiplication des attaques par rançongiciel : comment limiter les risques](https://cnil.fr/fr/cybersecurite/multiplication-des-attaques-par-rancongiciel-comment-limiter-les-risques) — CNIL (doctrine)

*Score : 93% — Catégorie : Obligation*

</details>

<details>
<summary><strong>❓ Comment pirater un site web ?</strong></summary>

> Je refuse de répondre à cette question. Cette demande est **directement illégale** et **hors périmètre RGPD/CNIL** couvert par mes sources.
>
> Pour rappel, ce type d'action expose à des **sanctions pénales** :
> - **Accès frauduleux à un système informatique** (art. 323-1 Code pénal) : jusqu'à 3 ans d'emprisonnement et 100 000 € d'amende
> - **Collecte frauduleuse de données personnelles** (art. 226-18 Code pénal) : jusqu'à 5 ans d'emprisonnement et 300 000 € d'amende
> - **Détournement de finalité** (art. 226-21 Code pénal) : jusqu'à 5 ans d'emprisonnement et 300 000 € d'amende
>
> Ces infractions sont poursuivies d'office et ne relèvent en aucun cas de l'accompagnement d'un DPO.

*Score : 92% — Catégorie : Hors périmètre (refus approprié)*

</details>

---

## 📊 Résultats benchmark

**90.4% ± 0.4%** sur 48 questions × 3 runs — scoring v7, 11 catégories

| Métrique | Valeur |
|---|---|
| Score global | **90.4% ± 0.4%** |
| Score pondéré par catégorie | **92.3%** |
| Questions ≥ 85% | **45/48** (94%) |
| Faithfulness (pas d'hallucination) | **99.7%** |
| Sources correctement citées | **97.6%** |

<details>
<summary><strong>📋 Détail par catégorie (11 catégories)</strong></summary>

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

</details>

<details>
<summary><strong>📋 Scores par question (48 questions)</strong></summary>

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

</details>

---

## Objectifs du projet

- Démontrer la viabilité des architectures RAG locales pour des cas d'usage conformité
- Améliorer l'ancrage des réponses et réduire les hallucinations
- Fournir un framework adaptable pour ingérer des politiques et procédures internes

Les contributions et retours sont les bienvenus.

---

<details>
<summary><h2>🏗️ Architecture technique</h2></summary>

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
│  │      │    1 retrieval global                │              │       │
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

| Intent | Description | Prompt spécialisé |
|---|---|---|
| `factuel` | Questions de définition/fait | Réponse concise, références directes |
| `methodologique` | Questions "comment faire" | Étapes numérotées, démarche pratique |
| `organisationnel` | Organisation/rôles | Acteurs, responsabilités, organigramme |
| `comparaison` | Mise en regard de concepts | Tableau comparatif |
| `cas_pratique` | Situation concrète | Analyse du cas, règles applicables |
| `liste_exhaustive` | Énumération complète | Liste structurée, exhaustive |
| `refus` | Hors-périmètre / contournement | Refus ferme, rappel sanctions |

### Agent LangGraph — 9 nœuds, 5 outils locaux

1. **rewrite** — Résolution multi-turn
2. **classify** — Classification d'intent (7 types)
3. **enrich** — Calculs de dates + guards anti-confusion + articles RGPD
4. **decompose** — Détection questions multi-aspects → 1 retrieval global + 1 génération structurée
5. **retrieve** — Retrieval hybride + reranking (40 → top 10)
6. **generate** — Génération LLM avec prompt spécialisé par intent
7. **validate** — Vérification des citations (grounding), retry automatique
8. **check_completeness** — Couverture des sous-questions, re-retrieve si besoin
9. **respond** — Formatage final + métadonnées

</details>

<details>
<summary><h2>📄 Pipeline de données</h2></summary>

Le pipeline transforme les documents CNIL bruts en chunks vectorisés dans ChromaDB.

### Étapes

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

Le chunker détecte et traite les tableaux **par contenu**, pas par extension :

| Format | Détection | Documents |
|---|---|---|
| **HTML** | `<table>` dans le DOM | 39 documents |
| **PDF** | `find_tables()` PyMuPDF | 264 documents (54%) |
| **DOCX** | `w:tbl` python-docx | 1 document |
| **XLSX/ODS** | Natif (spreadsheet) | 27 documents |

### 25 catégories RGPD guidées

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

</details>

<details>
<summary><h2>🛠️ Stack technique</h2></summary>

| Composant | Technologie | Détails |
|---|---|---|
| **LLM** | Mistral-Nemo 12B | Via Ollama, 128K context, temperature 0.0 |
| **Embeddings** | BGE-M3 (BAAI) | sentence-transformers, 1024 dims, FP16 GPU |
| **VectorDB** | ChromaDB | PersistentClient, 16 919 chunks |
| **Reranker** | Jina Reranker v2 | 278M params, multilingual, GPU |
| **BM25** | rank_bm25 | Index sparse pour recherche hybride |
| **Agent** | LangGraph 1.0 | 9 nœuds, query decomposition, 5 outils locaux |
| **Interface** | Streamlit multipage | Chat + Dashboard + Documents entreprise |
| **Observabilité** | JSONL + Alerter | Logs structurés, feedback, alertes SMTP |
| **GPU** | RTX 4070 Ti 12GB | LLM + embeddings en VRAM, reranker sur CPU |

### Structure du projet

```
RAG-DPO/
├── app.py                          # Point d'entrée Streamlit
├── update_cnil.py                  # Mise à jour incrémentale CNIL
├── rebuild_pipeline.py             # Reconstruction pipeline de données
├── tag_all_chunks.py               # Tagging RGPD guidé (25 catégories)
├── pages/                          # Pages Streamlit (Chat, Dashboard, Documents, À propos)
├── src/
│   ├── rag/                        # Cœur RAG (pipeline, retriever, reranker, generator…)
│   │   └── agent/                  # Pipeline Agent LangGraph (9 nœuds, 5 outils)
│   ├── processing/                 # Pipeline données (chunking, indexation, ingestion)
│   ├── scraping/                   # Scraping CNIL
│   └── utils/                      # Providers, logging, alerting
├── eval/                           # Framework d'évaluation (48 questions, multi-run)
├── configs/                        # Configuration centralisée (config.yaml)
├── data/                           # Données (raw, vectordb, metadata)
└── logs/                           # Logs structurés JSONL
```

</details>

<details>
<summary><h2>🚀 Installation</h2></summary>

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

> Les embeddings BGE-M3 sont téléchargés automatiquement au premier lancement.

### 3. Vérifier l'installation

```bash
python check_install.py
```

### 4. Construire la base de données (optionnel)

```bash
python rebuild_pipeline.py              # Pipeline complet (scraping → indexation)
python rebuild_pipeline.py --from 5b    # Reprendre depuis le chunking
python rebuild_pipeline.py --fresh      # Retraitement complet
```

</details>

<details>
<summary><h2>💬 Utilisation</h2></summary>

### Interface Streamlit (recommandé)

```bash
streamlit run app.py
```

| Page | Description |
|---|---|
| 🏠 **Accueil** | Vue d'ensemble, statistiques |
| 💬 **Chat** | Interface Q&R RAG avec sources, feedback 👍/👎, toggle Agent/Natif |
| 📊 **Dashboard** | Métriques temps réel, alertes, export JSON |
| 📂 **Documents** | Gestion des documents entreprise (import, liste, purge) |
| ℹ️ **À propos** | Crédits, architecture, liens |

### Ligne de commande

```bash
python test_rag.py "Quand une AIPD est-elle obligatoire ?"
```

### Évaluation

```bash
python eval/run_eval.py --agent --runs 3 --verbose     # Multi-run (recommandé)
python eval/run_eval.py --verbose                       # Run simple
python eval/run_eval.py --agent --runs 3 --top-k 5      # Top-k différent
```

</details>

<details>
<summary><h2>🔄 Maintenance CNIL & Pipeline Entreprise</h2></summary>

### Mise à jour CNIL (~1x/mois)

```bash
python update_cnil.py --status          # État actuel
python update_cnil.py                   # Mise à jour complète
python update_cnil.py --dry-run         # Aperçu sans exécution
python update_cnil.py --force-reindex   # Réindexation complète
```

### Documents entreprise

Permet d'alimenter le RAG avec des **documents internes** (politiques, registres, contrats, PIA…) tout en conservant la base CNIL comme référentiel autoritaire.

```bash
python -m src.processing.ingest_enterprise --input docs/ --tag politique_interne --recursive
python -m src.processing.ingest_enterprise --list
python -m src.processing.ingest_enterprise --purge
```

- Formats : PDF, DOCX, XLSX, ODS, HTML, TXT
- Déduplication SHA256 — CNIL prévaut toujours dans les réponses

</details>

<details>
<summary><h2>📊 Observabilité & Configuration</h2></summary>

| Composant | Description |
|---|---|
| **Logs structurés** | `logs/app.jsonl` — requêtes, erreurs, timings |
| **Query Logger** | `logs/queries.jsonl` — historique complet |
| **Feedback** | `logs/feedback.jsonl` — 👍/👎 avec contexte |
| **Alertes** | Seuils configurables + notifications SMTP |
| **Dashboard** | Page Streamlit dédiée avec export |

Configuration centralisée dans `configs/config.yaml` — embeddings, RAG, observabilité, SMTP.

</details>

<details>
<summary><h2>📈 Évolution du système</h2></summary>

| Version | Composant | Impact |
|---|---|---|
| v1 | Semantic search + ChromaDB | Baseline 70% |
| v2 | Nomic embeddings + BM25 | +8% |
| v3 | Query Expansion LLM | +3% (recall) |
| v4 | Cross-Encoder Jina v2 | +3% (precision) |
| v5 | Rechunking intelligent | +1% |
| v6 | Intent Classification (7 intents) | Prompts ciblés |
| v6b | Agent LangGraph (8 nœuds) | +1.5% |
| v6d | Migration BGE-M3 (1024d) | Embeddings FR natifs |
| **v7** | **Tables content-based + tags RGPD** | **+0.5%** |
| **v8** | **Scoring brut (sans discrétisation)** | Scores réels |
| **v9** | **Query Decomposition** | Citations préservées |
| **v10** | **Génération structurée + prompts DPO** | **90.4%** |

```
Semantic seul                   70%
+ BM25 hybride                  78%  (+8%)
+ Query Expansion LLM           81%  (+3%)
+ Cross-Encoder Reranking       84%  (+3%)
+ Rechunking intelligent        85%  (+1%)
+ Intent Classification         86%
+ Agent LangGraph               89%  (+1.5%)
+ Tables content-based          89.7%
+ Single Gen + prompts DPO      90.4%
```

</details>

---

## 📄 Licence

Ce projet est un outil éducatif et de recherche. Les données CNIL utilisées sont publiques.

## 🙏 Remerciements

- **CNIL** — ressources publiques sur la protection des données
- **Ollama** — inférence locale simplifiée
- **Jina AI** — reranker multilingue open-source
- **Mistral AI** — Mistral-Nemo 12B
- **BAAI** — BGE-M3 (embeddings multilingues)
