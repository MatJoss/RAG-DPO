# Todo List — RAG-DPO System

**Dernière MAJ** : 2026-03-03

---

## 🏷️ Chantier Tags RGPD — Vocabulaire guidé + Semantic Matching

### Terminé ✅
- [x] `rgpd_topics.py` : 25 catégories RGPD normalisées + prompt guidé
- [x] `TopicMatcher` : boost sémantique via BGE-M3 (cache, cosine, bonus 0-0.15)
- [x] `parse_tags()` : split ,/; + normalisation lowercase + max 3
- [x] 3 chunkers : prompts alignés sur les 25 catégories
- [x] `intent_classifier.py` : prompt topics aligné
- [x] `reranker.py` : topic_boost intégré
- [x] `pipeline.py` : TopicMatcher initialisé, topics passés au reranker
- [x] `tag_all_chunks.py` : tqdm, --retry-failed, vocabulaire guidé
- [x] Intégré dans `rebuild_pipeline.py` (étape 6d) et `update_cnil.py` (étape 8b)
- [x] Test batch 10 : 14 tags uniques, quasi tous dans les 25 catégories ✅

### Détection tableaux content-based ✅
- [x] `_convert_table_rows()` : pipeline commun (zones → split → pipe-text → LLM Nemo)
- [x] `chunk_html()` : détection `<table>` → `_convert_html_table()` (39 HTML concernés)
- [x] `chunk_pdf()` : `_extract_pdf_tables()` via PyMuPDF `find_tables()` (264 PDF concernés, 54%)
- [x] `_chunk_word()` : détection `doc.tables` via python-docx DOM (1 DOCX concerné)
- [x] `_llm_convert_table()` / `_mechanical_fallback()` : préfixe "Feuille" supprimé (heading suffit)
- [x] `_split_long_text()` : renommé `sheet_name` → `context_name` (générique)
- [x] Tests validés : HTML 1 table → 7 chunks, PDF 1 table → 1 chunk (texte naturel)

### Terminé ✅ (v7 — mars 2026)
- [x] Rechunk complet (`rebuild_pipeline.py --fresh`) — 16 919 chunks (18h)
- [x] `tag_all_chunks.py --force` — 25 catégories guidées (2h)
- [x] Benchmark 42q × 3 runs — **92.1% ± 0.3%** (+2.9 pts vs v6b)
- [x] README rewrite (FR + EN) + git push

---

## 🗺️ Roadmap v2

| Prio | Chantier | Effort | Statut |
|------|----------|--------|--------|
| **1** | 🏷️ Metadata scraper (Last-Modified, hash, dates) | 1-2j | ✅ Done |
| **2** | 🧠 Migration BGE-M3 (remplace nomic + BM25 séparé) | 2-3j | ✅ Done |
| **3** | 📂 Pipeline ingestion entreprise (incrémental) | 1 sem | 🔨 En cours |
| **4** | 📊 Observabilité (logs, feedback, dashboard) | 2-3j | ✅ Done |
| **5** | 🐳 Docker (on package le produit fini) | 2-3j | 🔲 |
| **5b** | 🔄 Init & Update CNIL (routine mensuelle) | 1j | ✅ Done |
| **6** | 🧠 Intent-Aware Pipeline (assistant structuré) | 2-3j | ✅ Done |
| **6b** | 🤖 Pipeline Agent LangGraph | 1-2j | ✅ Done |
| **6c** | 📊 Eval v3 — Semantic Similarity + Conciseness intent-aware | 0.5j | ✅ Done |
| **6c'** | 🔧 Agent tools locaux (articles RGPD, délais, completeness) | 0.5j | ✅ Done |
| **6d** | 🔧 Tuning prompts intent (régressions q16/q02/q17) | 0.5j | 🔲 |
| **6e** | 🧠 Multi-turn memory agent | 1-2j | 🔲 |
| **7** | 🕸️ Graph RAG (LightRAG) | 2 sem | 🔲 v2 |

---

## 🧠 Prio 6 — Intent-Aware Pipeline

### Diagnostic
Le RAG actuel est un **moteur documentaire** : il assemble des chunks sans raisonnement transversal.
Quand la question est méthodologique ou organisationnelle (ex: "Comment mener une AIPD ?"),
le modèle ne peut que réciter les sources sans structurer de démarche opérationnelle.

Le system prompt actuel **interdit** au LLM d'utiliser ses connaissances RGPD générales :
> "Tu réponds UNIQUEMENT à partir des sources fournies"

C'est correct pour les questions factuelles, mais destructeur pour les questions méthodologiques.

### Contrainte hardware
- RTX 4070 Ti : 12 GB VRAM
- Budget actuel : Mistral-Nemo ~8 GB + BGE-M3 ~1 GB + Jina (CPU) = ~9 GB
- **Pas de place pour un 2ème modèle GPU**
- Solution : dual-pass sur le **même** Mistral-Nemo (0 VRAM supplémentaire)

### Architecture cible

```
Question DPO
    ↓
┌─────────────────────────────────────────────┐
│ PHASE 0 — Intent Classification (Nemo)      │
│ 1 appel LLM, prompt court → JSON            │
│ ~0.3-0.5s, ~100 tokens                      │
│                                             │
│ Sortie :                                    │
│   intent: factuel | methodologique |        │
│           organisationnel | comparaison |   │
│           cas_pratique | liste_exhaustive   │
│   scope_international: true/false           │
│   needs_methodology: true/false             │
│   expected_structure: steps|list|comparison │
│   topics: [aipd, transfert, violation...]   │
└──────────────┬──────────────────────────────┘
               ↓
       [Phases 1-6 existantes]
       Seul changement : le system prompt
       est sélectionné selon l'intent
```

### Plan d'implémentation

#### Étape 1 : Intent Classifier (`src/rag/intent_classifier.py`) — NOUVEAU
- [ ] Créer le module `IntentClassifier`
- [ ] Prompt court → JSON structuré
- [ ] 6 classes d'intent
- [ ] Détection dimension internationale, besoin méthodologique
- [ ] Topics détectés
- [ ] Fallback gracieux : parsing JSON échoué → intent=factuel
- [ ] Temps cible : <1s

#### Étape 2 : Prompts adaptatifs (`src/rag/context_builder.py`) — MODIFIÉ
- [ ] Garder SYSTEM_PROMPT actuel comme `SYSTEM_PROMPT_FACTUEL`
- [ ] Ajouter prompts méthodologique, organisationnel, cas_pratique, comparaison, liste
- [ ] Adapter USER_PROMPT_TEMPLATE avec instruction anti-dérive conditionnelle
- [ ] `build_context()` reçoit l'intent et sélectionne le bon prompt
- [ ] Distinction [Source X] vs [Connaissance RGPD] dans prompts méthodologiques

#### Étape 3 : Intégration pipeline (`src/rag/pipeline.py`) — MODIFIÉ
- [ ] IntentClassifier dans `RAGPipeline.__init__()`
- [ ] Appeler classify() en phase 0 AVANT retrieval
- [ ] Passer intent au context_builder
- [ ] Logger l'intent (structured logging)
- [ ] Ajouter intent dans RAGResponse
- [ ] create_pipeline() instancie l'IntentClassifier

#### Étape 4 : Validation
- [ ] Test unitaire : 10 questions → intent correct
- [ ] Test e2e : question AIPD → réponse structurée
- [ ] Benchmark comparatif avant/après
- [ ] Vérifier latence <1s ajoutée
- [ ] Vérifier pas de régression factuelles

### Coût
- +1 appel LLM/question (~0.3-0.5s, ~100 tokens)
- 0 VRAM supplémentaire
- 0 dépendance nouvelle

---

## 🚀 Migration BGE-M3 (TERMINÉE)

### Objectif
Remplacer nomic-embed-text (anglais seul, 137M) par BGE-M3 (multilingue, 568M, dense+sparse intégré).
Gains attendus : meilleur retrieval FR, simplification archi (BM25 séparé → sparse intégré).

### Sous-tâches
- [x] Installer FlagEmbedding / sentence-transformers pour BGE-M3
- [x] Créer `src/utils/embedding_provider.py` — provider BGE-M3 dédié (FP16, GPU, lazy load)
- [x] Adapter `create_chromadb_index.py` pour embeddings BGE-M3 (1024 dims au lieu de 768)
- [x] Supprimer troncature legacy nomic ([:2500] coupait 49.5% des chunks)
- [x] Adapter `retriever.py` — méthode `_embed()` avec fallback
- [x] Adapter `pipeline.py`, `app.py`, `test_rag.py`, `eval/run_eval.py`
- [x] Mettre à jour `config.yaml` — section embeddings BGE-M3
- [x] Re-indexer 14,388 chunks (100% succès, 1024 dims, texte complet)
- [x] Adapter le VRAM budget (Mistral-Nemo ~7-8GB + BGE-M3 ~1.07GB = ~9GB / 12GB)
- [x] Benchmark comparatif nomic vs BGE-M3 sur les 18 questions (BGE-M3 91.8% vs nomic 90.7%)
- [ ] Évaluer si le sparse intégré BGE-M3 peut remplacer `bm25_index.py` (v2)

---

## 📂 Pipeline Données Entreprise

### Objectif
Permettre aux DPO d'alimenter le RAG avec **leurs propres documents internes** (politiques internes, registres de traitement, contrats, PIA, etc.) tout en conservant la base CNIL comme référentiel autoritaire.

### Fonctionnalités à implémenter

#### 1. Pipeline d'ingestion entreprise
- [x] Script d'import fichiers entreprise (PDF, DOCX, XLSX, HTML, TXT)
- [x] Extraction texte + chunking (réutilise `StructuralChunker` de `process_and_chunk.py`)
- [x] Métadonnées source : `source: "enterprise"` vs `source: "cnil"`
- [ ] Classification automatique (nature/index) adaptée au contexte entreprise
- [x] Support batch (dossier) et unitaire (fichier)
- [x] CLI complète : `--purge`, `--list`, `--stats`, `--recursive`
- [x] Déduplication par hash SHA256 (relancer = pas de doublons)

#### 2. Stratégie VectorDB — ✅ Option A (Append) + pré-filtrage natif
- [x] Chunks entreprise dans la même collection ChromaDB `rag_dpo_chunks`
- [x] Filtrage par `source: "ENTREPRISE"` pour purge/listing
- [x] Purge entreprise sans affecter CNIL (testé)
- [x] Tags booléens par chunk (`tag_registre: true`, `tag_pia: true`) → filtrable nativement par ChromaDB
- [x] Pré-filtrage ChromaDB `$or` : CNIL toujours inclus + tags entreprise sélectionnés
- [x] Registre auto `configs/enterprise_tags.json` (MAJ à chaque ingestion/purge)
- [x] UI Streamlit : multiselect des tags avec labels lisibles
- [x] Tags par défaut (ex: politique_interne) pré-sélectionnés dans l'UI
- [x] Distinction [CNIL] / [Interne] dans le context builder et les cartes sources

#### 3. Système de pondération (importance weights)
- [ ] Poids par source : CNIL > entreprise (CNIL = référentiel, entreprise = contexte)
- [ ] Poids par nature de document : GUIDE > FAQ > ACTUALITE
- [ ] Poids configurable dans `config.yaml`
- [ ] Intégration dans le scoring RRF du retriever
- [ ] Boosting contextuel : si la question porte sur l'interne → boost entreprise

#### 4. Gestion multi-tenant (optionnel, moyen terme)
- [ ] Isolation par entreprise (1 VectorDB par client)
- [ ] Config par tenant dans `configs/`
- [ ] Interface Streamlit : sélecteur d'entreprise

### Architecture implémentée
```
Question DPO
    → [UI] Sélection tags entreprise (multiselect sidebar)
    → build_enterprise_where_filter() → $or[CNIL, tag_X, tag_Y]
    → Query Expansion (x3)
    → Hybrid Retrieval (ChromaDB + BM25) avec pré-filtre natif
    → Jina Reranker (40 → 10 chunks)
    → Dual Generation + Grounding
    → Réponse avec sources [CNIL] et [Interne]
```

### Contraintes
- 100% local (pas de cloud)
- CNIL prévaut TOUJOURS sur les docs entreprise
- Traçabilité : chaque réponse indique si la source est CNIL ou interne
- Purge entreprise sans affecter CNIL

---

## 🔧 Améliorations en cours

### Retrieval (questions encore faibles)
- [ ] q09 (60%) : retrieval "CV 2 ans dernier contact" — chunk existe mais pas retrouvé
- [ ] q10 (73%) : erreur factuelle vidéosurveillance — intérêt légitime possible avec mise en balance
- [ ] q06 (80%) : réponse superficielle liste noire/blanche AIPD

### Interface Streamlit
- [ ] Test end-to-end complet
- [ ] Historique conversation (session state)
- [ ] Export conversations (PDF/Markdown)

### Qualité (optionnel)
- [ ] Fine-tuning paramètres hybrides (α BM25, RRF k)
- [ ] Enrichissement synonymes juridiques RGPD
- [ ] Dataset évaluation élargi (50-100 questions)
- [ ] Fine-tuning embeddings vocabulaire RGPD

---

## ✅ Historique — Réalisations

### v1.0 — Pipeline complet (2026-02-12)
- [x] Scraping CNIL : 8236 HTML + 1026 PDFs + 43 docs + 221 images
- [x] Classification hybride LLM : 2568 keep (31.2%)
- [x] Déduplication corpus : 3702 → 1847 docs (-50.1%)
- [x] Chunking sémantique : 14,388 chunks (overlap 50w, heading propagé)
- [x] Indexation ChromaDB (nomic-embed-text, 768 dims)
- [x] Résumés structurés LLM (1829 docs)
- [x] RAG hybride : BM25 + Semantic + RRF + Query Expansion + Jina Reranker
- [x] Dual Generation (self-consistency via context order)
- [x] Grounding Validation (citations sources)
- [x] Interface Streamlit
- [x] Évaluation 18 questions : **93% global, 84% correctness, 100% faithfulness**

### v1.2 — Observabilité production-ready (2026-03-01)
- [x] QueryLogger — log JSONL queries + feedback, rotation, stats analytiques
- [x] Structured Logger — JSON structured logging (`logs/app.jsonl`) pour tous les modules
- [x] Alerter — seuils configurables (error rate, temps, satisfaction, citations)
- [x] Email SMTP — alertes email configurables dans `config.yaml` (désactivé par défaut)
- [x] Streamlit multipage — page accueil + Chat + Dashboard dédié
- [x] Dashboard — métriques, queries récentes, feedback, alertes, export JSON
- [x] Feedback 👍/👎 — boutons inline dans le chat avec log JSONL
- [x] Config `observability` dans `config.yaml` — logging + alerting + SMTP
- [x] Tests complets (`tasks/_test_observability.py`) — 3 composants validés

### v1.1 — Metadata enrichies (2026-03-01)
- [x] Scraper v2 : capture Last-Modified HTTP, content_hash SHA256, dates page CNIL
- [x] Backfill 16 909 metadata existantes (content_hash + published_at + schéma unifié)
- [x] Mode `--update` : scraping incrémental avec requêtes conditionnelles (If-Modified-Since → 304)
- [x] Mode `--backfill` : enrichissement metadata sans requêtes HTTP
- [x] Schéma metadata unifié v2 (HTML et PDF convergent)

### Évolution scores évaluation
| Version | Global | Correctness | Temps/q |
|---------|--------|-------------|---------|
| v1 Baseline | 86% | 65% | 6.3s |
| v2 Query Expansion | 89% | 73% | 13.2s |
| v3 Eval Fixes | 92% | 80% | 8.2s |
| v4 Jina Reranker | 92% | 83% | 9.5s |
| v5 Rechunking | 93% | 84% | 31.9s |
| v6 BM25 Boost | 92% | 81% | 14.0s |
| **v7 Dual-Gen (BEST)** | **93%** | **84%** | **17.3s** |
| v7b Reverse test | 91% | 81% | 14.8s |
| **v8 BGE-M3 (scoring v2)** | **91.8%** | **86.3%** | **24.8s** |
| v8 nomic (scoring v2) | 90.7% | 83.9% | 28.9s |

---

## 📝 Règles métier DPO

1. **CNIL prévaut TOUJOURS** sur les docs entreprise
2. **Jamais inventer** — si pas de source, dire "je ne sais pas"
3. **Citations traçables obligatoires** (URL source)
4. **100% local** (pas de fuite données)
