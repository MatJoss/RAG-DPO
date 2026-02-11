# Todo List — RAG-DPO System

**Dernière MAJ** : 2026-02-12

---

## 🚀 PROCHAINE ÉTAPE : Pipeline Données Entreprise

### Objectif
Permettre aux DPO d'alimenter le RAG avec **leurs propres documents internes** (politiques internes, registres de traitement, contrats, PIA, etc.) tout en conservant la base CNIL comme référentiel autoritaire.

### Fonctionnalités à implémenter

#### 1. Pipeline d'ingestion entreprise
- [ ] Script d'import fichiers entreprise (PDF, DOCX, XLSX, HTML)
- [ ] Extraction texte + chunking (réutiliser `process_and_chunk.py`)
- [ ] Métadonnées source : `source: "enterprise"` vs `source: "cnil"`
- [ ] Classification automatique (nature/index) adaptée au contexte entreprise
- [ ] Support batch (dossier) et unitaire (fichier)

#### 2. Stratégie VectorDB
- [ ] **Option A — Append** : ajouter les chunks entreprise dans le même ChromaDB
  - Avantage : recherche unifiée, simple
  - Risque : contamination si mauvaise pondération
- [ ] **Option B — VectorDB séparé** : ChromaDB dédié entreprise
  - Avantage : isolation, purge facile
  - Nécessite : fusion au query-time (multi-collection retrieval)
- [ ] **Option C — Hybride** : VectorDB séparé + fusion pondérée au retrieval
  - Retrieval parallèle CNIL + entreprise
  - RRF fusion avec poids différenciés
  - **← Recommandé**

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

### Architecture cible
```
Question DPO
    → Query Expansion (x3)
    → Retrieval CNIL (ChromaDB CNIL, BM25 CNIL)
    → Retrieval Entreprise (ChromaDB Entreprise, BM25 Entreprise)
    → RRF Fusion pondérée (w_cnil=1.0, w_enterprise=0.7)
    → Jina Reranker (top-20 → top-8)
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

---

## 📝 Règles métier DPO

1. **CNIL prévaut TOUJOURS** sur les docs entreprise
2. **Jamais inventer** — si pas de source, dire "je ne sais pas"
3. **Citations traçables obligatoires** (URL source)
4. **100% local** (pas de fuite données)
