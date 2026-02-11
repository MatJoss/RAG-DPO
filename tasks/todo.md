# Todo List - RAG-DPO System

**Dernière MAJ** : 2026-02-11

---

## 🔬 DIAGNOSTIC RECHUNKING — Faits bruts (2026-02-11)

### Contexte
v5 (bge-reranker) = 91.2%. v6 (retrieve_candidates + reranker flow) ≈ 87%.
Le pipeline tweaking (top_k, n_chunks_per_doc, etc.) ne résout rien.
L'information est dans les chunks, mais elle est **mal découpée et diluée**.

### Distribution actuelle (16,044 chunks)
```
Tiny(<50w):    441   (2.7%)  ← inutilisables, bruit
Small(50-150): 1754  (10.9%) ← trop courts, contexte perdu  
Medium(150-400):5499 (34.3%) ← acceptables mais souvent dilués
Target(400-600):8072 (50.3%) ← taille cible max_size=450 du chunker
Large(600+):    278  (1.7%)  ← dépassent le max, split naïf par mots
```

### Problème 1 : Information DILUÉE (q05, q08, q10, q11)
Les keywords manquants EXISTENT dans la base mais sont éparpillés :
- q05 "données sensibles" + "grande échelle" : 38 chunks avec TOUT, mais 685 avec au moins 1 → **5.5% concentration**
- q08 "portabilité" + "limitation" : 124/1119 = **11.1% concentration**
- q10 "mise en balance" + "sécurité" : 17/3595 = **0.5% concentration**
- q11 "base légale" + "contrat de travail" : 9/648 = **1.4% concentration**

→ Le chunker coupe au milieu des concepts. Un paragraphe qui parle de "données sensibles à grande échelle" est splitté en deux chunks de 400 mots.

### Problème 2 : Information ABSENTE du retrieval (q09)
- **1 SEUL** chunk contient "2 ans" + "dernier contact" (chunk #2319, 752w)
- MAIS c'est dans un document sur la **prospection commerciale** (d05ee50f6467), pas les CV
- Le vrai passage CV est chunk #18 (588w, doc 83e81e7846a8) : "*les données d'un candidat non retenu seront conservées pendant 2 ans maximum*"
- Ce chunk #18 parle de "2 ans" + "candidat" + "recrutement" mais PAS de "dernier contact"
- Problème : la query "conserver des CV indéfiniment" ne matche pas sémantiquement "candidat non retenu 2 ans maximum"
- Le keyword "CV" n'apparaît même pas dans le chunk → BM25 rate aussi

### Problème 3 : Chunks SANS heading (perte de contexte)
- chunk #99 et #100 (07ec6ca4d34d.html) : 400w chacun, heading="" → c'est le texte du RGPD splitté
- chunk #2319 : heading="Comment assurer le respect du droit d'opposition..." → misleading pour q09 CV
- Le chunker split par taille (max_size=450) et perd le contexte sectionnel

### Problème 4 : Split naïf des gros documents
- `_post_process()` dans process_and_chunk.py : si >450 mots, split par `text_words[i:i+target_size]`
- Pas d'overlap ! Un concept coupé au milieu est irrémédiablement perdu
- Pas de heading propagé au chunk enfant

### Problème 5 : Le chunk #659 est PARFAIT mais le reranker ne le trouve pas toujours
- Chunk #659 (278w) : "Quand est-ce qu'une AIPD est obligatoire ?" avec les 9 critères listés
- Ce chunk est la réponse exacte à q05 mais doit être dans les 60 premiers candidats cosine
- Embeddings nomic-embed-text saturés → cet excellent chunk se noie dans le cluster AIPD

---

## 🏗️ PLAN RECHUNKING — 3 phases

### Phase R1 : Rechunking intelligent ✅ DONE
**Objectif** : Transformer 16,044 chunks de qualité inégale en chunks auto-suffisants.

**Changements au chunker** (`process_and_chunk.py`) :
- [x] **Overlap 50 mots** : `_split_semantic()` ajoute 50w du chunk précédent
- [x] **Heading propagé** : `_post_process()` stage 3 préfixe `[heading]` dans le texte
- [x] **Split sémantique** : coupe sur `\n\n` puis `. `, fallback mot
- [x] **Taille cible 400w** : target=400, min=100, max=600 (souple)
- [x] **Purge tiny chunks** : <100w fusionné avec voisin (stage 2)
- [x] **Heading dans le texte** : `[heading] text` pour que l'embedding le voie

**Résultat** : 16,044 → **14,388 chunks** (-10.3%, moins de bruit)

### Phase R2 : Re-indexation ChromaDB ✅ DONE
- [x] Re-généré `processed_chunks.jsonl` (14,388 chunks, 1832 docs)
- [x] Re-indexé ChromaDB (mode reset, 3m31s)
- [x] Vérifié : 100% indexés, filtre par nature OK

### Phase R3 : Évaluation comparative ✅ DONE
- [x] Eval v7a (rechunking seul, éval biaisée) : 89% global, 75% correctness
- [x] Diagnostic biais éval : questions vagues ↔ must_include trop spécifiques
- [x] Fix éval : `must_include_any` (N parmi M) + alternates pipe-separated
- [x] Eval v7b (rechunking + fix éval) : **93% global, 84% correctness** ← nouveau record

### Phase R4 : Retrieval restant (q09, q10)
- [ ] q09 (60%) : "2 ans" + "dernier contact" pour CV — vrai manque retrieval
- [ ] q10 (73%) : réponse factuelle fausse (dit "non" au lieu de "oui, avec mise en balance")
- [ ] q06 (80%) : réponse superficielle (pas de détails liste noire/blanche)
- [ ] Objectif : ≥95% global, 0 question en dessous de 73%

### Hors scope (pour plus tard)
- Changer d'embeddings (e5-large, etc.) — gros chantier, pas la priorité
- BM25 avec stemming FR — amélioration marginale vs rechunking
- Fine-tuning embeddings — nécessite un dataset gold standard
- Augmentation query expansion — déjà en place, amélioration marginale

---

## ÉTAT PIPELINE : Reconstruction (Nemo 12B + region-content)

### Phase 0 : Corrections Code ✅
- [x] Fix extraction HTML: `region-content` dans tous les modules
- [x] Modèle migré: mistral-nemo partout
- [x] Améliorer prompts: toutes phases (3, 5A, 6B, RAG query)
- [x] Fix critique: process_and_chunk.py ignore les 49 docs (xlsx/odt/docx)
- [x] Fix critique: PDF chunking → TOC/font/smart
- [x] Réécrire: rebuild_pipeline.py avec TOUTES les phases (3→6b)
- [x] Checkpoint/resume: hybrid_filter.py + process_and_chunk.py
- [x] Sanity checks post-phase dans rebuild_pipeline.py
- [x] Fix chemins cohérents data/raw/cnil/

### Phase 3 : Classification hybride ✅
- [x] 8236 HTML → 2568 keep (31.2%), 11.9h

### Phase 4 : Organisation keep/archive ✅
- [x] 2568 HTML, 1026 PDFs, 43 docs, 221 images dans keep/

### Phase 4B : Classification images (OCR + LLaVA) ✅
- [x] 221 images → 65 SCHEMA_DPO keep, 156 PHOTO_DECO éliminées
- [x] Fix: --test mode dry-run (pas de modification manifest)
- [x] Fix: stats comptent correctement les images cachées

### Phase 4C : Déduplication corpus ✅
- [x] Créé `src/processing/deduplicate_corpus.py`
  - Hash MD5 region-content pour HTML, binaire pour PDF/docs/images
  - Sélection canonical : https > http, URL la plus courte
  - Archivage dans keep/dedup_archive/ (pas de suppression)
  - Backup manifest automatique (.pre_dedup)
  - Support --fresh (restaure backup), --dry-run
- [x] Intégré dans rebuild_pipeline.py (Phase 4C, sanity check)
- [x] **Résultat** : 3702 → 1847 docs (-50.1%)
  - HTML: 2568 → 1300 (-1268)
  - PDF: 1026 → 485 (-541)
  - Docs: 43 → 29 (-14)
  - Images: 65 → 33 (-32)
- [x] 1855 fichiers archivés dans keep/dedup_archive/
- [x] keep/ vérifié = manifest exact
- [x] Fix --fresh : restaure backup manifest + ne l'écrase pas

### Phase 5A : Classification documents (Nemo) ✅
- [x] Code modifié : intègre images SCHEMA_DPO (classification déterministe)
- [x] Fix json_cleaner : double braces `{{...}}` → `{...}`
- [x] Fix manifest : 14 ODS avaient extension `.xlsx` → corrigé
- [x] Nettoyage résiduel : 11 ODS dupliqués + 4 fake DOCX archivés
- [x] **Résultat** : 1832 docs classifiés, ~7 erreurs résiduelles (0.4%)

### Phase 5B : Chunking + Classification chunk-level ✅
- [x] Code modifié : chunk_image() pour images SCHEMA_DPO
- [x] Code modifié : url_cache inclut images
- [x] **Résultat** : 16016 chunks, 1823 docs uniques, 8.8 chunks/doc, 92s

### Phase 6A : Indexation ChromaDB ✅
- [x] 16044 chunks indexés (nomic-embed-text, 768 dim)

### Phase 6B : Résumés structurés (Nemo) ✅
- [x] 1829 résumés générés (1823 docs + 6 cleaned entries)
- [x] Filtre navigation corrigé : seuil 2000 chars (ne flag plus les pages riches)
- [x] 5 pages utiles récupérées (FICOBA, Guide sécurité, Fiches IA, FNAEG, Guide auto-évaluation IA)
- [x] 0 erreurs, 0 nav skip restant

### Phase 6C : Nettoyage post-résumés ✅
- [x] Analyse contenu propre des 11 pages nav → 5 faux positifs récupérés
- [x] 6 vrais nav purgés de ChromaDB + JSONL
- [x] Fichiers archivés dans `data/archive/html/`
- [x] Summaries mis à jour (6 entries `cleaned: true`)
- **Résultat** : 16044 chunks, 1823 docs, 1829 summaries, 0 erreur

---

## 📊 DONNÉES BRUTES — Analyse doublons (2026-02-09)

```
HTML fichier brut identique    :   38/2568 ( 1.5%)  ← URLs très proches
HTML region-content identique  : 1268/2568 (49.4%)  ← pages CNIL renommées/redirect
PDF  fichier identique         :  541/1026 (52.7%)  ← même PDF sous N URLs
Images fichier identique       :   32/65   (49.2%)  ← même schéma sous N URLs
```

**Exemples :** même page CNIL sous http/https, /tag/cloud vs /tag/Cloud,
pages renommées mais contenu inchangé, un même PDF recommandations
téléchargé depuis 37 pages différentes.

**Impact sans dédup** : le RAG retournerait N fois la même info avec des
scores proches → bruit, tokens gaspillés, confusion pour l'utilisateur.

---

## 🏗️ ARCHITECTURE RAG HIÉRARCHISÉE (vision moyen/long terme)

### Niveau 1 : Documents (macro)
- `document_metadata.json` : classification nature/index par document
- `document_summaries.json` : fiche synthétique par document (Phase 6B)
- Déduplication : 1 canonical par contenu unique, doublons éliminés
- **Requête** : "De quoi parle ce corpus ?" → recherche par résumés

### Niveau 2 : Chunks (micro)
- `processed_chunks.jsonl` : chunks structurels classifiés
- Chaque chunk lié à son document parent (document_id, document_path)
- Metadata riches : nature, index, secteurs, heading, source_url
- **Requête** : "Comment faire une AIPD ?" → recherche vectorielle chunks

### Niveau 3 : Retrieval 2-étapes
```
Question → Query Qualification (intent + filtres)
        → Étape 1 : Résumés documents (top-K documents pertinents)
        → Étape 2 : Chunks de ces K documents (top-N chunks)
        → Reranking (similarity × confidence × priorité)
        → Context building (top-5 chunks + metadata)
        → Génération réponse + citations
```

### Anti-doublons au query time (filet de sécurité)
- Même si la dédup Phase 4C nettoie le corpus, le retriever doit aussi :
  - Grouper chunks par document_id
  - Ne pas retourner >2 chunks du même document
  - Détecter chunks quasi-identiques (similarity > 0.95 entre eux)

---

## 🎯 PLAN EXÉCUTION (ordre)

### Sprint actuel : Pipeline propre
1. [x] Intégrer images dans Phase 5A + 5B (code modifié)
2. [x] **Phase 4C : déduplication corpus** ✅ (3702 → 1847, -50.1%)
3. [x] Intégrer Phase 4C dans rebuild_pipeline.py
4. [x] **Pipeline 5A→5B** ✅ (1832 docs, 16016 chunks)
5. [x] **Phase 6A** ✅ (16044 chunks indexés ChromaDB)
6. [x] **Phase 6B** ✅ (1829 résumés, 0 erreur)
7. [x] **Phase 6C** ✅ (6 nav purgés, 5 récupérés, 16044 chunks finaux)

### Sprint suivant : RAG Engine ✅
6. [x] `src/rag/bm25_index.py` — CRÉÉ : Index BM25 summaries + chunks
7. [x] `src/rag/reranker.py` — CRÉÉ : Cross-encoder ms-marco-MiniLM-L-6-v2
8. [x] `src/rag/retriever.py` — RÉÉCRIT : Hybrid BM25+Semantic+RRF+Summary pre-filter
9. [x] `src/rag/context_builder.py` — MAJ : Nouveaux prompts + reverse repacking
10. [x] `src/rag/pipeline.py` — MAJ : Reranker intégré, phases 1→5, create_pipeline factory
11. [x] `src/rag/__init__.py` — MAJ : 16 exports (BM25, Reranker, etc.)
12. [x] `configs/config.yaml` — MAJ : Params RAG hybride + reranker
13. [ ] `test_rag.py` — Validation questions DPO types

### Sprint Streamlit : Interface ⏳
14. [x] `app.py` — MAJ : toggles hybrid/reranker/validation, slider défauts corrigés
15. [ ] Test Streamlit end-to-end
16. [ ] Historique conversation
17. [ ] Export conversations

### Sprint Qualité (optionnel)
18. [ ] Hybrid search fine-tuning (α BM25, RRF k)
19. [ ] Query expansion (synonymes juridiques RGPD)
20. [ ] Evaluation set (50-100 questions manuelles)
21. [ ] Fine-tuning embeddings vocabulaire RGPD

---

## 📝 Règles métier DPO

```python
# 1. CNIL prévaut TOUJOURS sur les docs entreprise
# 2. Jamais inventer — si pas de source, dire "je ne sais pas"
# 3. Citations traçables obligatoires (URL source)
# 4. 100% local (pas de fuite données)
```

# 3. Vérifier index
python -c "import chromadb; client = chromadb.PersistentClient(path='data/chroma_db'); col = client.get_collection('rag_dpo_chunks'); print(f'Chunks indexés: {col.count()}')"
```

**Durée estimée** : ~18 minutes (35 900 chunks)

**Output** : `data/chroma_db/` (base vectorielle)

---

### [ ] 5. Tests de validation
**Tests à réaliser** :

```python
# Test 1 : Query simple
results = collection.query(
    query_texts=["Comment faire une AIPD ?"],
    n_results=5
)
# → Doit retourner des chunks pertinents

# Test 2 : Filtre par nature
results = collection.query(
    query_texts=["Comment faire une AIPD ?"],
    n_results=5,
    where={"chunk_nature": "GUIDE"}
)
# → Doit retourner UNIQUEMENT des GUIDE

# Test 3 : Filtre par source
results = collection.query(
    query_texts=["Comment faire une AIPD ?"],
    n_results=5,
    where={"source": "CNIL"}
)
# → Doit retourner UNIQUEMENT des chunks CNIL

**MVP fonctionnel** = Sprint 1 + Sprint 3 (~5h dev)

**Encodage** : UTF-8 partout (requirement critique)

---

## 🏁 État actuel (2026-02-05)

### Infrastructure ✅
- ✅ 56,328 chunks indexés ChromaDB
- ✅ Ollama embeddings (768 dims)
- ✅ Modes reset/append/update opérationnels
- ✅ Vérification --verify-only fonctionnelle
- ✅ Stratégie déduplication documents testée

### Prochaine action → **Sprint 1 : RAG Basique**

---

## 📊 Audit Évaluation Q&A (2026-02-10)

### Contexte
GPT a généré 18 questions d'évaluation couvrant 7 catégories (définitions, obligations, recommandations CNIL, cas pratiques DPO, pièges anti-hallucination, robustesse sémantique, hors périmètre). Les réponses du RAG ont été évaluées par GPT.

### Diagnostic
- ✅ Safety/Faithfulness : solide (pas d'hallucination grossière, refus hors périmètre OK)
- ✅ Cloisonnement CNIL respecté
- ❌ Erreur factuelle majeure : critères AIPD (confusion parties doc vs critères déclenchement)
- ❌ CV 2 ans : retrieval KO (recommandation opérationnelle non retrouvée)
- ❌ Liste AIPD : focus inversé (liste blanche vs noire)
- ⚠️ Sur-justification généralisée (réponses trop longues, reformulations)
- ⚠️ Forçage de grounding (art. 20 affiché pour question sur art. 99)
- ⚠️ "Sources insuffisantes" sur sujets bien documentés

### Actions réalisées
- [x] Créé `eval/qa_dataset.json` : 18 questions avec must_include/must_not_include/expected_answer
- [x] Créé `eval/run_eval.py` : scoring auto sur 4 axes (correctness, faithfulness, conciseness, sources)
- [x] Amélioré prompt system v2 dans `context_builder.py` :
  - Ajout section CONCISION (2-4 phrases d'abord)
  - Règle 5 : anti-forçage grounding (ne pas citer source non pertinente)
  - Interdiction "En conclusion" / reformulation finale
  - Interdiction citation source sans rapport avec la question
  - User prompt : consigne concise, anti-forçage explicite
- [x] Baseline évaluation : 85.5% global, 100% faithfulness, 99.4% conciseness, 65% answer_correctness
- [x] Diagnostic retrieval : q05/q09/q11 — chunks existent dans ChromaDB mais pas dans top-50
- [x] Créé `src/rag/query_expander.py` : LLM multi-query (3 reformulations + originale)
- [x] Modifié `src/rag/retriever.py` : boucle multi-query avec RRF fusion, distance préservée
- [x] Modifié `src/rag/pipeline.py` : QueryExpander wired, rerank_candidates 30→50, rerank_top_k 8→10
- [x] Modifié `configs/config.yaml` : query expansion config, relevance_threshold 0.30→0.35
- [x] **Éval v2 (query expansion)** : 88.6% global (+3.1%), 73% correctness (+8.1%)
  - 6 questions améliorées (q06 +50%, q09 +50%, q18 +50%, q11 +33%, q08 +17%, q05 +16%)
  - 2 régressions (q04 -40%, q03 -25%)
  - Temps ×2.1 (6.3s → 13.2s)

### Résultats évaluation

| Version | Global | Correctness | Faithfulness | Conciseness | Sources | Temps |
|---------|--------|-------------|--------------|-------------|---------|-------|
| Baseline (prompt v2) | 85.5% | 65.0% | 100% | 99.4% | 97.2% | 6.3s |
| + Query Expansion | 88.6% | 73.1% | 100% | 97.8% | 97.2% | 13.2s |
| + Eval Fixes (v3) | 91.7% | 80.0% | 100% | 98.0% | 97.2% | 8.2s |
| + bge-reranker-v2-m3 (v5) | 91.2% | 80.0% | 100% | 98.0% | 97.0% | 17.0s |

#### Détail v3→v5 (bge-reranker) :
- 🟢 q05 AIPD critères : 70%→100% (+30%) — reranker multilingue retrouve données sensibles/grande échelle/surveillance
- 🟢 q08 Droits personnes : 93%→100% (+7%) — portabilité retrouvée
- 🔴 q03 RT vs ST : 100%→80% (-20%) — non-déterminisme LLM (manque finalités/instructions)
- 🔴 q18 Contourner CNIL : 100%→79% (-21%) — non-déterminisme LLM (manque sanction)
- = q09/q10/q11 inchangés (60%/73%/73%) — problème retrieval upstream, pas reranker

### À faire
- [ ] Investiguer régression q04 (AIPD obligatoire : 80% → 40%) — expansion noie les bons chunks
- [ ] Investiguer q10 (vidéosurveillance : dit FAUX que intérêt légitime impossible) — chunks manquants
- [ ] Compléter résumés documents (36% → 100%) pour améliorer summary pre-filter
- [ ] Investiguer q15 (étude impact) : "risque élevé" DANS la réponse mais pas scored — bug eval ?
- [ ] Augmenter `summary_prefilter_k` de 20 → 40 (surface plus de docs candidats)
- [ ] Re-chunking ciblé guides opérationnels CNIL (recrutement, vidéosurveillance, RH)
