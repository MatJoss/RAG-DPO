# Lessons Learned

## Python f-string : PAS de backslash dans les expressions {}
- **Erreur récurrente** : `f'{d["key"]}'` ou `f'{v["error"][:100]}'` → SyntaxError
- **Règle** : En Python < 3.12, les f-strings **interdisent** les backslashes (quotes échappées, `\n`, etc.) à l'intérieur des `{}`
- **Fix** : TOUJOURS extraire dans une variable AVANT le f-string :
  ```python
  # ❌ INTERDIT
  print(f'{d["error"][:100]}')
  
  # ✅ CORRECT
  err = d["error"][:100]
  print(f'{err}')
  ```
- **Aussi valable pour** : `str(v["key"])`, `path.replace("\\", "/")` dans un f-string
- **Contexte** : Python 3.11.9 (ce projet). Python 3.12+ lève cette restriction mais on n'y est pas.

## Vérifier les noms de clés JSONL avant de les utiliser
- Le fichier `processed_chunks.jsonl` utilise `document_path`, pas `doc_path`
- Toujours inspecter un sample avant d'écrire du code d'analyse

## Scripts diagnostiques : utiliser des fichiers .py, pas des one-liners -c
- Les one-liners PowerShell avec `python -c` sont fragiles (échappement quotes, backslashes, longueur)
- Préférer créer un petit script dans `tasks/` puis l'exécuter

## Filtre navigation : ne jamais utiliser "long_lines" comme proxy
- Le chunker produit des blocs de texte monoligne → `long_lines < 3` est TOUJOURS vrai pour des docs à 1-2 chunks
- Utiliser `len(content)` comme indicateur de richesse, pas le nombre de lignes longues
- Seuil trouvé : > 2000 chars = jamais navigation, même avec mots "Consulter"/"En savoir plus"
- Les mots nav apparaissent naturellement dans du contenu CNIL légitime (FICOBA, guides, etc.)
- **Règle** : toujours tester le filtre sur les faux positifs connus avant déploiement

## Prompt RAG : les instructions "INTÉGRALEMENT" et "COMPLÈTEMENT" causent la sur-justification
- **Problème** : le prompt system v1 disait "reproduis-les COMPLÈTEMENT", "Détaille INTÉGRALEMENT"
- **Conséquence** : le LLM recrache tout le chunk, reformule 3 fois, empile les sources pour le même fait
- **Fix** : remplacer par "Réponds en 2-4 phrases d'abord" + "N'ajoute des détails que si demandé"
- **Pattern** : pour un RAG factuel, la concision vaut plus que l'exhaustivité

## Prompt RAG : sans consigne anti-forçage, le LLM cite n'importe quelle source
- **Problème** : quand aucune source ne répond (ex: "art. 99 sur l'IA"), le LLM force une citation
- **Exemple** : question sur l'art. 99 → affiche un chunk sur l'art. 20 (portabilité)
- **Fix** : règle explicite "Si AUCUNE source ne répond, dis-le. Ne cite PAS une source non pertinente"
- **Pattern** : le LLM préfère toujours citer quelque chose plutôt que rien — il faut l'autoriser à ne pas citer

## Évaluation RAG : séparer retrieval correctness de answer correctness
- **Insight** : une erreur de réponse peut venir du retrieval OU de la compréhension
- **Exemple CV** : le RAG ne retrouve pas "2 ans" → retrieval KO → réponse vague (pas un bug LLM)
- **Exemple AIPD critères** : le RAG retrouve le bon doc mais confond "parties du document" avec "critères de déclenchement" → retrieval OK, compréhension KO
- **Pattern** : toujours diagnostiquer les deux axes séparément avant de corriger

## Query Expansion : aide les cas borderline, pas les cas profonds
- **Mise en place** : `src/rag/query_expander.py` — LLM génère 3 reformulations, multi-query retrieval + RRF fusion
- **Résultat** : answer_correctness +8.1% (65% → 73%), score global +3.1% (85.5% → 88.6%)
- **6 questions améliorées** : q06 (+50%), q09 (+50%), q18 (+50%), q11 (+33%), q08 (+17%), q05 (+16%)
- **2 régressions** : q04 (-40%), q03 (-25%) — les reformulations noient les bons chunks
- **Coût** : temps ×2.1 (6.3s → 13.2s) à cause de l'appel LLM pour expansion
- **Limite** : si le bon chunk n'est PAS dans le top-50 du vector space, aucune reformulation ne le fera remonter
- **Pattern** : le query expansion est un multiplicateur, pas un miracle. Pour les vrais gaps, il faut améliorer l'indexation/chunking.
- **Attention** : l'expansion peut DÉGRADER certaines questions en ramenant du bruit. Surveiller les régressions.

## Cross-encoder : ms-marco est CATASTROPHIQUE sur le français juridique
- **Diagnostic** : `tasks/_diag_crossencoder.py` — ms-marco-MiniLM-L-6-v2 donne des logits de -10 sur des paires FR pertinentes (sigmoid → 0.0000)
- **Cause** : entraîné sur MS MARCO (anglais, recherche web) → ne comprend pas le vocabulaire RGPD/CNIL
- **Impact** : 2/3 questions (q05 AIPD, q09 CV) avec discrimination NULLE entre chunk pertinent et bruit
- **Fix** : remplacement par `BAAI/bge-reranker-v2-m3` (568M params, multilingue natif)
- **Résultat** : q05 80% → 90%, q10 73% → 87%, discrimination correcte sur les 3 cas testés
- **Attention** : bge-reranker-v2-m3 est 25x plus gros (2.3GB vs 90MB) → `rerank_candidates` réduit de 80 à 30
- **Seuil** : les scores bge-reranker sont calibrés différemment → seuil `relevance_threshold` ajusté de 0.40 à 0.65

## Cross-encoder : distance post-rerank = 1 - rerank_score
- **Piège** : après reranking, `chunk.distance = 1.0 - rerank_score`. Le threshold signifie rerank_score ≥ (1 - threshold)
- **Conséquence** : des chunks distance 0.95-1.00 ne sont PAS "presque pertinents", ils sont REJETÉS par le cross-encoder (score 0.00-0.05)
- **Pattern** : ne pas confondre distance pré-rerank (embedding cosine) et post-rerank (1 - score cross-encoder)
- **Pattern** : quand on change de modèle cross-encoder, TOUJOURS recalibrer le seuil de distance

## Threshold reranker : calibrer sur la distribution réelle, pas sur l'intuition
- **Erreur** : threshold 0.65 semblait raisonnable mais rejetait des chunks pertinents (distance 0.69-0.80)
- **Symptôme** : fallback "Réponse insuffisante" systématique → récupération encore plus de docs → contexte >32K chars → crash Ollama
- **Diagnostic** : les distances bge-reranker-v2-m3 se répartissent en 2 clusters :
  - Pertinents : distance < 0.80 (score > 0.20)
  - Non pertinents : distance > 0.90 (score < 0.10)
  - Zone grise : 0.80-0.90
- **Fix** : threshold 0.65 → 0.80 — plus permissif, évite les fallbacks destructeurs
- **Pattern** : TOUJOURS analyser la distribution des distances AVANT de choisir un seuil
- **Pattern** : un threshold trop strict cause plus de dégâts qu'un threshold permissif (fallbacks cascadés, contexte explosé)

## Context builder : TOUJOURS try/except autour de map_reduce
- **Bug** : le map_reduce envoyait des batches à Ollama sans protection → timeout → crash de toute l'éval
- **Fix** : wrap dans try/except avec fallback troncature simple
- **Pattern** : tout appel LLM dans un pipeline d'évaluation DOIT avoir un fallback gracieux

## ChromaDB : pas de MMR natif
- **Fait** : ChromaDB ne supporte PAS le MMR (Maximal Marginal Relevance)
- **Alternative** : le multi-query expansion via reformulations différentes apporte de la diversité naturellement
- **Pattern** : ne pas chercher à implémenter MMR quand le multi-query + RRF fait le même job

## Pipeline tuning ≠ résolution des problèmes structurels
- **Erreur** : après le diagnostic structural (5 hypothèses GPT confirmées), j'ai passé 2h à tweaker top_k, n_chunks_per_doc, rerank_candidates
- **Résultat** : scores instables, régressions oscillantes, pas de gain net
- **Cause** : le problème est dans les CHUNKS, pas dans le PIPELINE qui les traite
- **Symptômes** : quand le meilleur chunk pour q09 est un texte sur la prospection commerciale (pas les CV), aucun réglage de retrieval ne peut aider
- **Pattern** : si l'information est mal chunké à la source, tout le reste est du sparadrap
- **Règle** : TOUJOURS faire le diagnostic chunk-level AVANT de toucher au pipeline
- **Règle** : si 3 ajustements successifs ne montrent pas de tendance claire, le problème est en amont

## Rechunking : les 4 défauts du chunker actuel (diagnostic 2026-02-11)
1. **Pas d'overlap** : `_post_process()` split par fenêtre de mots sans chevauchement → concept coupé = perdu
2. **Heading non propagé** : un chunk issu d'un split de gros document perd le heading de sa section
3. **Heading pas dans le texte embedé** : le heading est en metadata mais invisible pour nomic-embed-text
4. **Split naïf par mots** : coupe au milieu des phrases, pas aux frontières sémantiques (\n\n, .)
5. **Tiny chunks** : 441 chunks <50 mots = bruit pur dans la vectorDB

## Concentration = métrique clé de qualité de chunking
- **Définition** : (chunks contenant TOUS les keywords manquants) / (chunks contenant au moins 1)
- **Bonne concentration** : >30% = l'info est regroupée dans des chunks cohérents
- **Mauvaise concentration** : <5% = l'info est éparpillée, le retriever doit deviner
- **Exemples mesurés** :
  - q01 "identifiée ou identifiable" : 100% (l'expression est atomique)
  - q05 "données sensibles" + "grande échelle" : 5.5% (dilution sévère)
  - q09 "2 ans" + "dernier contact" : 0.5% (quasiment absent)
  - q10 "mise en balance" + "sécurité" : 0.5% (extrême dilution)

## Évaluation : questions vagues + must_include rigides = faux négatifs
- **Symptôme** : q05 demande "les trois critères principaux" (il y en a 9 équivalents), q08 demande "les droits" (sans exhaustivité), q11 attend "contrat de travail" alors que "contrat" suffit en contexte RH
- **Impact mesuré** : 89% → 93% global rien qu'en fixant l'éval (même réponses LLM)
- **Fix** : ajout de `must_include_any` (N parmi M) et alternates pipe-separated (`contrat de travail|contrat|exécution du contrat`)
- **Pattern** : le score d'évaluation doit mesurer la QUALITÉ de la réponse, pas sa correspondance littérale à un template
- **Règle** : si une question vague accepte plusieurs réponses correctes, utiliser `must_include_any` au lieu de `must_include` strict
- **Règle** : toujours mettre des alternates sémantiques pour les concepts qui se reformulent (`obligation légale|fondé sur|base légale`)
- **Diagnostic rapide** : si le LLM donne une réponse factuelle correcte mais que l'éval dit 0% correctness → le biais est dans l'éval, pas dans le RAG