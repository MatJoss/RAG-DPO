"""
Évaluation automatique du RAG-DPO

Score chaque réponse sur 4 axes :
1. Retrieval Correctness  — Les bons chunks ont été récupérés ?
2. Answer Correctness     — La réponse contient les éléments attendus ?
3. Faithfulness           — Pas d'hallucination / invention ?
4. Conciseness            — Réponse disciplinée (pas de sur-justification) ?

Usage :
    python eval/run_eval.py                     # Toutes les questions
    python eval/run_eval.py --ids q01 q05 q09   # Questions spécifiques
    python eval/run_eval.py --dry-run            # Affiche les questions sans exécuter
    python eval/run_eval.py --verbose            # Affiche les réponses complètes
"""
import json
import sys
import time
import re
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
for noisy in ("httpx", "ollama", "chromadb", "sentence_transformers", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Scoring Functions
# ═══════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Normalise le texte pour comparaison flexible."""
    text = text.lower()
    # Accents déjà gérés par Python, on normalise les espaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Nombres écrits en lettres → chiffres (pour matching flexible)
_NUMBER_WORDS = {
    "zéro": "0", "un": "1", "une": "1", "deux": "2", "trois": "3",
    "quatre": "4", "cinq": "5", "six": "6", "sept": "7", "huit": "8",
    "neuf": "9", "dix": "10", "onze": "11", "douze": "12", "treize": "13",
    "quatorze": "14", "quinze": "15", "vingt": "20", "trente": "30",
}


def _normalize_numbers(text: str) -> str:
    """Remplace les nombres écrits en lettres par leurs chiffres."""
    words = text.split()
    result = []
    for w in words:
        result.append(_NUMBER_WORDS.get(w, w))
    return " ".join(result)


def _flexible_match(item_norm: str, answer_norm: str) -> bool:
    """
    Matching flexible : gère singulier/pluriel, accents, nombres lettres/chiffres.
    
    Stratégie :
    1. Match exact
    2. Match avec normalisation des nombres ("deux ans" → "2 ans")
    3. Match par mots individuels avec tolérance pluriel
       ("risque élevé" matche "risques élevés")
    """
    # 1. Match exact
    if item_norm in answer_norm:
        return True
    
    # 2. Match avec normalisation des nombres
    item_num = _normalize_numbers(item_norm)
    answer_num = _normalize_numbers(answer_norm)
    if item_num in answer_num:
        return True
    
    # 3. Match par mots avec tolérance pluriel (s, es, ées, és)
    item_words = item_norm.split()
    if len(item_words) >= 2:
        # Chaque mot de l'item doit apparaître dans la réponse (avec tolérance pluriel)
        all_found = True
        for word in item_words:
            # Chercher le mot exact ou ses variantes singulier/pluriel
            stems = {word}
            # Ajouter le pluriel
            stems.add(word + "s")
            stems.add(word + "es")
            # Retirer le pluriel (si le mot finit par s/es)
            if word.endswith("es") and len(word) > 3:
                stems.add(word[:-2])
            if word.endswith("s") and len(word) > 2:
                stems.add(word[:-1])
            # Variantes féminines/masculines
            if word.endswith("é"):
                stems.add(word + "e")
                stems.add(word + "es")
                stems.add(word + "s")
            if word.endswith("ée"):
                stems.add(word[:-1])
                stems.add(word + "s")
            if word.endswith("ées"):
                stems.add(word[:-1])
                stems.add(word[:-2])
                stems.add(word[:-3])
            
            # Vérifier qu'au moins une variante est dans la réponse
            word_found = False
            for stem in stems:
                # Match en tant que mot (pas substring partiel)
                if re.search(r'\b' + re.escape(stem) + r'\b', answer_norm):
                    word_found = True
                    break
                # Aussi vérifier avec normalisation nombres
                stem_num = _normalize_numbers(stem)
                if stem_num != stem and re.search(r'\b' + re.escape(stem_num) + r'\b', answer_num):
                    word_found = True
                    break
            
            if not word_found:
                all_found = False
                break
        
        if all_found:
            return True
    
    return False


def _match_item_or_alternates(item: str, answer_norm: str) -> bool:
    """
    Matche un item (éventuellement avec alternatives séparées par |).
    Ex: "contrat de travail|contrat|exécution du contrat" → match si au moins un trouvé.
    """
    alternates = [normalize_text(alt.strip()) for alt in item.split("|")]
    for alt in alternates:
        if _flexible_match(alt, answer_norm):
            return True
    
    # Variantes sémantiques hardcodées (rétro-compatibilité)
    first = alternates[0]
    if first == "non":
        if any(neg in answer_norm for neg in ["ne ", "n'", "pas possible", "n'autorise pas", "pas de", "pas d'"]):
            return True
    if first == "pas possible":
        if any(neg in answer_norm for neg in ["ne peut pas", "n'est pas possible", "pas possible", "ne pas", "impossible", "interdit", "illégal"]):
            return True
    if first in ("obligatoire", "obligation"):
        if any(v in answer_norm for v in ["obligatoire", "obligation", "obligatoirement", "imposé", "impose", "doit être réalisée", "doit être réalisé"]):
            return True
    if first == "sanction":
        if any(v in answer_norm for v in ["sanction", "sanctions", "sanctionner", "sanctionné", "peine", "amende"]):
            return True
    return False


def score_must_include(
    answer: str,
    must_include: List[str],
    must_include_any: Optional[Dict] = None,
) -> Tuple[float, List[str], List[str]]:
    """
    Vérifie que les éléments obligatoires sont présents.
    Matching flexible : gère singulier/pluriel, nombres en lettres, variantes.
    
    Supporte deux modes :
    - must_include : TOUS les items doivent être présents (AND)
    - must_include_any : Au moins N items parmi une liste (N parmi M)
      Format: {"min_count": N, "items": ["item1", "item2", ...]}
      Chaque item peut contenir des alternatives séparées par | :
      "contrat de travail|exécution du contrat"
    
    Le score combine les deux modes si les deux sont présents.
    
    Returns:
        (score 0-1, found_items, missing_items)
    """
    answer_norm = normalize_text(answer)
    found = []
    missing = []
    scores = []
    
    # --- Mode classique : TOUS les items (AND) ---
    if must_include:
        for item in must_include:
            if _match_item_or_alternates(item, answer_norm):
                found.append(item)
            else:
                missing.append(item)
        scores.append(len(found) / len(must_include))
    
    # --- Mode "N parmi M" (must_include_any) ---
    if must_include_any:
        any_items = must_include_any.get("items", [])
        min_count = must_include_any.get("min_count", 1)
        any_found = []
        any_missing = []
        
        for item in any_items:
            if _match_item_or_alternates(item, answer_norm):
                any_found.append(item)
            else:
                any_missing.append(item)
        
        n_matched = len(any_found)
        if n_matched >= min_count:
            any_score = 1.0
        else:
            any_score = n_matched / min_count
        scores.append(any_score)
        
        # Ajouter au rapport
        found.extend([f"[any:{it}]" for it in any_found])
        if n_matched < min_count:
            missing.append(f"[any: {n_matched}/{min_count} trouvés, besoin {min_count}]")
    
    if not scores:
        return 1.0, [], []
    
    score = sum(scores) / len(scores)
    return score, found, missing


def score_must_not_include(answer: str, must_not_include: List[str]) -> Tuple[float, List[str]]:
    """
    Vérifie que les éléments interdits sont absents.
    
    Returns:
        (score 0-1, violations)
    """
    if not must_not_include:
        return 1.0, []
    
    answer_norm = normalize_text(answer)
    violations = []
    
    for item in must_not_include:
        item_norm = normalize_text(item)
        if item_norm in answer_norm:
            violations.append(item)
    
    score = 1.0 - (len(violations) / len(must_not_include))
    return score, violations


def score_conciseness(answer: str, category: str) -> Tuple[float, str]:
    """
    Évalue la concision de la réponse.
    
    Seuils relaxés : la concision est un SIGNAL, pas un couperet.
    Un RAG avec un meilleur retrieval fournit plus de contexte → le LLM
    produit naturellement des réponses plus longues et plus complètes.
    On pénalise seulement la verbosité excessive (>2× l'idéal).
    
    Returns:
        (score 0-1, assessment)
    """
    word_count = len(answer.split())
    
    # Limites par catégorie (idéal, max_soft, max_hard)
    # idéal = score 1.0, max_soft = score 0.7, max_hard = score plancher 0.4
    limits = {
        "definition": (150, 400, 600),
        "obligation": (200, 500, 800),
        "recommandation": (200, 500, 800),
        "piège": (80, 250, 400),
        "hors_perimetre": (50, 150, 300),
    }
    
    ideal, max_soft, max_hard = limits.get(category, (200, 500, 800))
    
    if word_count <= ideal:
        return 1.0, f"✅ {word_count} mots (idéal ≤{ideal})"
    elif word_count <= max_soft:
        # Pente douce : 1.0 → 0.7
        score = 1.0 - 0.3 * (word_count - ideal) / (max_soft - ideal)
        return score, f"⚠️ {word_count} mots (idéal ≤{ideal}, max {max_soft})"
    elif word_count <= max_hard:
        # Pente plus forte : 0.7 → 0.4
        score = 0.7 - 0.3 * (word_count - max_soft) / (max_hard - max_soft)
        return score, f"❌ {word_count} mots (trop long, max {max_soft})"
    else:
        # Au-delà du max hard : plancher 0.3
        return 0.3, f"❌ {word_count} mots (excessif, max {max_hard})"


def score_source_quality(answer: str, category: str = "") -> Tuple[float, Dict]:
    """
    Évalue la qualité de l'usage des sources.
    
    Checks :
    - Présence de [Source X] citations
    - Pas de source inventée (numéro > 12 suspect)
    - Pas de chunk non pertinent affiché (art. 20 pour question sur art. 99)
    - Questions hors-périmètre / piège : ne pas pénaliser l'absence de source (refus = bon comportement)
    
    Returns:
        (score 0-1, details)
    """
    # Compter les citations
    citations = re.findall(r'\[Source\s*(\d+)\]', answer)
    unique_sources = set(citations)
    n_citations = len(citations)
    n_unique_sources = len(unique_sources)
    
    details = {
        "n_citations": n_citations,
        "n_unique_sources": n_unique_sources,
        "source_ids": sorted(unique_sources),
    }
    
    # Score de base
    if n_citations == 0:
        # Pour les refus (hors-périmètre, piège), ne pas citer = bon comportement
        if category in ("hors_perimetre", "piège"):
            return 1.0, details
        return 0.5, details
    
    # Vérifier cohérence des numéros de source
    max_source_id = max(int(s) for s in unique_sources)
    if max_source_id > 12:
        details["warning"] = f"Source {max_source_id} suspicieusement élevée"
        return 0.3, details
    
    # Score basé sur la présence de citations
    if n_unique_sources >= 1:
        return 1.0, details
    
    return 0.8, details


# ═══════════════════════════════════════════════════════════════
# LLM-as-Judge Scoring
# ═══════════════════════════════════════════════════════════════

_llm_judge_provider = None  # Initialisé une seule fois


def _get_llm_judge():
    """Singleton du provider LLM pour le judge."""
    global _llm_judge_provider
    if _llm_judge_provider is None:
        from src.utils.llm_provider import OllamaProvider
        _llm_judge_provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="mistral-nemo"
        )
    return _llm_judge_provider


def llm_judge_correctness(
    question: str,
    expected_answer: str,
    actual_answer: str,
) -> Tuple[float, str]:
    """
    Utilise le LLM pour juger si la réponse fournie couvre les concepts
    de la réponse attendue.
    
    Avantage vs keyword matching : comprend les synonymes, paraphrases,
    reformulations ("2 ans" = "24 mois" = "deux années").
    
    Returns:
        (score 0-1, justification)
    """
    provider = _get_llm_judge()
    
    prompt = f"""Tu es un évaluateur expert en RGPD/CNIL. Compare la réponse fournie avec la réponse attendue.

Question : {question}

Réponse attendue (référence) :
{expected_answer}

Réponse fournie (à évaluer) :
{actual_answer}

Évalue si la réponse fournie couvre les CONCEPTS CLÉS de la réponse attendue.
- Ne juge PAS les mots exacts, mais le SENS ("2 ans" = "24 mois" = "deux années")
- Une réponse qui dit la même chose avec des mots différents est CORRECTE
- Une réponse partielle (certains concepts manquants) mérite un score proportionnel
- Une réponse hors-sujet ou incorrecte = 0

Réponds UNIQUEMENT avec ce format exact :
SCORE: [nombre entier de 0 à 100]
JUSTIFICATION: [une phrase courte expliquant le score]"""
    
    try:
        response = provider.generate(
            prompt,
            temperature=0.0,
            max_tokens=150,
        )
        
        # Parser la réponse
        score = 0.0
        justification = "Parsing failed"
        
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("SCORE"):
                # Extraire le nombre
                import re as _re
                numbers = _re.findall(r'\d+', line)
                if numbers:
                    raw_score = int(numbers[0])
                    score = max(0.0, min(1.0, raw_score / 100.0))
            elif line.upper().startswith("JUSTIFICATION"):
                justification = line.split(":", 1)[-1].strip()
        
        return score, justification
        
    except Exception as e:
        logger.warning(f"⚠️  LLM judge error: {e}")
        return -1.0, f"Error: {e}"  # -1 = signal to fall back to keyword-only


def evaluate_single(qa_item: Dict, answer: str, sources: List[Dict] = None, use_llm_judge: bool = True) -> Dict:
    """
    Évalue une seule réponse du RAG contre le dataset attendu.
    
    Args:
        qa_item: Question du dataset avec expected_answer_summary, must_include, etc.
        answer: Réponse du RAG
        sources: Sources retournées (optionnel)
        use_llm_judge: Si True, utilise le LLM pour juger la correctness (plus fiable)
    
    Returns:
        Dict avec scores détaillés
    """
    result = {
        "id": qa_item["id"],
        "question": qa_item["question"],
        "category": qa_item["category"],
        "difficulty": qa_item["difficulty"],
    }
    
    # 1. Answer Correctness — Must Include
    include_score, found, missing = score_must_include(
        answer, qa_item.get("must_include", []),
        must_include_any=qa_item.get("must_include_any"),
    )
    result["answer_correctness"] = {
        "score": include_score,
        "found": found,
        "missing": missing,
    }
    
    # 1b. LLM Judge — Évaluation sémantique par le LLM
    llm_score = -1.0
    llm_justification = "disabled"
    if use_llm_judge and qa_item.get("expected_answer_summary"):
        llm_score, llm_justification = llm_judge_correctness(
            question=qa_item["question"],
            expected_answer=qa_item["expected_answer_summary"],
            actual_answer=answer,
        )
    
    # Combiner keyword + LLM judge pour le score final de correctness
    if llm_score >= 0:  # LLM judge a fonctionné
        # LLM judge 70%, keyword 30% — le LLM comprend les synonymes
        combined_correctness = 0.70 * llm_score + 0.30 * include_score
        result["answer_correctness"]["llm_judge_score"] = round(llm_score, 2)
        result["answer_correctness"]["llm_justification"] = llm_justification
        result["answer_correctness"]["keyword_score"] = round(include_score, 2)
        result["answer_correctness"]["score"] = round(combined_correctness, 2)
    else:
        # Fallback keyword-only
        combined_correctness = include_score
        result["answer_correctness"]["llm_judge_score"] = None
        result["answer_correctness"]["llm_justification"] = llm_justification
    
    # 2. Faithfulness — Must Not Include (anti-hallucination)
    not_include_score, violations = score_must_not_include(
        answer, qa_item.get("must_not_include", [])
    )
    result["faithfulness"] = {
        "score": not_include_score,
        "violations": violations,
    }
    
    # 3. Conciseness
    concise_score, concise_assessment = score_conciseness(
        answer, qa_item["category"]
    )
    result["conciseness"] = {
        "score": concise_score,
        "assessment": concise_assessment,
    }
    
    # 4. Source Quality
    source_score, source_details = score_source_quality(answer, qa_item.get("category", ""))
    result["source_quality"] = {
        "score": source_score,
        **source_details,
    }
    
    # Score global pondéré
    # Answer Correctness : 45%, Faithfulness : 25%, Conciseness : 10%, Sources : 20%
    result["global_score"] = round(
        0.45 * combined_correctness +
        0.25 * not_include_score +
        0.10 * concise_score +
        0.20 * source_score,
        2
    )
    
    return result


# ═══════════════════════════════════════════════════════════════
# Pipeline Init
# ═══════════════════════════════════════════════════════════════

def init_pipeline(embedding_mode: str = "bge-m3", enable_dual_gen: bool = True, use_agent: bool = False):
    """
    Initialise le pipeline RAG pour l'évaluation.
    
    Args:
        embedding_mode: 'bge-m3' (défaut) ou 'nomic' pour sélectionner
                        l'embedding provider et la collection ChromaDB correspondante.
        enable_dual_gen: Active/désactive la dual-generation (self-consistency).
        use_agent: Si True, utilise le pipeline LangGraph agent au lieu du natif.
    """
    import chromadb
    from src.utils.llm_provider import OllamaProvider
    from src.rag.pipeline import create_pipeline
    
    vectordb_path = project_root / "data" / "vectordb" / "chromadb"
    if not vectordb_path.exists():
        print(f"❌ VectorDB introuvable : {vectordb_path}")
        sys.exit(1)
    
    client = chromadb.PersistentClient(path=str(vectordb_path))
    
    # Sélection collection + embedding provider selon le mode
    if embedding_mode == "nomic":
        collection_name = "rag_dpo_chunks_nomic"
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            print(f"❌ Collection '{collection_name}' introuvable.")
            print(f"   Lancez d'abord : python eval/index_nomic.py")
            sys.exit(1)
        
        llm_provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="mistral-nemo"
        )
        # Nomic : embeddings via OllamaProvider (768 dims)
        # embedding_provider=None → le retriever utilisera llm_provider.embed()
        embedding_provider = None
        print(f"📐 Embedding: nomic-embed-text (Ollama, 768d) — collection: {collection_name}")
    else:
        collection_name = "rag_dpo_chunks"
        collection = client.get_collection(collection_name)
        
        llm_provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="mistral-nemo"
        )
        
        from src.utils.embedding_provider import EmbeddingProvider
        embedding_provider = EmbeddingProvider(
            cache_dir=str(project_root / "models" / "huggingface" / "hub"),
        )
        print(f"📐 Embedding: BGE-M3 (sentence-transformers, 1024d) — collection: {collection_name}")
    
    pipeline = create_pipeline(
        collection=collection,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        n_documents=5,
        n_chunks_per_doc=3,
        enable_hybrid=True,
        enable_reranker=True,
        enable_dual_gen=enable_dual_gen,
        enable_summary_prefilter=True,
        enable_validation=True,
        model="mistral-nemo",
        temperature=0.0,
        debug_mode=True,
    )
    
    if use_agent:
        from src.rag.agent import create_agent_pipeline
        pipeline = create_agent_pipeline(
            collection=collection,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            n_documents=5,
            n_chunks_per_doc=3,
            enable_hybrid=True,
            enable_reranker=True,
            enable_summary_prefilter=True,
            enable_validation=True,
            model="mistral-nemo",
            temperature=0.0,
            max_retries=1,
        )
        print(f"\U0001f916 Mode: Agent LangGraph (classify→retrieve→generate→validate→respond)")
    
    return pipeline


# ═══════════════════════════════════════════════════════════════
# Main Evaluation Loop
# ═══════════════════════════════════════════════════════════════

def run_evaluation(
    dataset_path: str,
    question_ids: Optional[List[str]] = None,
    dry_run: bool = False,
    verbose: bool = False,
    use_llm_judge: bool = True,
    embedding_mode: str = "bge-m3",
    enable_dual_gen: bool = True,
    use_agent: bool = False,
) -> Dict:
    """
    Exécute l'évaluation complète.
    
    Args:
        dataset_path: Chemin vers qa_dataset.json
        question_ids: Liste d'IDs à évaluer (None = tous)
        dry_run: Si True, affiche les questions sans les exécuter
        verbose: Si True, affiche les réponses complètes
        embedding_mode: 'bge-m3' ou 'nomic'
    
    Returns:
        Résultats complets de l'évaluation
    """
    # Charger le dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    
    # Filtrer si nécessaire
    if question_ids:
        dataset = [q for q in dataset if any(qid in q["id"] for qid in question_ids)]
    
    if not dataset:
        print("❌ Aucune question trouvée avec les IDs spécifiés")
        return {}
    
    emb_label = "BGE-M3" if embedding_mode == "bge-m3" else "nomic-embed-text"
    print(f"\n{'='*70}")
    print(f"🧪 ÉVALUATION RAG-DPO — {len(dataset)} questions [{emb_label}]")
    print(f"{'='*70}\n")
    
    if dry_run:
        for i, q in enumerate(dataset, 1):
            print(f"  {i:2d}. [{q['category']:20s}] {q['question']}")
            print(f"      Expected: {q['expected_answer_summary'][:100]}...")
            print()
        return {}
    
    # Init pipeline
    dual_label = "dual-gen" if enable_dual_gen else "single-gen"
    agent_label = " [Agent LangGraph]" if use_agent else ""
    print(f"\u23f3 Initialisation du pipeline RAG ({emb_label}, {dual_label}{agent_label})...")
    pipeline = init_pipeline(embedding_mode=embedding_mode, enable_dual_gen=enable_dual_gen, use_agent=use_agent)
    print("✅ Pipeline prêt\n")
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 1 : Génération des réponses RAG + scoring keyword
    # (Le pipeline RAG monopolise le GPU — on ne fait PAS de LLM judge ici)
    # ═══════════════════════════════════════════════════════════
    
    results = []
    answers_for_judge = []  # (index, qa_item, answer) pour phase 2
    total_time = 0
    
    print("📝 Phase 1 : Génération des réponses RAG\n")
    
    for i, qa_item in enumerate(dataset, 1):
        question = qa_item["question"]
        qid = qa_item["id"]
        
        print(f"── [{i}/{len(dataset)}] {qid} {'─'*40}")
        print(f"   Q: {question}")
        
        # Exécuter la query
        t_start = time.time()
        try:
            response = pipeline.query(question)
            elapsed = time.time() - t_start
            total_time += elapsed
            
            answer = response.answer
            sources = response.sources if response.sources else []
            
            # Évaluer (keyword-only pour l'instant, LLM judge en phase 2)
            eval_result = evaluate_single(qa_item, answer, sources, use_llm_judge=False)
            eval_result["elapsed_seconds"] = round(elapsed, 1)
            eval_result["answer_length_words"] = len(answer.split())
            eval_result["_raw_answer"] = answer  # Conserver pour phase 2
            
            if verbose:
                print(f"   R: {answer[:500]}{'...' if len(answer) > 500 else ''}")
            
            # Résumé rapide (keyword-only)
            global_score = eval_result["global_score"]
            icon = "🟢" if global_score >= 0.8 else "🟡" if global_score >= 0.5 else "🔴"
            
            ac = eval_result["answer_correctness"]
            ff = eval_result["faithfulness"]
            cc = eval_result["conciseness"]
            sq = eval_result["source_quality"]
            
            print(f"   {icon} Score: {global_score:.0%}  "
                  f"(Correct[kw]: {ac['score']:.0%} | "
                  f"Faithful: {ff['score']:.0%} | "
                  f"Concis: {cc['score']:.0%} | "
                  f"Sources: {sq['score']:.0%})  "
                  f"[{elapsed:.1f}s, {eval_result['answer_length_words']}w]")
            
            if ac["missing"]:
                print(f"   ⚠️  Manquant (kw): {', '.join(ac['missing'])}")
            if ff["violations"]:
                print(f"   🚫 Violations: {', '.join(ff['violations'])}")
            
            # Stocker pour phase 2
            if use_llm_judge:
                answers_for_judge.append((len(results), qa_item, answer))
            
            results.append(eval_result)
            
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"   ❌ ERREUR: {e}")
            results.append({
                "id": qid,
                "question": question,
                "error": str(e),
                "global_score": 0.0,
                "elapsed_seconds": round(elapsed, 1),
            })
        
        print()
    
    # ═══════════════════════════════════════════════════════════
    # PHASE 2 : LLM-as-Judge (séquentiel, après libération du GPU par le pipeline)
    # ═══════════════════════════════════════════════════════════
    
    if use_llm_judge and answers_for_judge:
        # Libérer le pipeline RAG de la mémoire avant le LLM judge
        del pipeline
        import gc
        gc.collect()
        
        print(f"\n{'='*70}")
        print(f"🤖 Phase 2 : LLM-as-Judge — {len(answers_for_judge)} réponses à évaluer")
        print(f"{'='*70}\n")
        
        for idx, qa_item, answer in answers_for_judge:
            qid = qa_item["id"]
            expected = qa_item.get("expected_answer_summary", "")
            
            if not expected:
                continue
            
            print(f"   🤖 [{qid}] ", end="", flush=True)
            
            llm_score, llm_justification = llm_judge_correctness(
                question=qa_item["question"],
                expected_answer=expected,
                actual_answer=answer,
            )
            
            if llm_score >= 0:
                r = results[idx]
                kw_score = r["answer_correctness"]["score"]
                combined = round(0.70 * llm_score + 0.30 * kw_score, 2)
                
                r["answer_correctness"]["llm_judge_score"] = round(llm_score, 2)
                r["answer_correctness"]["llm_justification"] = llm_justification
                r["answer_correctness"]["keyword_score"] = round(kw_score, 2)
                r["answer_correctness"]["score"] = combined
                
                # Recalculer le score global
                r["global_score"] = round(
                    0.45 * combined +
                    0.25 * r["faithfulness"]["score"] +
                    0.10 * r["conciseness"]["score"] +
                    0.20 * r["source_quality"]["score"],
                    2
                )
                
                icon = "🟢" if r["global_score"] >= 0.8 else "🟡" if r["global_score"] >= 0.5 else "🔴"
                print(f"{icon} LLM={llm_score:.0%} KW={kw_score:.0%} → {combined:.0%} (global={r['global_score']:.0%})")
                if verbose:
                    print(f"      📝 {llm_justification}")
            else:
                print(f"⚠️  Erreur — fallback keyword")
        
        print()
    
    # Nettoyer les champs internes
    for r in results:
        r.pop("_raw_answer", None)
    
    # ═══════════════════════════════════════════════════════════
    # Rapport Final
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"📊 RAPPORT D'ÉVALUATION")
    print(f"{'='*70}\n")
    
    # Scores globaux
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        avg_global = sum(r["global_score"] for r in valid_results) / len(valid_results)
        avg_correct = sum(r["answer_correctness"]["score"] for r in valid_results) / len(valid_results)
        avg_faithful = sum(r["faithfulness"]["score"] for r in valid_results) / len(valid_results)
        avg_concise = sum(r["conciseness"]["score"] for r in valid_results) / len(valid_results)
        avg_sources = sum(r["source_quality"]["score"] for r in valid_results) / len(valid_results)
        avg_time = sum(r["elapsed_seconds"] for r in valid_results) / len(valid_results)
        
        print(f"  📈 Score Global Moyen   : {avg_global:.0%}")
        print(f"  ✅ Answer Correctness   : {avg_correct:.0%}")
        print(f"  🛡️  Faithfulness         : {avg_faithful:.0%}")
        print(f"  📏 Conciseness          : {avg_concise:.0%}")
        print(f"  📚 Source Quality        : {avg_sources:.0%}")
        print(f"  ⏱️  Temps moyen/question : {avg_time:.1f}s")
        print(f"  ⏱️  Temps total          : {total_time:.0f}s")
        
        # Par catégorie
        categories = set(r["category"] for r in valid_results)
        print(f"\n  Par catégorie :")
        for cat in sorted(categories):
            cat_results = [r for r in valid_results if r["category"] == cat]
            cat_avg = sum(r["global_score"] for r in cat_results) / len(cat_results)
            icon = "🟢" if cat_avg >= 0.8 else "🟡" if cat_avg >= 0.5 else "🔴"
            print(f"    {icon} {cat:20s} : {cat_avg:.0%} ({len(cat_results)} questions)")
        
        # Top 3 pires
        sorted_results = sorted(valid_results, key=lambda r: r["global_score"])
        print(f"\n  🔴 Points faibles (3 pires) :")
        for r in sorted_results[:3]:
            print(f"    • {r['id']:35s} : {r['global_score']:.0%} — {r['question'][:60]}")
        
        # Top 3 meilleurs
        print(f"\n  🟢 Points forts (3 meilleurs) :")
        for r in sorted_results[-3:]:
            print(f"    • {r['id']:35s} : {r['global_score']:.0%} — {r['question'][:60]}")
    
    if results:
        errors = [r for r in results if "error" in r]
        if errors:
            print(f"\n  ❌ {len(errors)} erreur(s) :")
            for r in errors:
                print(f"    • {r['id']} : {r['error'][:100]}")
    
    # Sauvegarder les résultats
    emb_suffix = "nomic" if embedding_mode == "nomic" else "bge-m3"
    gen_suffix = "single" if not enable_dual_gen else "dual"
    agent_suffix = "_agent" if use_agent else ""
    output_path = project_root / "eval" / f"results_{emb_suffix}_{gen_suffix}{agent_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "embedding_model": emb_label,
            "embedding_mode": embedding_mode,
            "generation_mode": "single" if not enable_dual_gen else "dual",
            "pipeline_mode": "agent" if use_agent else "native",
            "n_questions": len(dataset),
            "n_valid": len(valid_results),
            "n_errors": len(results) - len(valid_results),
            "avg_global_score": round(avg_global, 3) if valid_results else 0,
            "avg_time_per_question": round(avg_time, 1) if valid_results else 0,
            "scoring_weights": {"correctness": 0.45, "faithfulness": 0.25, "conciseness": 0.10, "sources": 0.20},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Résultats sauvegardés : {output_path}")
    print(f"{'='*70}\n")
    
    return {
        "avg_global": avg_global if valid_results else 0,
        "results": results,
        "output_path": str(output_path),
    }


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Évaluation automatique du RAG-DPO")
    parser.add_argument("--ids", nargs="+", help="IDs des questions à évaluer (ex: q01 q05)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les questions sans exécuter")
    parser.add_argument("--verbose", action="store_true", help="Affiche les réponses complètes")
    parser.add_argument("--no-llm-judge", action="store_true", help="Désactive le LLM-as-judge (keyword matching seul)")
    parser.add_argument("--dataset", default=None, help="Chemin vers le dataset JSON")
    parser.add_argument("--embedding", choices=["bge-m3", "nomic"], default="bge-m3",
                        help="Modèle d'embedding à utiliser (défaut: bge-m3)")
    parser.add_argument("--no-dual", action="store_true",
                        help="Désactive la dual-generation (single-gen, plus rapide)")
    parser.add_argument("--agent", action="store_true",
                        help="Utilise le pipeline LangGraph agent au lieu du natif")
    
    args = parser.parse_args()
    
    dataset_path = args.dataset or str(project_root / "eval" / "qa_dataset.json")
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset introuvable : {dataset_path}")
        sys.exit(1)
    
    run_evaluation(
        dataset_path=dataset_path,
        question_ids=args.ids,
        dry_run=args.dry_run,
        verbose=args.verbose,
        use_llm_judge=not args.no_llm_judge,
        embedding_mode=args.embedding,
        enable_dual_gen=not args.no_dual,
        use_agent=args.agent,
    )


if __name__ == "__main__":
    main()
