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
    python eval/run_eval.py --runs 3             # 3 runs → moyenne ± σ (recommandé)
    python eval/run_eval.py --runs 3 --agent     # Multi-run avec pipeline agent
"""
import json
import sys
import time
import re
import argparse
import logging
import statistics
import numpy as np
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


def score_conciseness(answer: str, category: str, intent: str = "factuel") -> Tuple[float, str]:
    """
    Évalue la concision de la réponse.
    
    Seuils adaptatifs selon l'intent :
    - Les prompts méthodologiques/organisationnels DEMANDENT des réponses plus longues
      (étapes, acteurs, livrables) → seuils plus larges
    - Les questions factuelles doivent rester concises
    - On ne pénalise PAS un prompt structuré qui produit le format attendu
    
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
    
    # Ajustement intent-aware : les prompts structurés produisent naturellement
    # des réponses plus longues — ce n'est PAS de la verbosité, c'est le format
    intent_multiplier = {
        "factuel": 1.0,
        "methodologique": 1.6,      # Étapes + acteurs + livrables = ~400-500 mots normal
        "organisationnel": 1.4,     # Rôles + processus = ~300-400 mots normal
        "comparaison": 1.3,         # Tableau comparatif
        "cas_pratique": 1.4,        # Analyse structurée
        "liste_exhaustive": 1.5,    # Exhaustivité prime sur concision
        "refus": 0.3,              # Réponse très courte attendue (1-3 phrases)
    }
    mult = intent_multiplier.get(intent, 1.0)
    
    base_ideal, base_soft, base_hard = limits.get(category, (200, 500, 800))
    ideal = int(base_ideal * mult)
    max_soft = int(base_soft * mult)
    max_hard = int(base_hard * mult)
    
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
# Semantic Similarity Scoring (BGE-M3)
# ═══════════════════════════════════════════════════════════════

_semantic_scorer = None  # Initialisé une seule fois


def _get_semantic_scorer():
    """Singleton du modèle d'embedding pour la similarité sémantique."""
    global _semantic_scorer
    if _semantic_scorer is None:
        from src.utils.embedding_provider import EmbeddingProvider
        _semantic_scorer = EmbeddingProvider(
            cache_dir=str(project_root / "models" / "huggingface" / "hub"),
        )
    return _semantic_scorer


def score_semantic_similarity(
    expected_answer: str,
    actual_answer: str,
) -> float:
    """
    Calcule la similarité sémantique via BGE-M3 embeddings.
    
    Avantage : capture le sens sans dépendre des mots exacts.
    "responsable de traitement" ≈ "celui qui détermine les finalités" → score élevé.
    
    BGE-M3 est déjà normalisé → cosine similarity = dot product.
    
    Returns:
        score 0-1 (cosine similarity)
    """
    try:
        scorer = _get_semantic_scorer()
        embeddings = scorer.embed([expected_answer, actual_answer])
        
        # Dot product (vecteurs déjà normalisés)
        sim = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, sim))  # Clamp to [0, 1]
    except Exception as e:
        logger.warning(f"⚠️  Semantic similarity error: {e}")
        return -1.0  # Signal fallback


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
    Juge concept-aware optimisé pour Mistral-Nemo 12B.
    
    Stratégie v7 : score LIBRE 0-100 par le LLM, utilisé directement.
    Le LLM n'est pas contraint par des paliers fixes, ce qui évite le
    clustering et préserve toute la granularité du jugement.
    
    Un élément essentiel = un élément dont l'absence changerait
    le sens juridique de la réponse.
    
    Returns:
        (score 0-1, justification)
    """
    provider = _get_llm_judge()
    
    prompt = f"""Tu es un évaluateur juridique spécialisé en RGPD.
Compare la réponse fournie avec la réponse attendue.
Juge le SENS et la COUVERTURE, jamais la formulation.

Question : {question}

Réponse attendue (référence) :
{expected_answer}

Réponse fournie (à évaluer) :
{actual_answer}

RÈGLE 1 — NE PÉNALISE JAMAIS pour :
- Reformulation différente (c'est normal et attendu)
- Formulation plus détaillée ou plus synthétique
- Ordre différent des éléments
- Information supplémentaire correcte
- Équivalences : "RGPD"="GDPR", "2 ans"="24 mois", "données sensibles"="catégories particulières", "grande échelle"="traitement massif", "RT"="responsable de traitement"

RÈGLE 2 — erreur_factuelle :
true = la réponse affirme quelque chose de FAUX (ex: "le DPO doit être avocat", "tous les organismes doivent nommer un DPO", "la CNIL réalise les AIPD")
false = aucune affirmation fausse

DÉFINITION : un élément essentiel est un élément dont l'absence changerait le sens juridique de la réponse.

RÈGLE 3 — SCORE :
Si erreur_factuelle=true → score=0 (TOUJOURS)
Si erreur_factuelle=false → donne un score de 0 à 100 selon la couverture :
  0 = aucun rapport avec la réponse attendue
  ~30 = la majorité des éléments essentiels manquent
  ~50 = un élément essentiel manque
  ~70 = plusieurs détails secondaires manquent
  ~85 = un détail secondaire manque
  ~95-100 = la réponse couvre les éléments essentiels correctement

Donne le score que tu estimes le plus juste. N'arrondis PAS aux paliers ci-dessus, ce sont des repères.

Réponds en JSON :
{{"erreur_factuelle": true/false, "score": 0-100, "justification": "..."}}"""
    
    try:
        response = provider.generate(
            prompt,
            temperature=0.0,
            max_tokens=300,  # 150 trop court → JSON tronqué "Unterminated string"
            format='json',
        )
        
        # Parser la réponse JSON
        import json as _json
        import re as _re
        score = 0.0
        justification = "Parsing failed"
        
        # Nettoyer la réponse : parfois le LLM ajoute du texte avant/après le JSON
        raw_response = response.strip()
        
        # Tenter de réparer un JSON tronqué (justification coupée par max_tokens)
        json_str = raw_response
        try:
            data = _json.loads(json_str)
        except _json.JSONDecodeError:
            # Tentative 1 : fermer une string tronquée + fermer l'objet
            if '"justification"' in json_str and not json_str.rstrip().endswith('}'):
                json_str_fixed = json_str.rstrip()
                # Supprimer le dernier caractère incomplet si c'est au milieu d'un mot
                if not json_str_fixed.endswith('"'):
                    json_str_fixed += '"'
                if not json_str_fixed.endswith('}'):
                    json_str_fixed += '}'
                try:
                    data = _json.loads(json_str_fixed)
                    logger.debug("JSON réparé (string tronquée fermée)")
                except _json.JSONDecodeError:
                    data = None
            else:
                data = None
            
            # Tentative 2 : extraction regex des champs
            if data is None:
                logger.warning(f"⚠️  JSON parse failed, extraction regex")
                err_match = _re.search(r'"erreur_factuelle"\s*:\s*(true|false)', raw_response, _re.IGNORECASE)
                score_match = _re.search(r'"score"\s*:\s*(\d+)', raw_response)
                just_match = _re.search(r'"justification"\s*:\s*"([^"]*)', raw_response)
                
                if score_match:
                    raw_score = int(score_match.group(1))
                    erreur = err_match.group(1).lower() == "true" if err_match else False
                    justification = just_match.group(1) if just_match else "extraction regex"
                    
                    if erreur:
                        score = 0.0
                    else:
                        score = max(0.0, min(1.0, raw_score / 100.0))
                    
                    logger.debug(f"Regex extraction: raw={raw_score}, erreur={erreur} → {score}")
                    return score, justification
                else:
                    return 0.5, "JSON parse failed — score par défaut"
        
        # Parsing JSON réussi (direct ou réparé)
        raw_score = int(data.get("score", 0))
        justification = data.get("justification", "")
        
        # Garde-fou : si erreur_factuelle=true, forcer score = 0
        if data.get("erreur_factuelle", False):
            score = 0.0
            logger.debug(f"Judge raw={raw_score} → erreur_factuelle=true → 0")
        else:
            # Score libre normalisé 0-1 (pas de discrétisation)
            score = max(0.0, min(1.0, raw_score / 100.0))
            logger.debug(f"Judge raw={raw_score} → {score}")
        
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
    
    # 1. Answer Correctness — Must Include (keyword garde-fou)
    include_score, found, missing = score_must_include(
        answer, qa_item.get("must_include", []),
        must_include_any=qa_item.get("must_include_any"),
    )
    result["answer_correctness"] = {
        "score": include_score,
        "found": found,
        "missing": missing,
    }
    
    # 1b. Semantic Similarity — BGE-M3 cosine (gratuit, local, pas de faux négatif KW)
    sem_sim_score = -1.0
    expected_summary = qa_item.get("expected_answer_summary", "")
    if expected_summary:
        sem_sim_score = score_semantic_similarity(
            expected_answer=expected_summary,
            actual_answer=answer,
        )
    result["answer_correctness"]["semantic_similarity"] = round(sem_sim_score, 3) if sem_sim_score >= 0 else None
    
    # 1c. LLM Judge — Évaluation sémantique par le LLM
    llm_score = -1.0
    llm_justification = "disabled"
    if use_llm_judge and expected_summary:
        llm_score, llm_justification = llm_judge_correctness(
            question=qa_item["question"],
            expected_answer=expected_summary,
            actual_answer=answer,
        )
    
    # Combiner les signaux pour le score final de correctness
    # LLM judge : compréhension sémantique profonde (meilleur signal)
    # Semantic Sim : proxy rapide et stable (pas de stochasticité LLM)
    # Keyword : tracé uniquement, ne pèse plus dans le score
    #   Raison : "5 ans" vs "cinq ans", refus correct sans mot-clé "sanction"
    #   → le regex pénalise injustement des réponses sémantiquement correctes
    if llm_score >= 0 and sem_sim_score >= 0:
        # Dual sémantique : LLM 60% + Semantic 40%
        combined_correctness = 0.60 * llm_score + 0.40 * sem_sim_score
        result["answer_correctness"]["llm_judge_score"] = round(llm_score, 2)
        result["answer_correctness"]["llm_justification"] = llm_justification
        result["answer_correctness"]["keyword_score"] = round(include_score, 2)  # tracé uniquement
        result["answer_correctness"]["score"] = round(combined_correctness, 2)
    elif llm_score >= 0:
        # LLM seul (pas de semantic)
        combined_correctness = llm_score
        result["answer_correctness"]["llm_judge_score"] = round(llm_score, 2)
        result["answer_correctness"]["llm_justification"] = llm_justification
        result["answer_correctness"]["keyword_score"] = round(include_score, 2)  # tracé uniquement
        result["answer_correctness"]["score"] = round(combined_correctness, 2)
    elif sem_sim_score >= 0:
        # Semantic seul (pas de LLM) — keyword en fallback
        combined_correctness = 0.65 * sem_sim_score + 0.35 * include_score
        result["answer_correctness"]["keyword_score"] = round(include_score, 2)
        result["answer_correctness"]["score"] = round(combined_correctness, 2)
    else:
        # Fallback keyword-only (aucun signal sémantique)
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
    
    # 3. Conciseness — Intent-aware (prompts structurés → seuils plus larges)
    intent_str = qa_item.get("_intent", "factuel")  # Injecté par le caller si disponible
    concise_score, concise_assessment = score_conciseness(
        answer, qa_item["category"], intent=intent_str
    )
    result["conciseness"] = {
        "score": concise_score,
        "assessment": concise_assessment,
        "intent_used": intent_str,
    }
    
    # 4. Source Quality
    source_score, source_details = score_source_quality(answer, qa_item.get("category", ""))
    result["source_quality"] = {
        "score": source_score,
        **source_details,
    }
    
    # Score global pondéré
    # Answer Correctness : 55%, Faithfulness : 25%, Sources : 20%
    # Conciseness désactivée : mesurer la brièveté n'est pas un critère de qualité
    # en contexte juridique DPO. On garde le calcul pour traçabilité uniquement.
    result["global_score"] = round(
        0.55 * combined_correctness +
        0.25 * not_include_score +
        0.00 * concise_score +
        0.20 * source_score,
        2
    )
    
    return result


# ═══════════════════════════════════════════════════════════════
# Pipeline Init
# ═══════════════════════════════════════════════════════════════

def init_pipeline(enable_dual_gen: bool = False, use_agent: bool = False, rerank_top_k: int = 10):
    """
    Initialise le pipeline RAG pour l'évaluation.
    
    Args:
        enable_dual_gen: Active/désactive la dual-generation (self-consistency).
        use_agent: Si True, utilise le pipeline LangGraph agent au lieu du natif.
        rerank_top_k: Nombre de chunks gardés après reranking (défaut: 10).
    """
    import chromadb
    from src.utils.llm_provider import OllamaProvider
    from src.rag.pipeline import create_pipeline
    
    vectordb_path = project_root / "data" / "vectordb" / "chromadb"
    if not vectordb_path.exists():
        print(f"❌ VectorDB introuvable : {vectordb_path}")
        sys.exit(1)
    
    client = chromadb.PersistentClient(path=str(vectordb_path))
    
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
        rerank_top_k=rerank_top_k,
    )
    print(f"📊 Rerank top_k: {rerank_top_k}")
    
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
            rerank_top_k=rerank_top_k,
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
    enable_dual_gen: bool = False,
    use_agent: bool = False,
    rerank_top_k: int = 10,
) -> Dict:
    """
    Exécute l'évaluation complète.
    
    Args:
        dataset_path: Chemin vers qa_dataset.json
        question_ids: Liste d'IDs à évaluer (None = tous)
        dry_run: Si True, affiche les questions sans les exécuter
        verbose: Si True, affiche les réponses complètes
        rerank_top_k: Nombre de chunks gardés après reranking (défaut: 10)
    
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
    
    emb_label = "BGE-M3"
    topk_label = f", top_k={rerank_top_k}" if rerank_top_k != 10 else ""
    print(f"\n{'='*70}")
    print(f"🧪 ÉVALUATION RAG-DPO — {len(dataset)} questions [{emb_label}{topk_label}]")
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
    topk_label = f" [top_k={rerank_top_k}]" if rerank_top_k != 10 else ""
    print(f"\u23f3 Initialisation du pipeline RAG ({emb_label}, {dual_label}{agent_label}{topk_label})...")
    pipeline = init_pipeline(enable_dual_gen=enable_dual_gen, use_agent=use_agent, rerank_top_k=rerank_top_k)
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
            
            # Injecter l'intent du pipeline pour le scoring conciseness
            intent_str = "factuel"
            if hasattr(response, 'intent') and response.intent:
                intent_obj = response.intent
                if hasattr(intent_obj, 'intent'):
                    intent_str = intent_obj.intent
                elif isinstance(intent_obj, str):
                    intent_str = intent_obj
            qa_item["_intent"] = intent_str
            
            # Évaluer (semantic sim + keyword, LLM judge en phase 2)
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
            
            sem_display = f"Sem:{ac.get('semantic_similarity', 0):.0%}" if ac.get('semantic_similarity') else "Sem:N/A"
            intent_display = f"[{intent_str}]" if intent_str != "factuel" else ""
            
            print(f"   {icon} Score: {global_score:.0%}  "
                  f"(KW:{ac['score']:.0%} {sem_display} | "
                  f"Faith:{ff['score']:.0%} | "
                  f"Conc:{cc['score']:.0%} | "
                  f"Src:{sq['score']:.0%})  "
                  f"[{elapsed:.1f}s, {eval_result['answer_length_words']}w] {intent_display}")
            
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
                kw_score = r["answer_correctness"].get("keyword_score", r["answer_correctness"]["score"])
                sem_score = r["answer_correctness"].get("semantic_similarity")
                
                # Dual sémantique si dispo, sinon LLM seul (keyword tracé uniquement)
                if sem_score is not None and sem_score >= 0:
                    combined = round(0.60 * llm_score + 0.40 * sem_score, 2)
                else:
                    combined = round(llm_score, 2)
                
                r["answer_correctness"]["llm_judge_score"] = round(llm_score, 2)
                r["answer_correctness"]["llm_justification"] = llm_justification
                r["answer_correctness"]["keyword_score"] = round(kw_score, 2)
                r["answer_correctness"]["score"] = combined
                
                # Recalculer le score global (v4 : conciseness désactivée)
                r["global_score"] = round(
                    0.55 * combined +
                    0.25 * r["faithfulness"]["score"] +
                    0.00 * r["conciseness"]["score"] +
                    0.20 * r["source_quality"]["score"],
                    2
                )
                
                icon = "🟢" if r["global_score"] >= 0.8 else "🟡" if r["global_score"] >= 0.5 else "🔴"
                sem_disp = f" Sem={sem_score:.0%}" if sem_score is not None and sem_score >= 0 else ""
                print(f"{icon} LLM={llm_score:.0%}{sem_disp} → {combined:.0%} (global={r['global_score']:.0%}) [KW={kw_score:.0%} tracé]")
                if verbose:
                    print(f"      📝 {llm_justification}")
            else:
                print(f"⚠️  Erreur — fallback keyword")
        
        print()
    
    # Transférer _raw_answer → answer pour sauvegarde (auditabilité du judge)
    for r in results:
        raw = r.pop("_raw_answer", None)
        if raw is not None:
            r["answer"] = raw
    # Nettoyer les _intent injectés dans le dataset
    for q in dataset:
        q.pop("_intent", None)
    
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
        
        # Score pondéré par catégorie (moyenne des moyennes de catégories)
        cat_averages = {}
        weighted_by_category = 0.0
        for cat in sorted(categories):
            cat_results_list = [r for r in valid_results if r["category"] == cat]
            cat_averages[cat] = sum(r["global_score"] for r in cat_results_list) / len(cat_results_list)
        if cat_averages:
            weighted_by_category = sum(cat_averages.values()) / len(cat_averages)
            print(f"\n  🎯 Score pondéré par catégorie : {weighted_by_category:.0%}")
            print(f"     (moyenne des moyennes — chaque catégorie pèse autant)")
        
        # Top 3 pires
        sorted_results = sorted(valid_results, key=lambda r: r["global_score"])
        print(f"\n  🔴 Points faibles (3 pires) :")
        for r in sorted_results[:3]:
            print(f"    • {r['id']:35s} : {r['global_score']:.0%} — {r['question'][:60]}")
        
        # Top 3 meilleurs
        print(f"\n  🟢 Points forts (3 meilleurs) :")
        for r in sorted_results[-3:]:
            print(f"    • {r['id']:35s} : {r['global_score']:.0%} — {r['question'][:60]}")
        
        # Corpus gap analysis — identifier les questions où le RAG manque de contenu
        # Exclure hors_perimetre et piège : pas de source = comportement attendu
        corpus_gap_threshold = 0.75
        expected_no_source_cats = {"hors_perimetre", "piège", "piege"}
        weak_questions = [r for r in valid_results if r["global_score"] < corpus_gap_threshold]
        if weak_questions:
            print(f"\n  🔍 Corpus Gap Analysis (questions < {corpus_gap_threshold:.0%}) :")
            for r in sorted(weak_questions, key=lambda x: x["global_score"]):
                cat = r.get("category", "")
                llm_s = r.get("answer_correctness", {}).get("llm_judge_score", "?")
                kw_s = r.get("answer_correctness", {}).get("keyword_score", "?")
                sem_s = r.get("answer_correctness", {}).get("semantic_similarity", "?")
                missing_kw = r.get("answer_correctness", {}).get("missing", [])
                src_count = r.get("source_quality", {}).get("count", 0)
                diag = []
                if isinstance(llm_s, (int, float)) and llm_s < 0.5:
                    diag.append("LLM juge réponse insuffisante")
                if isinstance(kw_s, (int, float)) and kw_s < 0.5:
                    diag.append(f"mots-clés manquants: {missing_kw[:3]}")
                if isinstance(sem_s, (int, float)) and sem_s < 0.5:
                    diag.append("faible similarité sémantique")
                if src_count == 0 and cat not in expected_no_source_cats:
                    diag.append("aucune source trouvée")
                if cat in expected_no_source_cats:
                    diag.append(f"[{cat}] score bas attendu")
                diag_str = " | ".join(diag) if diag else "vérifier corpus"
                qid_short = r['id'][:35]
                score_pct = r['global_score']
                print(f"    • {qid_short:35s} {score_pct:.0%} → {diag_str}")
    
    if results:
        errors = [r for r in results if "error" in r]
        if errors:
            print(f"\n  ❌ {len(errors)} erreur(s) :")
            for r in errors:
                print(f"    • {r['id']} : {r['error'][:100]}")
    
    # Sauvegarder les résultats
    gen_suffix = "single" if not enable_dual_gen else "dual"
    agent_suffix = "_agent" if use_agent else ""
    topk_suffix = f"_topk{rerank_top_k}" if rerank_top_k != 10 else ""
    output_path = project_root / "eval" / f"results_bge-m3_{gen_suffix}{agent_suffix}{topk_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "embedding_model": "BGE-M3",
            "generation_mode": "single" if not enable_dual_gen else "dual",
            "pipeline_mode": "agent" if use_agent else "native",
            "n_questions": len(dataset),
            "n_valid": len(valid_results),
            "n_errors": len(results) - len(valid_results),
            "avg_global_score": round(avg_global, 3) if valid_results else 0,
            "avg_global_score_by_category": round(weighted_by_category, 3) if cat_averages else 0,
            "avg_time_per_question": round(avg_time, 1) if valid_results else 0,
            "rerank_top_k": rerank_top_k,
            "scoring_version": "v7_free_score",
            "scoring_weights": {"correctness": 0.55, "faithfulness": 0.25, "conciseness": 0.00, "sources": 0.20},
            "correctness_formula": "0.60*llm_judge + 0.40*semantic_sim (keyword traced only)",
            "judge_mode": "free 0-100 (raw score, no discretization)",
            "conciseness_mode": "disabled (traced only)",
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

def run_multi_evaluation(
    n_runs: int,
    dataset_path: str,
    question_ids: Optional[List[str]] = None,
    verbose: bool = False,
    use_llm_judge: bool = True,
    enable_dual_gen: bool = False,
    use_agent: bool = False,
    rerank_top_k: int = 10,
) -> Dict:
    """
    Exécute N runs d'évaluation et agrège les résultats avec moyenne ± écart-type.
    
    Approche scientifiquement correcte : temp=0 ne garantit pas le déterminisme
    (GPU float arithmetic, batch scheduling CUDA, LLM-as-judge variance).
    Moyenner sur N runs donne un score robuste et un intervalle de confiance.
    
    Args:
        n_runs: Nombre de runs (≥2)
        Autres: identiques à run_evaluation
    
    Returns:
        Dict avec scores moyens, écart-types, et détails par run
    """
    print(f"\n{'='*70}")
    topk_label = f" [top_k={rerank_top_k}]" if rerank_top_k != 10 else ""
    print(f"🔄 MULTI-RUN EVALUATION — {n_runs} runs{topk_label}")
    print(f"{'='*70}\n")
    
    all_runs = []       # Liste de dicts retournés par run_evaluation
    all_results = []    # Liste de listes de résultats par question
    run_output_paths = []
    total_start = time.time()
    
    for run_idx in range(n_runs):
        print(f"\n{'━'*70}")
        print(f"  🔁 RUN {run_idx + 1}/{n_runs}")
        print(f"{'━'*70}")
        
        result = run_evaluation(
            dataset_path=dataset_path,
            question_ids=question_ids,
            dry_run=False,
            verbose=verbose if run_idx == 0 else False,  # Verbose seulement au 1er run
            use_llm_judge=use_llm_judge,
            enable_dual_gen=enable_dual_gen,
            use_agent=use_agent,
            rerank_top_k=rerank_top_k,
        )
        
        all_runs.append(result)
        all_results.append(result.get("results", []))
        if result.get("output_path"):
            run_output_paths.append(result["output_path"])
    
    total_elapsed = time.time() - total_start
    
    # ═══════════════════════════════════════════════════════════
    # Agrégation : moyenne ± σ par question et global
    # ═══════════════════════════════════════════════════════════
    
    # Collecter les scores par question
    qid_scores = {}  # qid → {"global": [s1, s2, ...], "correctness": [...], ...}
    
    for run_results in all_results:
        for r in run_results:
            if "error" in r:
                continue
            qid = r.get("id", "?")
            if qid not in qid_scores:
                qid_scores[qid] = {
                    "global": [], "correctness": [], "faithfulness": [],
                    "sources": [], "conciseness": [], "words": [],
                    "time": [], "llm_judge": [], "semantic_sim": [], "keyword": [],
                    "question": r.get("question", ""),
                    "category": r.get("category", ""),
                }
            s = qid_scores[qid]
            s["global"].append(r.get("global_score", 0))
            s["correctness"].append(r.get("answer_correctness", {}).get("score", 0))
            s["faithfulness"].append(r.get("faithfulness", {}).get("score", 0))
            s["sources"].append(r.get("source_quality", {}).get("score", 0))
            s["conciseness"].append(r.get("conciseness", {}).get("score", 0))
            s["words"].append(r.get("answer_length_words", 0))
            s["time"].append(r.get("elapsed_seconds", 0))
            s["llm_judge"].append(r.get("answer_correctness", {}).get("llm_judge_score", 0))
            s["semantic_sim"].append(r.get("answer_correctness", {}).get("semantic_similarity", 0))
            s["keyword"].append(r.get("answer_correctness", {}).get("keyword_score", 0))
            if r.get("answer"):
                s.setdefault("answers", []).append(r["answer"])
    
    # Scores globaux agrégés
    all_globals = [r.get("avg_global", 0) for r in all_runs]
    mean_global = statistics.mean(all_globals)
    std_global = statistics.stdev(all_globals) if len(all_globals) >= 2 else 0
    
    # ═══════════════════════════════════════════════════════════
    # Rapport Multi-Run
    # ═══════════════════════════════════════════════════════════
    
    print(f"\n{'='*70}")
    print(f"📊 RAPPORT MULTI-RUN — {n_runs} runs")
    print(f"{'='*70}\n")
    
    print(f"  📈 Score Global : {mean_global:.1%} ± {std_global:.1%}")
    
    # Score pondéré par catégorie (chaque catégorie pèse autant)
    cat_means_multi = {}
    for qid, s in qid_scores.items():
        cat = s["category"]
        if cat not in cat_means_multi:
            cat_means_multi[cat] = []
        cat_means_multi[cat].append(statistics.mean(s["global"]))
    if cat_means_multi:
        cat_avgs_multi = {c: statistics.mean(scores) for c, scores in cat_means_multi.items()}
        weighted_by_cat = sum(cat_avgs_multi.values()) / len(cat_avgs_multi)
        print(f"  🎯 Score pondéré/catégorie : {weighted_by_cat:.1%}")
        for cat in sorted(cat_avgs_multi.keys()):
            n = len(cat_means_multi[cat])
            icon = "🟢" if cat_avgs_multi[cat] >= 0.8 else "🟡" if cat_avgs_multi[cat] >= 0.5 else "🔴"
            print(f"     {icon} {cat:20s} : {cat_avgs_multi[cat]:.0%} ({n}q)")
    
    print(f"  ⏱️  Temps total  : {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  🔁 Runs         : {', '.join(f'{g:.1%}' for g in all_globals)}")
    
    # Tableau par question
    print(f"\n  {'Question':<35} {'Moyenne':>8} {'±σ':>6} {'Min':>6} {'Max':>6} {'Spread':>7} {'Mots':>10}")
    print(f"  {'-'*35} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*10}")
    
    high_variance_questions = []
    
    for qid in sorted(qid_scores.keys()):
        s = qid_scores[qid]
        g = s["global"]
        
        if len(g) < 2:
            continue
        
        m = statistics.mean(g)
        sd = statistics.stdev(g)
        mn = min(g)
        mx = max(g)
        spread = mx - mn
        
        w = s["words"]
        w_mean = statistics.mean(w)
        w_sd = statistics.stdev(w) if len(w) >= 2 else 0
        
        # Flag variance haute
        flag = ""
        if spread >= 0.10:
            flag = " ⚠️"
            high_variance_questions.append((qid, m, sd, spread, w_mean, w_sd))
        
        short = qid[:33] + ".." if len(qid) > 35 else qid
        print(f"  {short:<35} {m:>7.1%} {sd:>5.1%} {mn:>5.0%} {mx:>5.0%} {spread:>6.0%} {w_mean:>5.0f}±{w_sd:>3.0f}w{flag}")
    
    # Résumé variance
    all_spreads = [max(s["global"]) - min(s["global"]) for s in qid_scores.values() if len(s["global"]) >= 2]
    avg_spread = statistics.mean(all_spreads) if all_spreads else 0
    max_spread = max(all_spreads) if all_spreads else 0
    
    print(f"\n  📉 Variance inter-runs :")
    print(f"     Spread moyen  : {avg_spread:.1%} (écart min-max par question)")
    print(f"     Spread max    : {max_spread:.1%}")
    print(f"     σ global      : {std_global:.1%}")
    
    if high_variance_questions:
        print(f"\n  ⚠️  {len(high_variance_questions)} question(s) avec spread ≥10% :")
        for qid, m, sd, spread, w_mean, w_sd in high_variance_questions:
            print(f"     • {qid}: {m:.0%} ± {sd:.0%} (spread {spread:.0%}, {w_mean:.0f}±{w_sd:.0f} mots)")
    else:
        print(f"\n  ✅ Aucune question avec spread ≥10% — résultats stables")
    
    # Conclusion
    print(f"\n  {'─'*60}")
    if std_global < 0.01:
        print(f"  ✅ Résultats TRÈS STABLES (σ < 1%) — 1 run suffirait")
    elif std_global < 0.02:
        print(f"  ✅ Résultats STABLES (σ < 2%) — fiable pour comparaisons")
    elif std_global < 0.05:
        print(f"  ⚠️  Variance MODÉRÉE (σ = {std_global:.1%}) — moyenner 3+ runs pour comparer")
    else:
        print(f"  🔴 Variance ÉLEVÉE (σ = {std_global:.1%}) — résultats non fiables sans 5+ runs")
    
    # Sauvegarder le rapport agrégé
    gen_suffix = "single" if not enable_dual_gen else "dual"
    agent_suffix = "_agent" if use_agent else ""
    topk_suffix = f"_topk{rerank_top_k}" if rerank_top_k != 10 else ""
    output_path = project_root / "eval" / f"results_bge-m3_{gen_suffix}{agent_suffix}{topk_suffix}_{n_runs}runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Construire résultats agrégés par question
    aggregated_results = []
    for qid in sorted(qid_scores.keys()):
        s = qid_scores[qid]
        g = s["global"]
        entry = {
            "id": qid,
            "question": s["question"],
            "category": s["category"],
            "n_runs": len(g),
            "global_score": {"mean": round(statistics.mean(g), 3), "std": round(statistics.stdev(g), 3) if len(g) >= 2 else 0, "min": round(min(g), 3), "max": round(max(g), 3), "runs": [round(x, 3) for x in g]},
            "correctness":  {"mean": round(statistics.mean(s["correctness"]), 3), "std": round(statistics.stdev(s["correctness"]), 3) if len(s["correctness"]) >= 2 else 0, "runs": [round(x, 3) for x in s["correctness"]]},
            "faithfulness":  {"mean": round(statistics.mean(s["faithfulness"]), 3), "std": round(statistics.stdev(s["faithfulness"]), 3) if len(s["faithfulness"]) >= 2 else 0, "runs": [round(x, 3) for x in s["faithfulness"]]},
            "sources":       {"mean": round(statistics.mean(s["sources"]), 3), "std": round(statistics.stdev(s["sources"]), 3) if len(s["sources"]) >= 2 else 0, "runs": [round(x, 3) for x in s["sources"]]},
            "llm_judge":     {"mean": round(statistics.mean(s["llm_judge"]), 3), "std": round(statistics.stdev(s["llm_judge"]), 3) if len(s["llm_judge"]) >= 2 else 0, "runs": [round(x, 3) for x in s["llm_judge"]]},
            "semantic_sim":  {"mean": round(statistics.mean(s["semantic_sim"]), 3), "std": round(statistics.stdev(s["semantic_sim"]), 3) if len(s["semantic_sim"]) >= 2 else 0, "runs": [round(x, 3) for x in s["semantic_sim"]]},
            "keyword":       {"mean": round(statistics.mean(s["keyword"]), 3), "std": round(statistics.stdev(s["keyword"]), 3) if len(s["keyword"]) >= 2 else 0, "runs": [round(x, 3) for x in s["keyword"]]},
            "answer_length_words": {"mean": round(statistics.mean(s["words"]), 0), "std": round(statistics.stdev(s["words"]), 0) if len(s["words"]) >= 2 else 0, "runs": [int(x) for x in s["words"]]},
        }
        # Ajouter les réponses RAG brutes pour auditabilité
        if s.get("answers"):
            entry["answers"] = s["answers"]
        aggregated_results.append(entry)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "type": "multi_run_aggregated",
            "timestamp": datetime.now().isoformat(),
            "n_runs": n_runs,
            "embedding_model": "BGE-M3",
            "generation_mode": "single" if not enable_dual_gen else "dual",
            "pipeline_mode": "agent" if use_agent else "native",
            "rerank_top_k": rerank_top_k,
            "scoring_version": "v7_free_score",
            "global_score": {"mean": round(mean_global, 3), "std": round(std_global, 3), "runs": [round(x, 3) for x in all_globals]},
            "global_score_by_category": {cat: round(avg, 3) for cat, avg in cat_avgs_multi.items()} if cat_means_multi else {},
            "global_score_weighted_by_category": round(weighted_by_cat, 3) if cat_means_multi else 0,
            "variance_analysis": {
                "avg_spread_per_question": round(avg_spread, 3),
                "max_spread": round(max_spread, 3),
                "high_variance_questions": [q[0] for q in high_variance_questions],
            },
            "total_time_seconds": round(total_elapsed, 0),
            "individual_run_files": run_output_paths,
            "results": aggregated_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Rapport agrégé : {output_path}")
    for i, p in enumerate(run_output_paths, 1):
        print(f"   Run {i}: {p}")
    print(f"{'='*70}\n")
    
    return {
        "mean_global": mean_global,
        "std_global": std_global,
        "n_runs": n_runs,
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Évaluation automatique du RAG-DPO")
    parser.add_argument("--ids", nargs="+", help="IDs des questions à évaluer (ex: q01 q05)")
    parser.add_argument("--dry-run", action="store_true", help="Affiche les questions sans exécuter")
    parser.add_argument("--verbose", action="store_true", help="Affiche les réponses complètes")
    parser.add_argument("--no-llm-judge", action="store_true", help="Désactive le LLM-as-judge (keyword matching seul)")
    parser.add_argument("--dataset", default=None, help="Chemin vers le dataset JSON")
    parser.add_argument("--dual", action="store_true",
                        help="Active la dual-generation (single-gen par défaut, dual plus lent)")
    parser.add_argument("--agent", action="store_true",
                        help="Utilise le pipeline LangGraph agent au lieu du natif")
    parser.add_argument("--runs", type=int, default=1,
                        help="Nombre de runs à exécuter et moyenner (défaut: 1, recommandé: 3)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Nombre de chunks gardés après reranking (défaut: 10, tester: 4-15)")
    
    args = parser.parse_args()
    
    dataset_path = args.dataset or str(project_root / "eval" / "qa_dataset.json")
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset introuvable : {dataset_path}")
        sys.exit(1)
    
    if args.runs >= 2:
        run_multi_evaluation(
            n_runs=args.runs,
            dataset_path=dataset_path,
            question_ids=args.ids,
            verbose=args.verbose,
            use_llm_judge=not args.no_llm_judge,
            enable_dual_gen=args.dual,
            use_agent=args.agent,
            rerank_top_k=args.top_k,
        )
    else:
        run_evaluation(
            dataset_path=dataset_path,
            question_ids=args.ids,
            dry_run=args.dry_run,
            verbose=args.verbose,
            use_llm_judge=not args.no_llm_judge,
            enable_dual_gen=args.dual,
            use_agent=args.agent,
            rerank_top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
