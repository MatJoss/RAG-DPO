"""
RAG Agent Nodes — Fonctions de nœud pour le graphe LangGraph.

Chaque fonction prend le state en entrée et retourne un dict partiel
qui sera mergé dans le state. Pattern standard LangGraph.

Les nœuds réutilisent les composants existants du pipeline natif
(retriever, context_builder, generator, reranker, validators)
sans duplication de logique.
"""
import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..context_builder import ContextBuilder
from ..generator import Generator
from ..intent_classifier import IntentClassifier, QuestionIntent
from ..pipeline import build_enterprise_where_filter
from ..reranker import CrossEncoderReranker, RankedChunk
from ..retriever import RAGRetriever, RetrievedChunk, RetrievedDocument
from ..validators import GroundingValidator
from .state import RAGState
from .tools import (
    RGPD_ARTICLES,
    RGPD_DEADLINES,
    calculate_deadline,
    check_answer_completeness,
    decompose_question,
    lookup_article,
    search_articles_by_topic,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Container pour les composants partagés entre nœuds
# ═══════════════════════════════════════════════════════════════

class NodeComponents:
    """Conteneur pour les composants injectés dans les nœuds.
    
    LangGraph nodes sont des fonctions pures (state → partial state).
    On injecte les composants via closure (voir graph.py).
    """
    def __init__(
        self,
        retriever: RAGRetriever,
        context_builder: ContextBuilder,
        generator: Generator,
        intent_classifier: Optional[IntentClassifier] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        grounding_validator: Optional[GroundingValidator] = None,
        rerank_candidates: int = 40,
        rerank_top_k: int = 10,
        max_retries: int = 1,
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator
        self.intent_classifier = intent_classifier
        self.reranker = reranker
        self.grounding_validator = grounding_validator
        self.rerank_candidates = rerank_candidates
        self.rerank_top_k = rerank_top_k
        self.max_retries = max_retries


# ═══════════════════════════════════════════════════════════════
# Node functions
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Rewrite node (multi-turn)
# ═══════════════════════════════════════════════════════════════

REWRITE_PROMPT = """Réécris la QUESTION ACTUELLE en une question autonome et complète.

Règles STRICTES :
1. Résous les pronoms, "ça", "eux", "les mêmes", "et pour X ?" en utilisant l'HISTORIQUE
2. La question réécrite doit être compréhensible SANS l'historique
3. Conserve le français et le sens exact — ne change RIEN d'autre
4. Ne RÉPONDS PAS à la question — tu la REFORMULES seulement
5. N'invente AUCUNE information absente de l'historique
6. Retourne UNIQUEMENT la question reformulée, sans explication ni guillemets
7. Si la question est déjà autonome et claire, retourne-la EXACTEMENT telle quelle

HISTORIQUE :
{history}

QUESTION ACTUELLE : {question}

QUESTION REFORMULÉE :"""


def make_rewrite_node(components: NodeComponents):
    """Crée le nœud de réécriture multi-turn.
    
    Si l'historique de conversation est présent, reformule la question
    en une question autonome (résolution d'anaphores, de pronoms, etc.).
    Sinon, passe la question telle quelle.
    
    Position dans le graphe : START → **rewrite** → classify
    """
    
    def rewrite(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        conversation_history = state.get("conversation_history")
        
        # Pas d'historique → pas de réécriture
        if not conversation_history:
            return {"original_question": question}
        
        # Filtrer : au moins un échange user+assistant
        user_msgs = [m for m in conversation_history if m.get("role") == "user"]
        if not user_msgs:
            return {"original_question": question}
        
        logger.info(f"✏️  [Agent] Phase -1 : Query Rewrite (multi-turn, {len(conversation_history)} messages)")
        
        # Formater l'historique (5 derniers échanges max)
        history_lines = []
        for msg in conversation_history[-10:]:  # 5 échanges = 10 messages max
            role = "Utilisateur" if msg["role"] == "user" else "Assistant"
            # Tronquer les réponses longues pour ne pas exploser le prompt
            content = msg["content"]
            if role == "Assistant" and len(content) > 500:
                content = content[:500] + "..."
            history_lines.append(f"{role}: {content}")
        
        history_str = "\n".join(history_lines)
        
        # Appel LLM pour réécriture
        prompt = REWRITE_PROMPT.format(history=history_str, question=question)
        
        try:
            rewritten = components.generator.llm_provider.chat(
                messages=[
                    {"role": "system", "content": "Tu réécris des questions pour les rendre autonomes."},
                    {"role": "user", "content": prompt},
                ],
                model=components.generator.model,
                temperature=0.0,
                max_tokens=200,
            ).strip()
            
            # Nettoyage basique
            # Enlever les guillemets encadrants si présents
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]
            
            # Vérification : la réécriture ne doit pas être vide ou absurde
            if not rewritten or len(rewritten) < 5:
                logger.warning(f"   ⚠️ Réécriture trop courte, on garde l'originale")
                return {"original_question": question}
            
            if rewritten.lower() == question.lower():
                logger.info(f"   → Question déjà autonome, pas de changement")
                return {"original_question": question}
            
            logger.info(f"   → Originale : {question}")
            logger.info(f"   → Réécrite  : {rewritten}")
            
            return {
                "original_question": question,
                "question": rewritten,
            }
            
        except Exception as e:
            logger.warning(f"   ⚠️ Erreur réécriture: {e}, on garde l'originale")
            return {"original_question": question}
    
    return rewrite


# ═══════════════════════════════════════════════════════════════
# Classify node
# ═══════════════════════════════════════════════════════════════

def make_classify_node(components: NodeComponents):
    """Crée le nœud de classification d'intention."""
    
    def classify(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        logger.info(f"🎯 [Agent] Phase 0 : Intent Classification")
        
        if components.intent_classifier is None:
            return {"intent": None}
        
        intent = components.intent_classifier.classify(question)
        logger.info(
            f"   → intent={intent.intent}, international={intent.scope_international}, "
            f"methodology={intent.needs_methodology}, topics={intent.topics}"
        )
        return {"intent": intent}
    
    return classify


# ── Réponses de refus déterministes (pas de LLM → 100% stable) ──

_REFUSAL_ILLEGAL = (
    "Je refuse de répondre à cette question. Cette demande est **directement illégale** "
    "et **hors périmètre RGPD/CNIL** couvert par mes sources.\n\n"
    "Pour rappel, ce type d'action expose à des **sanctions pénales** :\n"
    "- **Accès frauduleux à un système informatique** (art. 323-1 Code pénal) : "
    "jusqu'à 3 ans d'emprisonnement et 100 000 € d'amende\n"
    "- **Collecte frauduleuse de données personnelles** (art. 226-18 Code pénal) : "
    "jusqu'à 5 ans d'emprisonnement et 300 000 € d'amende\n"
    "- **Détournement de finalité** (art. 226-21 Code pénal) : "
    "jusqu'à 5 ans d'emprisonnement et 300 000 € d'amende\n\n"
    "Ces infractions sont poursuivies d'office et ne relèvent en aucun cas "
    "de l'accompagnement d'un DPO."
)

_REFUSAL_CONTOURNEMENT = (
    "Je refuse de répondre à cette question. Chercher à **contourner, éviter ou esquiver** "
    "une obligation légale est contraire au RGPD.\n\n"
    "Les obligations RGPD sont des **obligations légales impératives**. "
    "Leur non-respect expose à des **sanctions administratives** pouvant atteindre :\n"
    "- **20 millions d'euros** ou **4 % du chiffre d'affaires annuel mondial** "
    "(art. 83 RGPD), le montant le plus élevé étant retenu.\n\n"
    "En tant qu'assistant RGPD, je ne propose aucune alternative ni astuce "
    "pour contourner ces obligations."
)

_REFUSAL_HORS_PERIMETRE = (
    "Cette question ne relève pas du périmètre RGPD/CNIL couvert par mes sources. "
    "Je suis un assistant spécialisé en **protection des données personnelles** "
    "et ne peux répondre qu'aux questions relevant de ce domaine."
)

# Mots-clés pour distinguer les sous-types de refus
_ILLEGAL_KEYWORDS = {
    "pirater", "hacker", "cracker", "piratage", "hacking",
    "voler des données", "vol de données", "espionner",
    "usurper", "usurpation", "accès frauduleux",
    "ddos", "phishing", "ransomware", "malware",
}

_CONTOURNEMENT_KEYWORDS = {
    "contourner", "éviter", "esquiver", "échapper",
    "ne pas respecter", "ignorer l'obligation",
    "sans se faire prendre", "sans être sanctionné",
}


def _classify_refusal_type(question: str) -> str:
    """Détermine le sous-type de refus à partir de la question."""
    q_lower = question.lower()
    for kw in _ILLEGAL_KEYWORDS:
        if kw in q_lower:
            return "illegal"
    for kw in _CONTOURNEMENT_KEYWORDS:
        if kw in q_lower:
            return "contournement"
    return "hors_perimetre"


def make_refusal_node(components: NodeComponents):
    """Crée le nœud de refus déterministe (court-circuit, pas de LLM)."""

    def refusal(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        refusal_type = _classify_refusal_type(question)

        if refusal_type == "illegal":
            answer = _REFUSAL_ILLEGAL
        elif refusal_type == "contournement":
            answer = _REFUSAL_CONTOURNEMENT
        else:
            answer = _REFUSAL_HORS_PERIMETRE

        logger.info(
            f"🚫 [Agent] Refus déterministe ({refusal_type}) — "
            f"court-circuit retrieve/generate"
        )

        return {
            "answer": answer,
            "documents": [],
            "retrieval_time": 0.0,
            "generation_time": 0.0,
            "model": "deterministic_refusal",
        }

    return refusal


def make_retrieve_node(components: NodeComponents):
    """Crée le nœud de retrieval (hybrid search + optional reranking)."""
    
    def retrieve(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        where_filter = state.get("where_filter")
        enterprise_tags = state.get("enterprise_tags")
        
        # Si on est en mode sous-question (re-retrieve ciblé), utiliser la suggested_query
        completeness = state.get("completeness")
        if completeness and not completeness.get("is_complete", True):
            suggested = completeness.get("suggested_queries", [])
            if suggested:
                question = suggested[0]
                logger.info(f"📥 [Agent] Phase 1 : Re-Retrieval ciblé: {question[:80]}...")
            else:
                logger.info(f"📥 [Agent] Phase 1 : Re-Retrieval (question originale)")
        else:
            logger.info(f"📥 [Agent] Phase 1 : Retrieval")
        
        # Construire le filtre
        effective_filter = build_enterprise_where_filter(where_filter, enterprise_tags)
        
        retrieval_start = time.time()
        
        if components.reranker is not None:
            # Flow reranker : récupérer des candidats bruts puis reranker
            raw_candidates = components.retriever.retrieve_candidates(
                query=question,
                n_candidates=components.rerank_candidates,
                where_filter=effective_filter,
            )
            
            if not raw_candidates:
                retrieval_time = time.time() - retrieval_start
                return {
                    "documents": [],
                    "retrieval_time": retrieval_time,
                    "error": "No candidates found",
                }
            
            logger.info(f"   {len(raw_candidates)} candidats bruts récupérés")
            
            # Reranking
            ranked_chunks = components.reranker.rerank(
                query=question,
                chunks=raw_candidates,
                top_k=components.rerank_top_k,
            )
            
            # Reconstruire documents depuis chunks reranked
            documents = _rebuild_documents_from_ranked_chunks(ranked_chunks)
            
            retrieval_time = time.time() - retrieval_start
            total_chunks = sum(len(d.chunks) for d in documents)
            logger.info(
                f"   ✅ Reranking: {len(raw_candidates)} → {total_chunks} chunks, "
                f"{len(documents)} docs en {retrieval_time:.2f}s"
            )
        else:
            # Flow classique sans reranker
            documents = components.retriever.retrieve(
                query=question,
                where_filter=effective_filter,
            )
            retrieval_time = time.time() - retrieval_start
            logger.info(f"   ✅ {len(documents)} docs en {retrieval_time:.2f}s")
        
        return {
            "documents": documents,
            "retrieval_time": retrieval_time,
        }
    
    return retrieve


def make_generate_node(components: NodeComponents):
    """Crée le nœud de génération (context building + LLM call)."""
    
    def generate(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        documents = state.get("documents", [])
        intent = state.get("intent")
        conversation_history = state.get("conversation_history")
        temperature = state.get("temperature")
        tool_results = state.get("tool_results")
        
        if not documents:
            return {
                "answer": "Je n'ai pas trouvé d'information pertinente dans ma base de connaissance pour répondre à cette question.",
                "generation_time": 0.0,
                "error": "No documents to generate from",
            }
        
        logger.info(f"🏗️  [Agent] Phase 3 : Context Building + Generation")
        
        # Context building (single pass, pas de dual-gen)
        context = components.context_builder.build_context(
            documents=documents,
            question=question,
            conversation_history=conversation_history,
            intent=intent,
        )
        
        # ── Injection des tool_results dans le user prompt ──
        # Respecte max_context_length : on ne dépasse jamais le budget
        if tool_results:
            enrichment_lines = []
            
            if "rgpd_articles" in tool_results:
                enrichment_lines.append("\n--- Références RGPD pertinentes ---")
                for art in tool_results["rgpd_articles"]:
                    enrichment_lines.append(f"• {art['titre']} : {art['description']}")
            
            if "deadlines" in tool_results:
                enrichment_lines.append("\n--- Délais réglementaires ---")
                for dl in tool_results["deadlines"]:
                    enrichment_lines.append(
                        f"• {dl['description']} : {dl['delai']} ({dl['article']})"
                    )
            
            if "topic_articles" in tool_results:
                enrichment_lines.append("\n--- Articles RGPD liés au sujet ---")
                for art in tool_results["topic_articles"][:3]:
                    enrichment_lines.append(f"• Art. {art['article']} : {art['description'][:100]}")
            
            if enrichment_lines:
                enrichment_block = "\n".join(enrichment_lines)
                max_len = components.context_builder.max_context_length
                current_len = len(context["user"])
                available = max_len - current_len - 50  # 50 chars de marge séparateur
                
                if available >= len(enrichment_block):
                    # L'enrichissement tient dans le budget → injecter tel quel
                    context["user"] = context["user"] + f"\n\n{enrichment_block}\n"
                    logger.info(f"   🔧 {len(enrichment_lines)} lignes d'enrichissement injectées ({len(enrichment_block)} chars)")
                elif available >= 200:
                    # Budget serré → tronquer l'enrichissement
                    truncated = enrichment_block[:available] + "\n[...enrichissement tronqué...]"
                    context["user"] = context["user"] + f"\n\n{truncated}\n"
                    logger.warning(f"   🔧 Enrichissement tronqué ({len(enrichment_block)} → {available} chars)")
                else:
                    # Pas assez de place → skip silencieux
                    logger.warning(f"   ⚠️  Enrichissement skippé (contexte déjà {current_len}/{max_len} chars)")
        
        logger.info(f"   Context: {len(context['user'])} chars (max {components.context_builder.max_context_length})")
        
        # Generation
        generation_start = time.time()
        generated = components.generator.generate(
            system_prompt=context["system"],
            user_prompt=context["user"],
            temperature=temperature,
        )
        generation_time = time.time() - generation_start
        
        if generated.error:
            logger.error(f"   ❌ Erreur génération: {generated.error}")
            return {
                "answer": "Une erreur s'est produite lors de la génération de la réponse.",
                "context_used": context,
                "generation_time": generation_time,
                "model": generated.model,
                "error": generated.error,
            }
        
        logger.info(f"   ✅ Réponse: {len(generated.text)} chars en {generation_time:.2f}s")
        
        # Strip section "Sources" / "Références" si le LLM en ajoute une
        # (les sources sont affichées par l'UI, pas besoin de les répéter dans le texte)
        answer_text = generated.text.strip()
        _src_section = re.search(
            r'\n\s*(?:Sources|Références|Liste des sources)\s*:\s*\n',
            answer_text, re.IGNORECASE,
        )
        if _src_section:
            answer_text = answer_text[:_src_section.start()].rstrip()
            logger.info(f"   🧹 Section Sources supprimée du texte généré")
        
        return {
            "answer": answer_text,
            "context_used": context,
            "generation_time": generation_time,
            "model": generated.model,
        }
    
    return generate


# ═══════════════════════════════════════════════════════════════
# Expert Refinement node
# ═══════════════════════════════════════════════════════════════

# Prompt unique de polishing contraint — ne restructure PAS, polit seulement.
# Contrairement à l'ancien "expert rewriting" (4 prompts par intent),
# ce prompt interdit tout changement d'ordre, de structure ou de livrables.
STRUCTURAL_POLISHING_PROMPT = """Polis cette réponse RGPD pour un DPO senior. Tu NE RESTRUCTURES PAS, tu POLIS.

RÈGLES ABSOLUES — toute violation = rejet :
1. NE CHANGE PAS l'ordre des sections ou des étapes
2. NE SUPPRIME aucun livrable, aucune référence normative (article, décret, délibération)
3. NE SUPPRIME et N'AJOUTE aucune information factuelle
4. Conserve TOUS les [Source N] exactement où ils sont
5. NE RÉORDONNE PAS les points numérotés

CE QUE TU PEUX FAIRE :
- Supprimer les phrases introductives pédagogiques ("Le RGPD est un règlement qui...", "Il est important de noter que...")
- Supprimer les conclusions génériques ("En suivant ces étapes, vous...", "N'hésitez pas à...")
- Supprimer les répétitions exactes d'une même information
- Clarifier les formulations ambiguës en les rendant plus directes
- Supprimer les définitions évidentes pour un professionnel

RÉPONSE À POLIR :
{answer}

RÉPONSE POLIE :"""

# Intents qui ne bénéficient PAS du polishing
# (listes = déjà structurées, comparaisons = déjà binaires, refus = court)
SKIP_REFINEMENT_INTENTS = {"liste_exhaustive", "comparaison", "refus"}


def make_expert_refinement_node(components: NodeComponents):
    """Crée le nœud de polishing structurel contraint post-génération.
    
    Polit la réponse brute du LLM sans la restructurer :
    - Supprime le ton pédagogique ("le RGPD est un règlement qui...")
    - Supprime les intros/conclusions génériques et les répétitions
    - NE change PAS l'ordre des sections, les livrables, les refs normatives
    - Conserve 100% des faits et des citations [Source N]
    
    Position dans le graphe : generate → **expert_refinement** → validate
    
    Coût : ~3-5s, même modèle, 0 VRAM supplémentaire.
    Ne s'applique PAS aux intents liste_exhaustive, comparaison, refus.
    """
    
    def expert_refinement(state: RAGState) -> Dict[str, Any]:
        import re
        
        answer = state.get("answer", "")
        intent = state.get("intent")
        error = state.get("error")
        
        # Skip si erreur ou réponse trop courte
        if error or len(answer) < 100:
            return {}
        
        # Skip pour certains intents (déjà bien structurés)
        intent_type = intent.intent if intent else "factuel"
        if intent_type in SKIP_REFINEMENT_INTENTS:
            logger.info(f"🎓 [Agent] Phase 3b : Expert Refinement — skip (intent={intent_type})")
            return {}
        
        # Skip si la réponse n'a AUCUNE citation [Source N]
        # → pas la peine de reformuler un texte déjà non-groundé, ça le gonfle inutilement
        original_sources = set(re.findall(r'\[Source \d+\]', answer))
        if not original_sources:
            logger.info(f"🎓 [Agent] Phase 3b : Expert Refinement — skip (aucune source dans la réponse)")
            return {}
        
        logger.info(f"🎓 [Agent] Phase 3b : Structural Polishing (intent={intent_type})")
        
        prompt = STRUCTURAL_POLISHING_PROMPT.format(answer=answer)
        
        try:
            refinement_start = time.time()
            refined = components.generator.llm_provider.chat(
                messages=[
                    {"role": "system", "content": "Tu polis des réponses RGPD pour des DPO seniors. Tu ne changes JAMAIS le fond NI la structure. Tu ne fais que supprimer le bruit rédactionnel."},
                    {"role": "user", "content": prompt},
                ],
                model=components.generator.model,
                temperature=0.0,
                max_tokens=3000,
            ).strip()
            refinement_time = time.time() - refinement_start
            
            # Vérifications de sécurité
            if not refined or len(refined) < 50:
                logger.warning(f"   ⚠️ Polishing trop court ({len(refined)} chars), on garde l'original")
                return {}
            
            # Vérifier que les [Source N] sont conservées
            refined_sources = set(re.findall(r'\[Source \d+\]', refined))
            lost_sources = original_sources - refined_sources
            
            if lost_sources:
                logger.warning(
                    f"   ⚠️ Polishing a perdu {len(lost_sources)} source(s): {lost_sources}, on garde l'original"
                )
                return {}
            
            # Vérifier que la réponse n'a pas explosé en taille (signe d'hallucination)
            # Double garde-fou : ratio relatif (1.8x) ET seuil absolu (+500 chars)
            max_allowed = min(len(answer) * 1.8, len(answer) + 500)
            if len(refined) > max_allowed:
                logger.warning(
                    f"   ⚠️ Polishing trop long ({len(refined)} vs {len(answer)} chars, max={int(max_allowed)}), on garde l'original"
                )
                return {}
            
            logger.info(
                f"   ✅ Polishing: {len(answer)} → {len(refined)} chars "
                f"({refined_sources} sources conservées) en {refinement_time:.1f}s"
            )
            
            return {"answer": refined}
            
        except Exception as e:
            logger.warning(f"   ⚠️ Erreur polishing: {e}, on garde l'original")
            return {}
    
    return expert_refinement


def make_validate_node(components: NodeComponents):
    """Crée le nœud de validation grounding."""
    
    def validate(state: RAGState) -> Dict[str, Any]:
        answer = state.get("answer", "")
        context = state.get("context_used")
        retry_count = state.get("retry_count", 0)
        
        # Skip pour les questions composites (grounding fait par construction dans chaque sous-question)
        sub_questions = state.get("sub_questions")
        if sub_questions and len(sub_questions) > 1:
            logger.info(f"🔍 [Agent] Phase 4 : Validation Grounding — skip (question composite, {len(sub_questions)} sous-réponses)")
            return {"validation_passed": True, "validation_reason": "Composite — grounded individually"}
        
        # Pas de validator → passer
        if components.grounding_validator is None or context is None:
            return {"validation_passed": True, "validation_reason": "No validator"}
        
        logger.info(f"🔍 [Agent] Phase 4 : Validation Grounding (retry={retry_count})")
        
        available_sources = [s["id"] for s in context.get("sources_metadata", [])]
        validation = components.grounding_validator.validate_response(
            response=answer,
            available_sources=available_sources,
            context=context["user"],
        )
        
        if validation.is_valid:
            logger.info(f"   ✅ Grounding validé (score={validation.score:.2f})")
            return {
                "validation_passed": True,
                "validation_reason": "OK",
            }
        
        # Grounding failed
        logger.warning(f"   ⚠️  Grounding: {validation.reason}")
        
        # Hallucination sévère → pas de retry, rejeter
        if "hallucination" in validation.reason.lower():
            import re
            n_issues = validation.reason.count(" ; ") + 1
            if n_issues >= 3:
                logger.error(f"   ❌ Hallucination sévère ({n_issues} problèmes)")
                return {
                    "validation_passed": False,
                    "validation_reason": f"Hallucination sévère: {validation.reason}",
                    "answer": "Je n'ai pas trouvé suffisamment d'informations fiables dans mes sources pour répondre précisément à cette question.",
                    "error": f"Hallucination detected: {validation.reason}",
                }
        
        # Sources inventées → nettoyer
        if "inventées" in validation.reason.lower():
            fixed = components.grounding_validator.fix_invented_sources(
                response=answer,
                available_sources=available_sources,
            )
            return {
                "answer": fixed,
                "validation_passed": True,
                "validation_reason": f"Fixed: {validation.reason}",
            }
        
        # Autres problèmes → retry si possible
        can_retry = retry_count < components.max_retries
        return {
            "validation_passed": not can_retry,  # passer si plus de retry
            "validation_reason": validation.reason,
            "retry_count": retry_count + 1,
        }
    
    return validate


def make_respond_node(components: NodeComponents):
    """Crée le nœud final qui assemble la RAGResponse."""
    
    def respond(state: RAGState) -> Dict[str, Any]:
        """Assemble les résultats finaux."""
        total_time = time.time() - state.get("start_time", time.time())
        
        original = state.get("original_question")
        question = state.get("question", "")
        rewritten = original and original != question
        
        logger.info(
            f"📝 [Agent] Réponse finale: "
            f"{len(state.get('answer', ''))} chars, "
            f"{total_time:.1f}s total"
            f"{' (question réécrite)' if rewritten else ''}"
        )
        
        return {"total_time": total_time}
    
    return respond


def make_enrich_node(components: NodeComponents):
    """Crée le nœud d'enrichissement par tools locaux.
    
    Détecte automatiquement les articles RGPD et délais mentionnés
    dans la question, et injecte les résultats dans le state.
    
    Position dans le graphe : classify → **enrich** → retrieve
    Cela permet d'enrichir la recherche avec des infos structurées.
    """
    
    def enrich(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        intent = state.get("intent")
        
        logger.info(f"🔧 [Agent] Phase 0b : Tool Enrichment")
        
        tool_results: Dict[str, Any] = {}
        
        # ── 1. Détection d'articles RGPD ──
        # Patterns: "article 33", "art. 33", "art 33", "l'article 33"
        article_matches = re.findall(
            r"(?:l['''])?art(?:icle)?\.?\s*(\d{1,2})",
            question,
            re.IGNORECASE,
        )
        if article_matches:
            articles = []
            for match in set(article_matches):
                num = int(match)
                result = lookup_article(num)
                if result.get("found"):
                    articles.append(result)
            if articles:
                tool_results["rgpd_articles"] = articles
                logger.info(f"   📖 {len(articles)} article(s) RGPD trouvé(s): {[a['article'] for a in articles]}")
        
        # ── 2. Détection de délais / deadlines ──
        deadline_keywords = {
            "72h": "notification_violation",
            "72 heures": "notification_violation",
            "notification": "notification_violation",
            "violation": "notification_violation",
            "breach": "notification_violation",
            "droit d'accès": "reponse_droits",
            "demande d'accès": "reponse_droits",
            "exercice des droits": "reponse_droits",
            "droit de rectification": "reponse_droits",
            "droit à l'effacement": "reponse_droits",
            "droit d'opposition": "reponse_droits",
            "conservation": "conservation_cv",
            "vidéosurveillance": "conservation_videosurveillance",
            "vidéo": "conservation_videosurveillance",
            "registre": "tenue_registre",
            "aipd": "aipd_avant_mise_en_oeuvre",
            "dpia": "aipd_avant_mise_en_oeuvre",
            "analyse d'impact": "aipd_avant_mise_en_oeuvre",
        }
        
        q_lower = question.lower()
        detected_deadlines = set()
        for keyword, deadline_type in deadline_keywords.items():
            if keyword in q_lower:
                detected_deadlines.add(deadline_type)
        
        if detected_deadlines:
            deadlines = []
            for dt in detected_deadlines:
                result = calculate_deadline(dt)
                deadlines.append({
                    "type": result.deadline_type,
                    "description": result.description,
                    "article": result.article,
                    "delai": f"{RGPD_DEADLINES[dt]['delai']} {RGPD_DEADLINES[dt]['unite']}" 
                             if RGPD_DEADLINES[dt]['delai'] else RGPD_DEADLINES[dt]['unite'],
                    "note": result.note,
                })
            tool_results["deadlines"] = deadlines
            logger.info(f"   ⏰ {len(deadlines)} délai(s) RGPD détecté(s)")
        
        # ── 3. Recherche thématique d'articles ──
        # Si l'intent porte sur un sujet spécifique mais sans numéro d'article
        if not article_matches and intent:
            for topic in (intent.topics or []):
                topic_articles = search_articles_by_topic(topic)
                if topic_articles:
                    tool_results.setdefault("topic_articles", []).extend(topic_articles[:3])
            
            if tool_results.get("topic_articles"):
                # Dédupliquer par numéro d'article
                seen = set()
                unique = []
                for a in tool_results["topic_articles"]:
                    if a["article"] not in seen:
                        seen.add(a["article"])
                        unique.append(a)
                tool_results["topic_articles"] = unique[:5]
                logger.info(f"   🏷️  {len(tool_results['topic_articles'])} article(s) par topic")
        
        # ── 4. Garde-fou confusions sémantiques connues ──
        # Règles déterministes pour compléter negative_topics
        # quand le classifier LLM ne les détecte pas.
        #
        # Principe : on formalise les frontières juridiques du RGPD
        # entre concepts sémantiquement proches mais procéduralement distincts.
        # Ce n'est pas du biais — c'est de la cohérence normative.
        #
        # Critères d'inclusion : la confusion doit changer la méthodologie,
        # pas juste être une différence lexicale.
        CONFUSION_GUARDS = {
            # ── AIPD (art. 35) : risque pour les personnes, processus formel ──
            "aipd": {
                "negative_topics": [
                    "aitd", "transfert international", "pays tiers",
                    "clauses contractuelles types",     # AITD ≠ AIPD
                    "registre des traitements",         # art.30 ≠ art.35 (description ≠ analyse de risque)
                    "analyse de risque SSI",            # risque orga ≠ risque personnes
                    "violation de données",             # ex post ≠ ex ante
                ],
                "guard_keywords": [
                    "transfert", "pays tiers", "international", "aitd", "clause",
                    "registre", "ebios", "iso 27005", "ssi",
                    "violation", "breach", "fuite",
                ],
            },
            # ── AITD (art. 46) : transferts internationaux ──
            "aitd": {
                "negative_topics": [
                    "aipd", "analyse d'impact relative", "risque élevé",
                ],
                "guard_keywords": ["aipd", "risque élevé", "analyse d'impact relative"],
            },
            # ── Violation de données (art. 33-34) : ex post ──
            "violation": {
                "negative_topics": ["aipd", "analyse d'impact"],
                "guard_keywords": ["aipd", "analyse d'impact", "dpia"],
            },
            # ── Registre (art. 30) : obligation documentaire ──
            "registre": {
                "negative_topics": ["aipd", "analyse d'impact"],
                "guard_keywords": ["aipd", "analyse d'impact", "dpia", "risque élevé"],
            },
            # ── Sous-traitant (art. 28) vs Responsable de traitement (art. 4) ──
            "sous_traitant": {
                "negative_topics": ["responsable de traitement"],
                "guard_keywords": ["responsable de traitement", "responsable du traitement"],
            },
            "responsable_traitement": {
                "negative_topics": ["sous-traitant"],
                "guard_keywords": ["sous-traitant", "sous traitant", "prestataire"],
            },
            # ── Base légale (art. 6) vs Finalité (art. 5) ──
            "base_legale": {
                "negative_topics": ["finalité du traitement"],
                "guard_keywords": ["finalité", "objectif du traitement"],
            },
        }
        
        if intent:
            for topic in (intent.topics or []):
                guard = CONFUSION_GUARDS.get(topic)
                if guard:
                    # Ne PAS ajouter les negative_topics si la question mentionne explicitement le sujet confusable
                    mentions_confusable = any(kw in q_lower for kw in guard["guard_keywords"])
                    if not mentions_confusable:
                        existing = set(intent.negative_topics or [])
                        added = [t for t in guard["negative_topics"] if t not in existing]
                        if added:
                            intent.negative_topics = list(existing) + added
                            logger.info(f"   🛡️  Garde-fou confusion: +{added} pour topic '{topic}'")
        
        if tool_results:
            logger.info(f"   ✅ Enrichissement: {list(tool_results.keys())}")
        else:
            logger.info(f"   ∅ Aucun enrichissement applicable")
        
        return {"tool_results": tool_results if tool_results else None, "intent": intent}
    
    return enrich


# ═══════════════════════════════════════════════════════════════
# Decompose node (query decomposition)
# ═══════════════════════════════════════════════════════════════

# Plus de MERGE_PROMPT LLM — fusion programmatique pure (0 appel LLM, 0 perte de citation)
# Le LLM 12B supprime systématiquement les [Source N] quand on lui demande de fusionner.


def make_decompose_node(components: NodeComponents):
    """Crée le nœud de décomposition de question composite.
    
    Détecte si la question couvre plusieurs aspects distincts.
    Si oui : retrieval unique + génération unique structurée par sous-questions.
    Si non : passe la question telle quelle au retrieve normal.
    
    Architecture (v2) :
    - 1 retrieval sur la question globale (au lieu de N)
    - 1 génération unique avec les sous-questions comme structure de prompt
    - Résultat parsé en sections pour les expanders UI
    
    Position dans le graphe : enrich → **decompose** → retrieve (simple) ou respond (composite)
    """
    
    def decompose(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        intent = state.get("intent")
        where_filter = state.get("where_filter")
        enterprise_tags = state.get("enterprise_tags")
        conversation_history = state.get("conversation_history")
        temperature = state.get("temperature")
        tool_results = state.get("tool_results")
        
        logger.info(f"🔀 [Agent] Phase 0c : Query Decomposition")
        
        # Décomposer la question
        sub_questions = decompose_question(question, components.generator.llm_provider)
        
        # Question simple → passer directement au retrieve normal
        if len(sub_questions) <= 1:
            logger.info(f"   → Question simple, pas de décomposition")
            return {"sub_questions": None}
        
        logger.info(f"   → {len(sub_questions)} sous-questions détectées")
        
        # ── Retrieval UNIQUE sur la question globale ──
        effective_filter = build_enterprise_where_filter(where_filter, enterprise_tags)
        retrieval_start = time.time()
        
        # Pool large pour couvrir tous les aspects
        global_top_k = min(components.rerank_top_k, 10)
        
        if components.reranker is not None:
            raw_candidates = components.retriever.retrieve_candidates(
                query=question,
                n_candidates=components.rerank_candidates,
                where_filter=effective_filter,
            )
            
            if raw_candidates:
                ranked_chunks = components.reranker.rerank(
                    query=question,
                    chunks=raw_candidates,
                    top_k=global_top_k,
                )
                all_documents = _rebuild_documents_from_ranked_chunks(ranked_chunks)
            else:
                all_documents = []
        else:
            all_documents = components.retriever.retrieve(
                query=question,
                where_filter=effective_filter,
            )
        
        total_retrieval_time = time.time() - retrieval_start
        logger.info(f"   ✅ Retrieval unique: {len(all_documents)} docs en {total_retrieval_time:.1f}s")
        
        if not all_documents:
            logger.warning(f"   ⚠️  Aucun document trouvé")
            return {"sub_questions": None}
        
        # ── Génération UNIQUE avec sous-questions comme structure ──
        # Au lieu de N générations séparées (N × 35s), on fait 1 seule génération
        # où les sous-questions structurent le prompt → réponse cohérente, sans répétition.
        COMPOSITE_MAX_TOKENS = 2000  # Budget pour la réponse complète structurée
        
        # Construire le contexte avec la question globale
        context = components.context_builder.build_context(
            documents=all_documents,
            question=question,
            conversation_history=conversation_history,
            intent=intent,
        )
        
        # Injecter tool_results si disponibles
        if tool_results:
            enrichment_lines = []
            if "rgpd_articles" in tool_results:
                enrichment_lines.append("\n--- Références RGPD pertinentes ---")
                for art in tool_results["rgpd_articles"]:
                    enrichment_lines.append(f"• {art['titre']} : {art['description']}")
            if "deadlines" in tool_results:
                enrichment_lines.append("\n--- Délais réglementaires ---")
                for dl in tool_results["deadlines"]:
                    enrichment_lines.append(
                        f"• {dl['description']} : {dl['delai']} ({dl['article']})"
                    )
            if enrichment_lines:
                enrichment_block = "\n".join(enrichment_lines)
                max_len = components.context_builder.max_context_length
                available = max_len - len(context["user"]) - 50
                if available >= len(enrichment_block):
                    context["user"] = context["user"] + f"\n\n{enrichment_block}\n"
        
        # Construire l'instruction de structure multi-sections
        sections_instruction = "\n".join(
            f"### {i}. {sq}" for i, sq in enumerate(sub_questions, 1)
        )
        
        composite_instruction = (
            f"La question couvre {len(sub_questions)} aspects distincts. "
            f"Structure ta réponse en {len(sub_questions)} sections Markdown avec EXACTEMENT ces titres :\n\n"
            f"{sections_instruction}\n\n"
            "RÈGLES CRITIQUES :\n"
            "- Chaque section traite UNIQUEMENT son aspect spécifique\n"
            "- NE RÉPÈTE PAS la même information entre sections (ex: méthodologie AIPD = une seule fois)\n"
            "- Chaque section cite ses sources [Source N] après chaque fait\n"
            "- Réponse totale : 1500-2500 caractères maximum\n"
            "- Ne génère PAS de section 'Sources' ou 'Références' en fin de réponse"
        )
        
        # Remplacer la question et l'instruction dans le user prompt
        context["user"] = context["user"].replace(
            question,
            f"{question}\n\n{composite_instruction}",
        )
        
        logger.info(f"   🚀 Génération unique structurée ({len(sub_questions)} sections, max_tokens={COMPOSITE_MAX_TOKENS})")
        
        gen_start = time.time()
        generated = components.generator.generate(
            system_prompt=context["system"],
            user_prompt=context["user"],
            temperature=temperature,
            max_tokens=COMPOSITE_MAX_TOKENS,
        )
        total_generation_time = time.time() - gen_start
        
        if generated.error:
            logger.error(f"   ❌ Erreur génération composite: {generated.error}")
            return {"sub_questions": None}
        
        answer_text = generated.text.strip()
        logger.info(f"   ✅ Réponse composite: {len(answer_text)} chars en {total_generation_time:.1f}s")
        
        # ── Strip section "Sources" / "Références" en fin de réponse ──
        _sources_section_re = re.compile(
            r'\n\s*(?:Sources|Références|Liste des sources)\s*:\s*\n.*',
            re.DOTALL | re.IGNORECASE,
        )
        cleaned = _sources_section_re.sub('', answer_text).rstrip()
        if len(cleaned) < len(answer_text):
            logger.info(f"   🧹 Section Sources supprimée ({len(answer_text) - len(cleaned)} chars)")
            answer_text = cleaned
        
        # ── Parser la réponse en sous-réponses pour les expanders UI ──
        # On cherche les sections ### pour extraire chaque sous-réponse
        sub_answers = _parse_sections(answer_text, sub_questions)
        
        # ── Compacter les IDs de sources ──
        def _renumber_sources(text: str, mapping: Dict[int, int]) -> str:
            if not mapping:
                return text
            def _replacer(m):
                old_id = int(m.group(1))
                return f"[Source {mapping.get(old_id, old_id)}]"
            return re.sub(r'\[Source\s+(\d+)\]', _replacer, text)
        
        unique_documents = all_documents
        cited_ids = sorted(set(int(s) for s in re.findall(r'\[Source\s+(\d+)\]', answer_text)))
        uncited_ids = [i for i in range(1, len(unique_documents) + 1) if i not in cited_ids]
        
        compact_mapping = {}
        new_idx = 1
        for old_id in cited_ids:
            compact_mapping[old_id] = new_idx
            new_idx += 1
        for old_id in uncited_ids:
            compact_mapping[old_id] = new_idx
            new_idx += 1
        
        has_gaps = any(old != new for old, new in compact_mapping.items())
        if has_gaps:
            answer_text = _renumber_sources(answer_text, compact_mapping)
            sub_answers = [_renumber_sources(sa, compact_mapping) for sa in sub_answers]
            reordered = [None] * len(unique_documents)
            for old_id, new_id in compact_mapping.items():
                reordered[new_id - 1] = unique_documents[old_id - 1]
            unique_documents = reordered
            n_compacted = sum(1 for o, n in compact_mapping.items() if o != n)
            logger.info(f"   🔢 Compactage: {n_compacted} source(s) renumérotée(s)")
        
        # ── Construire sources_metadata pour l'UI ──
        sources_metadata = components.context_builder._extract_sources_metadata(unique_documents)
        merged_context = {
            "system": "",
            "user": "",
            "sources_metadata": sources_metadata,
        }
        
        return {
            "sub_questions": sub_questions,
            "sub_answers": sub_answers,
            "answer": answer_text,
            "documents": unique_documents,
            "context_used": merged_context,
            "retrieval_time": total_retrieval_time,
            "generation_time": total_generation_time,
            "model": components.generator.model,
            "validation_passed": True,
            "validation_reason": "Composite — single generation with structured prompt",
        }
    
    return decompose


def _parse_sections(text: str, sub_questions: List[str]) -> List[str]:
    """Parse une réponse structurée en sections ### pour extraire les sous-réponses.
    
    Cherche les headers ### (numérotés ou non) et découpe le texte.
    Fallback : si le parsing échoue, retourne la réponse complète pour chaque section.
    """
    # Trouver tous les headers ### dans le texte
    header_pattern = re.compile(r'^###\s+', re.MULTILINE)
    matches = list(header_pattern.finditer(text))
    
    if len(matches) < 2:
        # Pas assez de sections trouvées → fallback
        return [text] * len(sub_questions)
    
    # Extraire le contenu entre chaque header
    sections = []
    for i, match in enumerate(matches):
        # Trouver la fin de la ligne du header
        header_end = text.find('\n', match.start())
        if header_end == -1:
            header_end = len(text)
        
        # Contenu = entre fin du header et début du header suivant (ou fin du texte)
        content_start = header_end + 1
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        sections.append(content)
    
    # Si on a plus de sections que de sous-questions, tronquer
    # Si on en a moins, compléter avec des vides
    while len(sections) < len(sub_questions):
        sections.append("")
    
    return sections[:len(sub_questions)]


def make_check_completeness_node(components: NodeComponents):
    """Crée le nœud de vérification de complétude post-validation.
    
    Après validation grounding, vérifie si la réponse couvre tous
    les aspects de la question. Si non, identifie les aspects
    manquants pour un éventuel re-retrieval ciblé.
    
    Position : validate → **check_completeness** → respond/retrieve
    Activé uniquement si retry_count < max_retries.
    """
    
    def check_completeness(state: RAGState) -> Dict[str, Any]:
        answer = state.get("answer", "")
        question = state["question"]
        retry_count = state.get("retry_count", 0)
        validation_passed = state.get("validation_passed", True)
        
        # Skip si validation échouée (le retry loop gère ça)
        if not validation_passed:
            return {}
        
        # Skip si déjà en retry (éviter boucle infinie)
        if retry_count >= components.max_retries:
            return {"completeness": {"is_complete": True, "coverage_pct": 100}}
        
        # Skip les réponses d'erreur
        if len(answer) < 50:
            return {"completeness": {"is_complete": True, "coverage_pct": 100}}
        
        logger.info(f"🔎 [Agent] Phase 5 : Completeness Check")
        
        result = check_answer_completeness(
            question=question,
            answer=answer,
            llm_provider=components.generator.llm_provider,
        )
        
        if result["is_complete"] or result.get("coverage_pct", 100) >= 80:
            logger.info(f"   ✅ Réponse complète ({result.get('coverage_pct', 100)}%)")
        else:
            logger.warning(
                f"   ⚠️  Couverture {result.get('coverage_pct', 0)}% — "
                f"Manquant: {result.get('missing_aspects', [])}"
            )
        
        # Incrémenter retry_count pour éviter boucle infinie sur re-retrieval
        new_retry = retry_count + 1 if not (result["is_complete"] or result.get("coverage_pct", 100) >= 80) else retry_count
        
        return {"completeness": result, "retry_count": new_retry}
    
    return check_completeness


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _rebuild_documents_from_ranked_chunks(
    ranked_chunks: List[RankedChunk],
) -> List[RetrievedDocument]:
    """Reconstruit des RetrievedDocument à partir de chunks reranked.
    
    Identique à RAGPipeline._rebuild_documents_from_ranked_chunks
    mais en standalone pour les nœuds LangGraph.
    """
    from collections import defaultdict
    
    doc_chunks: Dict[str, List[RankedChunk]] = defaultdict(list)
    doc_best_score: Dict[str, float] = {}
    
    for rc in ranked_chunks:
        doc_chunks[rc.document_path].append(rc)
        if rc.document_path not in doc_best_score or rc.rerank_score > doc_best_score[rc.document_path]:
            doc_best_score[rc.document_path] = rc.rerank_score
    
    sorted_paths = sorted(doc_best_score.keys(), key=lambda p: doc_best_score[p], reverse=True)
    
    documents = []
    for doc_path in sorted_paths:
        chunks = doc_chunks[doc_path]
        chunks.sort(key=lambda x: x.rerank_score, reverse=True)
        
        converted = []
        for rc in chunks:
            converted.append(RetrievedChunk(
                chunk_id=rc.chunk_id,
                text=rc.text,
                document_path=rc.document_path,
                chunk_nature=rc.metadata.get("chunk_nature", "UNKNOWN"),
                chunk_index=rc.metadata.get("chunk_index", 0),
                confidence=rc.metadata.get("confidence", "medium"),
                distance=1.0 - rc.rerank_score,
                metadata=rc.metadata,
                hybrid_score=rc.rerank_score,
            ))
        
        natures = [rc.metadata.get("chunk_nature", "UNKNOWN") for rc in chunks]
        primary_nature = max(set(natures), key=natures.count) if natures else "UNKNOWN"
        
        documents.append(RetrievedDocument(
            document_path=doc_path,
            chunks=converted,
            avg_similarity=doc_best_score[doc_path],
            primary_nature=primary_nature,
        ))
    
    return documents
