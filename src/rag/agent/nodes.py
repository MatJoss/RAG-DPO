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


def make_retrieve_node(components: NodeComponents):
    """Crée le nœud de retrieval (hybrid search + optional reranking)."""
    
    def retrieve(state: RAGState) -> Dict[str, Any]:
        question = state["question"]
        where_filter = state.get("where_filter")
        enterprise_tags = state.get("enterprise_tags")
        
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
                context["user"] = context["user"] + f"\n\n{enrichment_block}\n"
                logger.info(f"   🔧 {len(enrichment_lines)} lignes d'enrichissement injectées")
        
        logger.info(f"   Context: {len(context['user'])} chars")
        
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
        
        return {
            "answer": generated.text.strip(),
            "context_used": context,
            "generation_time": generation_time,
            "model": generated.model,
        }
    
    return generate


def make_validate_node(components: NodeComponents):
    """Crée le nœud de validation grounding."""
    
    def validate(state: RAGState) -> Dict[str, Any]:
        answer = state.get("answer", "")
        context = state.get("context_used")
        retry_count = state.get("retry_count", 0)
        
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
        
        logger.info(
            f"📝 [Agent] Réponse finale: "
            f"{len(state.get('answer', ''))} chars, "
            f"{total_time:.1f}s total"
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
        
        if tool_results:
            logger.info(f"   ✅ Enrichissement: {list(tool_results.keys())}")
        else:
            logger.info(f"   ∅ Aucun enrichissement applicable")
        
        return {"tool_results": tool_results if tool_results else None}
    
    return enrich


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
        
        return {"completeness": result}
    
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
