"""
RAG Agent Nodes — Fonctions de nœud pour le graphe LangGraph.

Chaque fonction prend le state en entrée et retourne un dict partiel
qui sera mergé dans le state. Pattern standard LangGraph.

Les nœuds réutilisent les composants existants du pipeline natif
(retriever, context_builder, generator, reranker, validators)
sans duplication de logique.
"""
import logging
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
