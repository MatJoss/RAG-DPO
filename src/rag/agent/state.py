"""
RAG Agent State — Schéma d'état typé pour le graphe LangGraph.

L'état circule entre les nœuds du graphe. Chaque nœud lit et écrit
des champs spécifiques. TypedDict garantit la cohérence statique.
"""
from typing import Any, Dict, List, Optional, TypedDict

from ..intent_classifier import QuestionIntent
from ..retriever import RetrievedDocument


class RAGState(TypedDict, total=False):
    """État complet du pipeline RAG agent.
    
    Chaque nœud lit/écrit des champs spécifiques :
    - classify  : question → intent
    - retrieve  : question, intent → documents, retrieval_time
    - rerank    : question, documents → documents (rerankés)
    - generate  : question, documents, intent → answer, context_used
    - validate  : answer, context_used → validation_passed, validation_reason
    - respond   : * → response (final)
    """
    # ── Input ──
    question: str
    where_filter: Optional[Dict]
    enterprise_tags: Optional[List[str]]
    conversation_history: Optional[List[Dict[str, str]]]
    temperature: Optional[float]
    
    # ── Intent ──
    intent: Optional[QuestionIntent]
    
    # ── Retrieval ──
    documents: List[RetrievedDocument]
    retrieval_time: float
    
    # ── Generation ──
    answer: str
    context_used: Optional[Dict[str, Any]]  # system + user prompts + sources_metadata
    generation_time: float
    model: str
    
    # ── Validation ──
    validation_passed: bool
    validation_reason: str
    retry_count: int
    
    # ── Timing ──
    start_time: float
    total_time: float
    
    # ── Error ──
    error: Optional[str]
