"""
RAG Agent Graph — Définition du StateGraph LangGraph.

Graphe d'états avec routage conditionnel :

    START → classify → retrieve → generate → validate ─→ respond → END
                                      ↑                    │
                                      └── (retry) ─────────┘

La boucle validate → generate est le vrai avantage par rapport au pipeline
natif : si le grounding échoue, on re-génère automatiquement au lieu
d'accepter une réponse dégradée.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from langgraph.graph import END, START, StateGraph

from ..context_builder import ContextBuilder, create_context_builder
from ..generator import Generator, create_generator
from ..intent_classifier import IntentClassifier
from ..pipeline import RAGResponse, build_enterprise_where_filter
from ..reranker import CrossEncoderReranker
from ..retriever import RAGRetriever, create_retriever
from ..validators import GroundingValidator
from .nodes import (
    NodeComponents,
    make_classify_node,
    make_generate_node,
    make_respond_node,
    make_retrieve_node,
    make_validate_node,
)
from .state import RAGState

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Graph Builder
# ═══════════════════════════════════════════════════════════════

def build_graph(components: NodeComponents) -> StateGraph:
    """Construit le graphe LangGraph à partir des composants.
    
    Architecture :
        classify → retrieve → generate → validate → respond
                                  ↑            │
                                  └── (retry) ─┘
    """
    graph = StateGraph(RAGState)
    
    # ── Nœuds ──
    graph.add_node("classify", make_classify_node(components))
    graph.add_node("retrieve", make_retrieve_node(components))
    graph.add_node("generate", make_generate_node(components))
    graph.add_node("validate", make_validate_node(components))
    graph.add_node("respond", make_respond_node(components))
    
    # ── Arêtes ──
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    
    # Routage conditionnel après validation
    def should_retry(state: RAGState) -> str:
        """Décide si on relance la génération ou on termine."""
        if state.get("validation_passed", True):
            return "respond"
        if state.get("retry_count", 0) > components.max_retries:
            return "respond"
        if state.get("error"):
            return "respond"
        # Retry : re-générer avec temperature plus basse
        return "generate"
    
    graph.add_conditional_edges(
        "validate",
        should_retry,
        {"respond": "respond", "generate": "generate"},
    )
    graph.add_edge("respond", END)
    
    return graph


# ═══════════════════════════════════════════════════════════════
# Agent Pipeline Wrapper
# ═══════════════════════════════════════════════════════════════

class RAGAgentPipeline:
    """Wrapper autour du graphe LangGraph compilé.
    
    Expose la même API query() que RAGPipeline pour compatibilité.
    """
    
    def __init__(self, components: NodeComponents):
        self.components = components
        graph = build_graph(components)
        self.app = graph.compile()
        logger.info("✅ [Agent] Graphe LangGraph compilé")
    
    def query(
        self,
        question: str,
        where_filter: Optional[Dict] = None,
        enterprise_tags: Optional[List[str]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> RAGResponse:
        """Exécute le graphe agent et retourne un RAGResponse compatible.
        
        Même signature que RAGPipeline.query() pour interchangeabilité.
        """
        start = time.time()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🤖 [Agent] RAG Query: {question[:100]}...")
        logger.info(f"{'='*80}\n")
        
        # État initial
        initial_state: RAGState = {
            "question": question,
            "where_filter": where_filter,
            "enterprise_tags": enterprise_tags,
            "conversation_history": conversation_history,
            "temperature": temperature,
            "start_time": start,
            "retry_count": 0,
        }
        
        # Exécuter le graphe
        try:
            final_state = self.app.invoke(initial_state)
        except Exception as e:
            logger.error(f"❌ [Agent] Erreur graphe: {e}")
            return RAGResponse(
                answer="Une erreur s'est produite lors du traitement de la question.",
                question=question,
                total_time=time.time() - start,
                error=str(e),
            )
        
        # Convertir le state final en RAGResponse
        context = final_state.get("context_used") or {}
        sources = context.get("sources_metadata", [])
        
        # Extraire les citations [Source X] de la réponse
        import re
        answer = final_state.get("answer", "")
        cited = sorted(set(int(s) for s in re.findall(r"Source\s+(\d+)", answer)))
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            cited_sources=cited,
            question=question,
            model=final_state.get("model", ""),
            retrieval_time=final_state.get("retrieval_time", 0.0),
            generation_time=final_state.get("generation_time", 0.0),
            total_time=final_state.get("total_time", time.time() - start),
            intent=final_state.get("intent"),
            error=final_state.get("error"),
        )
    
    def get_graph_image(self) -> Optional[bytes]:
        """Retourne l'image PNG du graphe (pour debug/visualisation).
        
        Nécessite graphviz et pygraphviz installés.
        Retourne None si non disponibles.
        """
        try:
            return self.app.get_graph().draw_mermaid_png()
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════
# Factory Function
# ═══════════════════════════════════════════════════════════════

def create_agent_pipeline(
    collection,
    llm_provider,
    embedding_provider=None,
    n_documents: int = 5,
    n_chunks_per_doc: int = 3,
    max_context_length: int = 32000,
    model: str = "mistral-nemo",
    temperature: float = 0.0,
    max_tokens: int = 2000,
    enable_validation: bool = True,
    enable_hybrid: bool = True,
    enable_reranker: bool = True,
    enable_summary_prefilter: bool = True,
    enable_query_expansion: bool = True,
    summaries_path: Optional[str] = None,
    rerank_candidates: int = 40,
    rerank_top_k: int = 10,
    max_retries: int = 1,
) -> RAGAgentPipeline:
    """Factory function pour créer un agent pipeline LangGraph.
    
    Même interface que create_pipeline() du pipeline natif.
    Réutilise les mêmes composants internes.
    """
    from pathlib import Path

    from ..bm25_index import ChunkBM25Index, SummaryBM25Index
    from ..query_expander import QueryExpander

    init_start = time.time()
    
    # ── Query Expander ──
    query_expander = None
    if enable_query_expansion:
        query_expander = QueryExpander(
            llm_provider=llm_provider,
            enabled=True,
            n_expansions=3,
            temperature=0.7,
            max_tokens=300,
        )
    
    # ── BM25 Indexes ──
    summary_bm25 = None
    chunk_bm25 = None
    
    if enable_hybrid or enable_summary_prefilter:
        if summaries_path is None:
            default_path = Path(__file__).parent.parent.parent.parent / "data" / "keep" / "cnil" / "document_summaries.json"
            if default_path.exists():
                summaries_path = str(default_path)
        
        if enable_summary_prefilter and summaries_path:
            summary_bm25 = SummaryBM25Index()
            summary_bm25.build(summaries_path)
        
        if enable_hybrid:
            chunk_bm25 = ChunkBM25Index()
            chunk_bm25.build_from_collection(collection)
    
    # ── Retriever ──
    retriever = create_retriever(
        collection=collection,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        summary_bm25_index=summary_bm25,
        chunk_bm25_index=chunk_bm25,
        query_expander=query_expander,
        n_documents=n_documents,
        n_chunks_per_doc=n_chunks_per_doc,
        summary_prefilter_k=40,
        enable_hybrid=enable_hybrid,
        enable_summary_prefilter=enable_summary_prefilter,
    )
    
    # ── Reranker ──
    reranker = None
    if enable_reranker:
        reranker = CrossEncoderReranker(
            device="cpu",
            batch_size=32,
            trust_remote_code=True,
        )
    
    # ── Context Builder ──
    context_builder = create_context_builder(
        max_context_length=max_context_length,
        include_metadata=True,
        llm_provider=llm_provider,
    )
    
    # ── Generator ──
    generator = create_generator(
        llm_provider=llm_provider,
        model=model,
        temperature=temperature,
    )
    
    # ── Intent Classifier ──
    intent_classifier = IntentClassifier(llm_provider=llm_provider)
    
    # ── Grounding Validator ──
    grounding_validator = None
    if enable_validation:
        grounding_validator = GroundingValidator(llm_provider)
    
    # ── Assemble ──
    components = NodeComponents(
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
        intent_classifier=intent_classifier,
        reranker=reranker,
        grounding_validator=grounding_validator,
        rerank_candidates=rerank_candidates,
        rerank_top_k=rerank_top_k,
        max_retries=max_retries,
    )
    
    init_time = time.time() - init_start
    logger.info(
        f"✅ [Agent] Pipeline LangGraph initialisé en {init_time:.1f}s "
        f"(nodes: classify→retrieve→generate→validate→respond, "
        f"retry_loop={max_retries}x)"
    )
    
    return RAGAgentPipeline(components)
