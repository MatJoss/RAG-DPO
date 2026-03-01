"""
RAG Agent — Pipeline LangGraph pour le système RAG-DPO.

Architecture en graphe d'états avec nœuds spécialisés :

    classify → retrieve → rerank → generate → validate ─→ respond
                                                   │
                                                   └── (retry) ──→ generate

Utilise les mêmes composants internes (retriever, context_builder, generator,
reranker, validators) que le pipeline natif, mais orchestrés via LangGraph.

Avantages par rapport au pipeline natif :
- Boucle de validation (retry automatique si grounding échoue)
- Graphe visible et debuggable
- Extensible (ajout de tools, nœuds, conditions)
- Architecture agent standard (LangGraph/LangChain écosystème)
"""

from .graph import create_agent_pipeline, RAGAgentPipeline
from .state import RAGState

__all__ = ["create_agent_pipeline", "RAGAgentPipeline", "RAGState"]
