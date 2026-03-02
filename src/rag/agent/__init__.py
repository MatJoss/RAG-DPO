"""
RAG Agent — Pipeline LangGraph pour le système RAG-DPO.

Architecture en graphe d'états avec nœuds spécialisés et tools locaux :

    rewrite → classify → enrich → retrieve → generate → validate ─→ check_completeness ─→ respond
                                       ↑           ↑                          │
                                       │           └── (grounding retry) ─────┤
                                       └── (completeness re-retrieve) ────────┘

Utilise les mêmes composants internes (retriever, context_builder, generator,
reranker, validators) que le pipeline natif, mais orchestrés via LangGraph.

Nœud rewrite (multi-turn) :
- Résolution d'anaphores ("et pour eux ?", "les mêmes ?", "ça")
- Reformulation en question autonome avant classification et retrieval
- Passthrough si pas d'historique de conversation

Tools locaux (100% offline, aucune donnée ne sort) :
- DateCalculator : délais RGPD (72h notification, 30j droits, etc.)
- ArticleLookup : index structuré des 99 articles du RGPD
- CompletenessChecker : vérifie la couverture de la réponse

Avantages par rapport au pipeline natif :
- Réécriture multi-turn (résolution de références conversationnelles)
- Boucle de validation (retry automatique si grounding échoue)
- Enrichissement pré-retrieval (articles RGPD, délais)
- Vérification post-generation de complétude
- Graphe visible et debuggable
- Extensible (ajout de tools, nœuds, conditions)
"""

from .graph import create_agent_pipeline, RAGAgentPipeline
from .tools import (
    RGPD_ARTICLES,
    RGPD_DEADLINES,
    calculate_deadline,
    list_deadlines,
    lookup_article,
    search_articles_by_topic,
)
from .state import RAGState

__all__ = ["create_agent_pipeline", "RAGAgentPipeline", "RAGState"]
