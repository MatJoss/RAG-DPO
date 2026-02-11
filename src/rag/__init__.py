"""
src/rag package - Retrieval Augmented Generation pour DPO

Architecture hybride :
- BM25 (sparse) + Semantic (dense) + RRF fusion
- Summary pre-filtering
- Cross-encoder reranking
- Reverse repacking
- LLM-based validation (pertinence + grounding)
"""

from .retriever import RAGRetriever, RetrievedChunk, RetrievedDocument, create_retriever
from .context_builder import ContextBuilder, create_context_builder
from .generator import Generator, GeneratedResponse, create_generator
from .pipeline import RAGPipeline, RAGResponse, create_pipeline
from .bm25_index import SummaryBM25Index, ChunkBM25Index
from .reranker import CrossEncoderReranker, RankedChunk

__all__ = [
    # Retriever
    'RAGRetriever',
    'RetrievedChunk',
    'RetrievedDocument',
    'create_retriever',
    
    # BM25 Indexes
    'SummaryBM25Index',
    'ChunkBM25Index',
    
    # Reranker
    'CrossEncoderReranker',
    'RankedChunk',
    
    # Context Builder
    'ContextBuilder',
    'create_context_builder',
    
    # Generator
    'Generator',
    'GeneratedResponse',
    'create_generator',
    
    # Pipeline
    'RAGPipeline',
    'RAGResponse',
    'create_pipeline',
]
