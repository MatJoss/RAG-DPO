"""
RAG Retriever - Hybrid Retrieval (Semantic + BM25) avec pre-filtering par summaries

Architecture :
1. Query Enhancement : expansion acronymes + LLM query expansion (multi-query)
2. Summary Pre-Filter : BM25 sur les fiches synthétiques → top-N documents pertinents
3. Multi-Query Hybrid Retrieval : BM25 (sparse) + ChromaDB (dense) × N queries
4. Reciprocal Rank Fusion : fusion des résultats sparse + dense + multi-query
5. Déduplication par document avec normalisation URL
"""
import logging
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict

from src.utils.acronyms import expand_query_with_acronyms
from src.rag.query_expander import QueryExpander

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Chunk récupéré avec métadonnées et score"""
    chunk_id: str
    text: str
    document_path: str
    chunk_nature: str
    chunk_index: int
    confidence: str
    distance: float  # Distance ChromaDB (plus bas = plus similaire)
    metadata: Dict[str, Any]
    
    # Scores hybrid
    bm25_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    
    @property
    def similarity_score(self) -> float:
        """Convertit distance en score de similarité (0-1, plus haut = meilleur)"""
        return 1.0 / (1.0 + self.distance)


@dataclass
class RetrievedDocument:
    """Document avec ses chunks associés"""
    document_path: str
    chunks: List[RetrievedChunk]
    avg_similarity: float
    primary_nature: str
    
    def __post_init__(self):
        if self.chunks:
            self.avg_similarity = sum(c.similarity_score for c in self.chunks) / len(self.chunks)
        else:
            self.avg_similarity = 0.0
        
        if self.chunks:
            natures = [c.chunk_nature for c in self.chunks]
            self.primary_nature = max(set(natures), key=natures.count)
        else:
            self.primary_nature = "UNKNOWN"


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60,
    weights: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Reciprocal Rank Fusion (RRF) pour combiner plusieurs rankings.
    
    Score RRF = sum(weight_i / (k + rank_i)) pour chaque ranking.
    k=60 est le standard (Cormack et al., 2009).
    
    Args:
        rankings: Liste de rankings (chaque ranking = liste d'IDs ordonnés)
        k: Constante RRF (défaut 60)
        weights: Poids par ranking (défaut: tous à 1.0). Permet de donner
                 plus d'importance à certains rankings (ex: query originale ×2)
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    
    scores = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] += weight / (k + rank + 1)
    return dict(scores)


class RAGRetriever:
    """
    Retriever hybride : Semantic (ChromaDB) + BM25 + Summary Pre-filtering
    
    Stratégie :
    1. Expansion acronymes dans la query
    2. BM25 sur summaries → identifie les ~20 documents pertinents
    3. Semantic search ChromaDB (dense) → top chunks
    4. BM25 sur chunks (sparse) → top chunks
    5. RRF fusion des deux rankings
    6. Déduplication par document
    7. Sélection top N documents avec top K chunks chacun
    """
    
    def __init__(
        self,
        collection,
        llm_provider,
        summary_bm25_index=None,
        chunk_bm25_index=None,
        query_expander: Optional[QueryExpander] = None,
        n_documents: int = 5,
        n_chunks_per_doc: int = 3,
        fetch_multiplier: int = 10,
        summary_prefilter_k: int = 20,
        enable_hybrid: bool = True,
        enable_summary_prefilter: bool = True,
    ):
        """
        Args:
            collection: Collection ChromaDB
            llm_provider: Provider pour embeddings
            summary_bm25_index: Index BM25 sur les summaries (pré-filtrage)
            chunk_bm25_index: Index BM25 sur les chunks (sparse retrieval)
            query_expander: QueryExpander pour multi-query retrieval (optionnel)
            n_documents: Nombre de documents uniques à retourner
            n_chunks_per_doc: Nombre max de chunks par document
            fetch_multiplier: Multiplicateur pour fetch initial sémantique
            summary_prefilter_k: Nombre de documents du pré-filtre summary
            enable_hybrid: Active la recherche hybride (sinon semantic-only)
            enable_summary_prefilter: Active le pré-filtrage par summaries
        """
        self.collection = collection
        self.llm_provider = llm_provider
        self.summary_bm25 = summary_bm25_index
        self.chunk_bm25 = chunk_bm25_index
        self.query_expander = query_expander
        self.n_documents = n_documents
        self.n_chunks_per_doc = n_chunks_per_doc
        self.fetch_multiplier = fetch_multiplier
        self.summary_prefilter_k = summary_prefilter_k
        self.enable_hybrid = enable_hybrid
        self.enable_summary_prefilter = enable_summary_prefilter
    
    def retrieve(
        self,
        query: str,
        where_filter: Optional[Dict[str, Any]] = None,
        n_documents: Optional[int] = None,
        n_chunks_per_doc: Optional[int] = None
    ) -> List[RetrievedDocument]:
        """
        Récupère les documents pertinents via multi-query hybrid retrieval.
        
        Flux :
        1. Acronym expansion
        2. LLM query expansion → N queries
        3. Summary pre-filter (sur query originale)
        4. Pour chaque query : semantic + BM25 → rankings
        5. RRF fusion de TOUS les rankings
        6. Déduplication par document
        """
        n_docs = n_documents or self.n_documents
        n_chunks = n_chunks_per_doc or self.n_chunks_per_doc
        
        # 0. Expansion des acronymes
        expanded_query = expand_query_with_acronyms(query)
        if expanded_query != query:
            logger.info(f"🔤 Query expanded: '{expanded_query[:150]}...'")
        
        logger.info(f"🔍 Query: '{query[:100]}...'")
        
        # 0.5 LLM Query Expansion (multi-query)
        if self.query_expander is not None:
            all_queries = self.query_expander.expand(expanded_query)
        else:
            all_queries = [expanded_query]
        
        # 1. Summary Pre-Filter (sur la query principale uniquement)
        doc_filter = None
        if (self.enable_summary_prefilter 
            and self.summary_bm25 is not None 
            and self.summary_bm25._is_built):
            
            doc_filter = self.summary_bm25.get_relevant_doc_paths(
                expanded_query, top_k=self.summary_prefilter_k
            )
            logger.info(f"📋 Summary pre-filter: {len(doc_filter)} documents retenus")
        
        # 2. Multi-query retrieval : semantic + BM25 pour CHAQUE query
        n_fetch = n_docs * self.fetch_multiplier
        all_rankings = []  # Liste de rankings pour RRF
        ranking_weights = []  # Poids par ranking (originale ×2, expansions ×1)
        chunk_map: Dict[str, RetrievedChunk] = {}  # chunk_id → chunk
        
        for q_idx, q in enumerate(all_queries):
            q_label = "principale" if q_idx == 0 else f"expansion #{q_idx}"
            q_weight = 2.0 if q_idx == 0 else 1.0  # Query originale pèse 2× plus
            
            # 2a. Semantic search (dense) via ChromaDB
            query_embedding = self.llm_provider.embed([q])[0]
            
            try:
                semantic_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_fetch,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                logger.error(f"❌ Erreur query ChromaDB ({q_label}): {e}")
                continue
            
            semantic_chunks = self._parse_chromadb_results(semantic_results)
            
            # Appliquer le filtre summary
            if doc_filter:
                filtered = [c for c in semantic_chunks if c.document_path in doc_filter]
                if len(filtered) < 5:
                    remaining = [c for c in semantic_chunks if c not in filtered][:5 - len(filtered)]
                    filtered.extend(remaining)
                semantic_chunks = filtered
            
            logger.info(f"📥 Semantic ({q_label}): {len(semantic_chunks)} chunks")
            
            for chunk in semantic_chunks:
                chunk.semantic_score = chunk.similarity_score
            
            # Ranking sémantique pour cette query
            semantic_ranking = [c.chunk_id for c in semantic_chunks]
            all_rankings.append(semantic_ranking)
            ranking_weights.append(q_weight)
            
            # Stocker les chunks (garder le meilleur score sémantique ET la meilleure distance)
            for chunk in semantic_chunks:
                if chunk.chunk_id not in chunk_map:
                    chunk_map[chunk.chunk_id] = chunk
                else:
                    existing = chunk_map[chunk.chunk_id]
                    # Garder la meilleure distance (la plus basse) pour la validation
                    if chunk.distance < existing.distance:
                        existing.distance = chunk.distance
                    # Garder le meilleur score sémantique
                    if chunk.semantic_score > existing.semantic_score:
                        existing.semantic_score = chunk.semantic_score
            
            # 2b. BM25 search (sparse) - seulement pour la query principale
            if (q_idx == 0
                and self.enable_hybrid 
                and self.chunk_bm25 is not None 
                and self.chunk_bm25.is_built):
                
                bm25_results = self.chunk_bm25.search(
                    q, top_k=n_fetch, doc_filter=doc_filter
                )
                logger.info(f"📥 BM25 ({q_label}): {len(bm25_results)} chunks")
                
                bm25_ranking = [r.doc_key for r in bm25_results]
                all_rankings.append(bm25_ranking)
                ranking_weights.append(q_weight)  # BM25 = query principale = poids 2.0
                
                # Ajouter chunks BM25 manquants au chunk_map
                for result in bm25_results:
                    if result.doc_key not in chunk_map:
                        meta = dict(result.metadata)
                        text = meta.pop("text", "")
                        chunk = RetrievedChunk(
                            chunk_id=result.doc_key,
                            text=text,
                            document_path=meta.get("document_path", ""),
                            chunk_nature=meta.get("chunk_nature", "UNKNOWN"),
                            chunk_index=meta.get("chunk_index", 0),
                            confidence=meta.get("confidence", "unknown"),
                            distance=1.0,
                            metadata=meta,
                        )
                        chunk_map[result.doc_key] = chunk
                    
                    chunk_map[result.doc_key].bm25_score = result.score
        
        # 3. RRF Fusion pondéré de TOUS les rankings (sémantique multi-query + BM25)
        if len(all_rankings) > 1:
            rrf_scores = reciprocal_rank_fusion(all_rankings, weights=ranking_weights)
            for chunk_id, chunk in chunk_map.items():
                chunk.hybrid_score = rrf_scores.get(chunk_id, 0.0)
            logger.info(f"🔀 RRF fusion pondéré: {len(all_rankings)} rankings (weights={ranking_weights}) → {len(chunk_map)} chunks uniques")
        else:
            for chunk in chunk_map.values():
                chunk.hybrid_score = chunk.semantic_score
            logger.info(f"📥 Mode single-query: {len(chunk_map)} chunks")
        
        all_chunks = list(chunk_map.values())
        all_chunks.sort(key=lambda c: c.hybrid_score, reverse=True)
        
        # 4. Déduplication par document
        documents = self._deduplicate_by_document(all_chunks, n_docs, n_chunks)
        logger.info(f"📄 {len(documents)} documents uniques retenus")
        
        return documents
    
    def retrieve_candidates(
        self,
        query: str,
        n_candidates: int = 100,
        where_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Récupère un large pool de chunks candidats SANS déduplication par document.
        
        Conçu pour alimenter le cross-encoder reranker qui a besoin de voir
        un maximum de candidats avant de trier. Le retriever normal déduplique
        par document (5 docs × 3 chunks = 15 max), ce qui enterre les bons chunks
        quand l'espace vectoriel est dense.
        
        Flux :
        1. Acronym expansion
        2. LLM query expansion → N queries
        3. Summary pre-filter
        4. Pour chaque query : semantic + BM25 → rankings
        5. RRF fusion de TOUS les rankings
        6. Retourne top-N chunks par score RRF (PAS de dédup document)
        
        Args:
            query: Question utilisateur
            n_candidates: Nombre max de chunks à retourner
            where_filter: Filtre ChromaDB optionnel
            
        Returns:
            Liste de RetrievedChunk triés par score hybrid (RRF), sans dédup document
        """
        n_fetch = max(n_candidates, 50)  # Chercher au moins autant que demandé
        
        # 0. Expansion des acronymes
        expanded_query = expand_query_with_acronyms(query)
        if expanded_query != query:
            logger.info(f"🔤 Query expanded: '{expanded_query[:150]}...'")
        
        logger.info(f"🔍 Candidate retrieval: '{query[:100]}...' (n_candidates={n_candidates})")
        
        # 0.5 LLM Query Expansion (multi-query)
        if self.query_expander is not None:
            all_queries = self.query_expander.expand(expanded_query)
        else:
            all_queries = [expanded_query]
        
        # 1. Summary Pre-Filter
        doc_filter = None
        if (self.enable_summary_prefilter 
            and self.summary_bm25 is not None 
            and self.summary_bm25._is_built):
            doc_filter = self.summary_bm25.get_relevant_doc_paths(
                expanded_query, top_k=self.summary_prefilter_k
            )
            logger.info(f"📋 Summary pre-filter: {len(doc_filter)} documents retenus")
        
        # 2. Multi-query retrieval
        all_rankings = []
        ranking_weights = []
        chunk_map: Dict[str, RetrievedChunk] = {}
        
        for q_idx, q in enumerate(all_queries):
            q_label = "principale" if q_idx == 0 else f"expansion #{q_idx}"
            q_weight = 2.0 if q_idx == 0 else 1.0
            
            # 2a. Semantic search (dense) — fetch plus large
            query_embedding = self.llm_provider.embed([q])[0]
            
            try:
                semantic_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_fetch,
                    where=where_filter,
                    include=["documents", "metadatas", "distances"]
                )
            except Exception as e:
                logger.error(f"❌ Erreur query ChromaDB ({q_label}): {e}")
                continue
            
            semantic_chunks = self._parse_chromadb_results(semantic_results)
            
            # Appliquer le filtre summary
            if doc_filter:
                filtered = [c for c in semantic_chunks if c.document_path in doc_filter]
                if len(filtered) < 10:
                    remaining = [c for c in semantic_chunks if c not in filtered][:10 - len(filtered)]
                    filtered.extend(remaining)
                semantic_chunks = filtered
            
            for chunk in semantic_chunks:
                chunk.semantic_score = chunk.similarity_score
            
            semantic_ranking = [c.chunk_id for c in semantic_chunks]
            all_rankings.append(semantic_ranking)
            ranking_weights.append(q_weight)
            
            for chunk in semantic_chunks:
                if chunk.chunk_id not in chunk_map:
                    chunk_map[chunk.chunk_id] = chunk
                else:
                    existing = chunk_map[chunk.chunk_id]
                    if chunk.distance < existing.distance:
                        existing.distance = chunk.distance
                    if chunk.semantic_score > existing.semantic_score:
                        existing.semantic_score = chunk.semantic_score
            
            # 2b. BM25 search — pour TOUTES les queries (pas juste la principale)
            if (self.enable_hybrid 
                and self.chunk_bm25 is not None 
                and self.chunk_bm25.is_built):
                
                bm25_results = self.chunk_bm25.search(
                    q, top_k=n_fetch, doc_filter=doc_filter
                )
                
                bm25_ranking = [r.doc_key for r in bm25_results]
                all_rankings.append(bm25_ranking)
                # BM25 boost: query originale ×1.5, expansions ×0.75 (vs ancien ×1.0 / ×0.5)
                # Justification: nomic-embed-text échoue sur queries multi-concepts FR (q10),
                # BM25 excelle dans ces cas → augmenter son poids de 41% à ~51%
                bm25_weight = q_weight * 1.5 if q_idx == 0 else q_weight * 0.75
                ranking_weights.append(bm25_weight)
                
                for result in bm25_results:
                    if result.doc_key not in chunk_map:
                        meta = dict(result.metadata)
                        text = meta.pop("text", "")
                        chunk = RetrievedChunk(
                            chunk_id=result.doc_key,
                            text=text,
                            document_path=meta.get("document_path", ""),
                            chunk_nature=meta.get("chunk_nature", "UNKNOWN"),
                            chunk_index=meta.get("chunk_index", 0),
                            confidence=meta.get("confidence", "unknown"),
                            distance=1.0,
                            metadata=meta,
                        )
                        chunk_map[result.doc_key] = chunk
                    
                    chunk_map[result.doc_key].bm25_score = max(
                        chunk_map[result.doc_key].bm25_score, result.score
                    )
        
        # 3. RRF Fusion
        if len(all_rankings) > 1:
            rrf_scores = reciprocal_rank_fusion(all_rankings, weights=ranking_weights)
            for chunk_id, chunk in chunk_map.items():
                chunk.hybrid_score = rrf_scores.get(chunk_id, 0.0)
            logger.info(f"🔀 Candidate RRF: {len(all_rankings)} rankings → {len(chunk_map)} chunks uniques")
        else:
            for chunk in chunk_map.values():
                chunk.hybrid_score = chunk.semantic_score
        
        all_chunks = list(chunk_map.values())
        all_chunks.sort(key=lambda c: c.hybrid_score, reverse=True)
        
        result = all_chunks[:n_candidates]
        logger.info(f"📦 Candidates pour reranking: {len(result)} chunks (sur {len(all_chunks)} totaux)")
        
        return result
    
    def _parse_chromadb_results(self, results: Dict) -> List[RetrievedChunk]:
        """Parse les résultats bruts de ChromaDB."""
        chunks = []
        
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            chunk = RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                document_path=metadata.get("document_path", ""),
                chunk_nature=metadata.get("chunk_nature", "UNKNOWN"),
                chunk_index=metadata.get("chunk_index", 0),
                confidence=metadata.get("confidence", "unknown"),
                distance=distance,
                metadata=metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _fuse_results(
        self,
        semantic_chunks: List[RetrievedChunk],
        bm25_results: List,
    ) -> List[RetrievedChunk]:
        """Fusionne résultats sémantiques et BM25 via Reciprocal Rank Fusion."""
        semantic_ranking = [c.chunk_id for c in semantic_chunks]
        bm25_ranking = [r.doc_key for r in bm25_results]
        
        rrf_scores = reciprocal_rank_fusion([semantic_ranking, bm25_ranking])
        
        chunk_map: Dict[str, RetrievedChunk] = {}
        
        for chunk in semantic_chunks:
            chunk_map[chunk.chunk_id] = chunk
        
        bm25_score_map = {r.doc_key: r.score for r in bm25_results}
        for result in bm25_results:
            if result.doc_key not in chunk_map:
                meta = dict(result.metadata)  # Copy to avoid mutation
                text = meta.pop("text", "")
                chunk = RetrievedChunk(
                    chunk_id=result.doc_key,
                    text=text,
                    document_path=meta.get("document_path", ""),
                    chunk_nature=meta.get("chunk_nature", "UNKNOWN"),
                    chunk_index=meta.get("chunk_index", 0),
                    confidence=meta.get("confidence", "unknown"),
                    distance=1.0,
                    metadata=meta,
                )
                chunk_map[result.doc_key] = chunk
        
        for chunk_id, chunk in chunk_map.items():
            if chunk_id in bm25_score_map:
                chunk.bm25_score = bm25_score_map[chunk_id]
            chunk.hybrid_score = rrf_scores.get(chunk_id, 0.0)
        
        all_chunks = list(chunk_map.values())
        all_chunks.sort(key=lambda c: c.hybrid_score, reverse=True)
        
        return all_chunks
    
    def _deduplicate_by_document(
        self,
        chunks: List[RetrievedChunk],
        n_documents: int,
        n_chunks_per_doc: int
    ) -> List[RetrievedDocument]:
        """Déduplique les chunks par document (URL normalisée)."""
        doc_chunks = defaultdict(list)
        for chunk in chunks:
            doc_chunks[chunk.document_path].append(chunk)
        
        documents = []
        seen_urls = set()
        
        for doc_path, doc_chunk_list in doc_chunks.items():
            sorted_chunks = sorted(
                doc_chunk_list,
                key=lambda c: c.hybrid_score if c.hybrid_score > 0 else c.similarity_score,
                reverse=True
            )
            
            selected_chunks = sorted_chunks[:n_chunks_per_doc]
            
            source_url = selected_chunks[0].metadata.get('source_url', '') if selected_chunks else ''
            if source_url:
                normalized = source_url.lower().replace('https://', '').replace('http://', '').replace('www.', '')
                if normalized in seen_urls:
                    continue
                seen_urls.add(normalized)
            
            doc = RetrievedDocument(
                document_path=doc_path,
                chunks=selected_chunks,
                avg_similarity=0.0,
                primary_nature=""
            )
            documents.append(doc)
        
        documents.sort(key=lambda d: d.avg_similarity, reverse=True)
        return documents[:n_documents]
    
    def retrieve_chunks_only(
        self,
        query: str,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Version simple : top K chunks sans déduplication (debug)."""
        query_embedding = self.llm_provider.embed([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        return self._parse_chromadb_results(results)
    
    def format_results_debug(self, documents: List[RetrievedDocument]) -> str:
        """Format les résultats pour debug."""
        lines = [f"\n📊 {len(documents)} documents trouvés\n"]
        
        for i, doc in enumerate(documents, 1):
            lines.append(f"\n{'='*80}")
            lines.append(f"Document #{i} - Score: {doc.avg_similarity:.3f}")
            lines.append(f"Path: {doc.document_path}")
            lines.append(f"Nature: {doc.primary_nature}")
            lines.append(f"Chunks: {len(doc.chunks)}")
            
            for j, chunk in enumerate(doc.chunks, 1):
                hybrid_info = ""
                if chunk.hybrid_score > 0:
                    hybrid_info = f" | Hybrid: {chunk.hybrid_score:.4f}"
                if chunk.bm25_score > 0:
                    hybrid_info += f" | BM25: {chunk.bm25_score:.2f}"
                    
                lines.append(
                    f"\n  Chunk {j}/{len(doc.chunks)} - "
                    f"Semantic: {chunk.similarity_score:.3f}{hybrid_info}"
                )
                lines.append(
                    f"  Nature: {chunk.chunk_nature} | "
                    f"Index: {chunk.chunk_index} | "
                    f"Confidence: {chunk.confidence}"
                )
                lines.append(f"  Text: {chunk.text[:200]}...")
        
        lines.append(f"\n{'='*80}\n")
        return "\n".join(lines)


def create_retriever(
    collection,
    llm_provider,
    summary_bm25_index=None,
    chunk_bm25_index=None,
    query_expander: Optional[QueryExpander] = None,
    n_documents: int = 5,
    n_chunks_per_doc: int = 3,
    summary_prefilter_k: int = 40,
    enable_hybrid: bool = True,
    enable_summary_prefilter: bool = True,
) -> RAGRetriever:
    """Factory function pour créer un retriever hybride."""
    return RAGRetriever(
        collection=collection,
        llm_provider=llm_provider,
        summary_bm25_index=summary_bm25_index,
        chunk_bm25_index=chunk_bm25_index,
        query_expander=query_expander,
        n_documents=n_documents,
        n_chunks_per_doc=n_chunks_per_doc,
        summary_prefilter_k=summary_prefilter_k,
        enable_hybrid=enable_hybrid,
        enable_summary_prefilter=enable_summary_prefilter,
    )
