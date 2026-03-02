"""
Reranker - Cross-encoder reranking pour améliorer la précision du retrieval

Modèle par défaut : jinaai/jina-reranker-v2-base-multilingual (278M params)
- Multilingue natif (MIRACL FR 54.83, meilleur que BGE-v2-m3 54.17)
- 7x plus rapide que BGE-v2-m3 sur CPU (8ms/paire vs 58ms/paire)
- Tourne sur CPU (pas de VRAM consommée)
- Nécessite trust_remote_code=True + einops

Prend une query + une liste de chunks candidats, retourne les chunks réordonnés par pertinence.
"""
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RankedChunk:
    """Chunk avec son score de reranking"""
    chunk_id: str
    text: str
    document_path: str
    rerank_score: float  # Score cross-encoder (0-1, plus haut = plus pertinent)
    original_rank: int   # Position originale avant reranking
    metadata: dict


class CrossEncoderReranker:
    """
    Reranker basé sur cross-encoder pour rescorer les chunks candidats.
    
    Le cross-encoder prend (query, document) en paire et produit un score
    de pertinence bien plus précis que la similarité cosinus des embeddings.
    
    Architecture : Retrieve N candidats → Rerank → Garde top-K
    """
    
    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: str = "cpu",     # CPU pour ne pas toucher au VRAM
        batch_size: int = 32,
        max_length: int = 512,   # Max tokens pour le cross-encoder
        trust_remote_code: bool = True,  # Requis pour Jina
        min_score: float = 0.08,  # Score minimum pour garder un chunk
    ):
        """
        Args:
            model_name: Modèle cross-encoder HuggingFace
            device: 'cpu' ou 'cuda' — on utilise CPU par défaut
            batch_size: Taille de batch pour l'inférence
            max_length: Longueur max des séquences
            trust_remote_code: Autoriser l'exécution de code custom du modèle
            min_score: Score minimum de reranking pour garder un chunk (filtre le bruit)
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self.min_score = min_score
        self._model = None
        self._is_loaded = False
    
    def _load_model(self):
        """Chargement lazy du modèle (au premier appel)."""
        if self._is_loaded:
            return
        
        logger.info(f"🔄 Chargement cross-encoder: {self.model_name}...")
        
        try:
            import warnings
            from sentence_transformers import CrossEncoder
            
            # Supprimer les warnings répétitifs de flash_attn et torch_dtype
            warnings.filterwarnings("ignore", message=".*flash_attn.*")
            warnings.filterwarnings("ignore", message=".*torch_dtype.*is deprecated.*")
            logging.getLogger("transformers").setLevel(logging.ERROR)
            
            kwargs = {
                "max_length": self.max_length,
                "device": self.device,
                "tokenizer_kwargs": {"fix_mistral_regex": True},
            }
            if self.trust_remote_code:
                kwargs["trust_remote_code"] = True
            
            self._model = CrossEncoder(
                self.model_name,
                **kwargs,
            )
            self._is_loaded = True
            logging.getLogger("transformers").setLevel(logging.WARNING)
            logger.info(f"✅ Cross-encoder chargé sur {self.device}: {self.model_name}")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement cross-encoder: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        chunks: List,
        top_k: int = 8,
    ) -> List[RankedChunk]:
        """
        Reranke les chunks candidats par pertinence vis-à-vis de la query.
        
        Args:
            query: Question utilisateur
            chunks: Liste de RetrievedChunk (du retriever)
            top_k: Nombre de chunks à garder après reranking
            
        Returns:
            Liste de RankedChunk triés par score décroissant
        """
        if not chunks:
            return []
        
        self._load_model()
        
        # Construire les paires (query, chunk_text) pour le cross-encoder
        pairs = []
        for chunk in chunks:
            # Enrichir le texte avec heading/titre si disponible
            text = chunk.text
            heading = chunk.metadata.get("heading", "")
            if heading:
                text = f"{heading}\n{text}"
            pairs.append((query, text[:self.max_length * 4]))  # Truncate côté texte
        
        # Scoring par le cross-encoder
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"❌ Erreur reranking: {e}")
            # Fallback : retourner dans l'ordre original
            return [
                RankedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    document_path=c.document_path,
                    rerank_score=c.similarity_score,
                    original_rank=i,
                    metadata=c.metadata,
                )
                for i, c in enumerate(chunks[:top_k])
            ]
        
        # Associer scores aux chunks
        ranked = []
        for i, (chunk, score) in enumerate(zip(chunks, scores)):
            ranked.append(RankedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                document_path=chunk.document_path,
                rerank_score=float(score),
                original_rank=i,
                metadata=chunk.metadata,
            ))
        
        # Trier par score de reranking (décroissant)
        ranked.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Log des mouvements significatifs
        for r in ranked[:top_k]:
            movement = r.original_rank - ranked.index(r)
            if abs(movement) >= 3:
                direction = "↑" if movement > 0 else "↓"
                logger.info(
                    f"   {direction} Chunk '{r.text[:60]}...' "
                    f"moved {abs(movement)} positions (score={r.rerank_score:.3f})"
                )
        
        # Filtrer par score minimum (élimine le bruit non pertinent)
        result = [r for r in ranked[:top_k] if r.rerank_score >= self.min_score]
        
        # Garder au moins 3 chunks même si sous le seuil (fallback)
        if len(result) < 3 and len(ranked) >= 3:
            result = ranked[:3]
        
        filtered_count = min(top_k, len(ranked)) - len(result)
        if filtered_count > 0:
            logger.info(
                f"   🔽 {filtered_count} chunk(s) filtrés (score < {self.min_score})"
            )
        
        logger.info(
            f"✅ Reranking: {len(chunks)} → {len(result)} chunks "
            f"(top score={result[0].rerank_score:.3f}, "
            f"bottom score={result[-1].rerank_score:.3f})"
        )
        
        return result
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
