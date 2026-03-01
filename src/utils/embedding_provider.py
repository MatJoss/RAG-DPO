"""
Embedding Provider — BGE-M3 via sentence-transformers (FP16, GPU)

Remplace nomic-embed-text (Ollama, anglais seul, 768 dims) par
BAAI/bge-m3 (multilingual, 1024 dims, dense+sparse intégré).

Usage :
    from src.utils.embedding_provider import EmbeddingProvider
    
    embedder = EmbeddingProvider()       # charge BGE-M3 en FP16 sur GPU
    vectors = embedder.embed(texts)      # List[List[float]], dim=1024
    embedder.unload()                    # libère VRAM
"""

import logging
import time
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────────────────────
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_DIMS = 1024
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
DEFAULT_BATCH_SIZE = 64       # BGE-M3 gère bien les gros batchs sur GPU
MAX_SEQ_LENGTH = 8192         # BGE-M3 supporte 8192 tokens max
TRUNCATE_CHARS = 20000        # Sécurité BGE-M3 : 8192 tokens ≈ ~24K chars (nos chunks max ~5K)


class EmbeddingProvider:
    """
    Provider d'embeddings BGE-M3 (dense).
    
    - Chargement lazy : le modèle est chargé au premier appel embed()
    - FP16 sur GPU par défaut (~1.07 GB VRAM)
    - Batch encoding pour la performance
    - Thread-safe pour Streamlit (st.cache_resource)
    """
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = DEFAULT_DEVICE,
        dtype: torch.dtype = DEFAULT_DTYPE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        self._model: Optional[SentenceTransformer] = None
        self._dims: int = DEFAULT_DIMS
        
        logger.info(
            f"📐 EmbeddingProvider configuré : {model_name} "
            f"({device}, {dtype}, batch={batch_size})"
        )
    
    # ── Propriétés ────────────────────────────────────────────────────────────
    
    @property
    def dims(self) -> int:
        """Dimensions des embeddings (1024 pour BGE-M3)."""
        return self._dims
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    # ── Chargement / Déchargement ────────────────────────────────────────────
    
    def load(self) -> "EmbeddingProvider":
        """Charge le modèle en mémoire (GPU). Idempotent."""
        if self._model is not None:
            return self
        
        t0 = time.time()
        logger.info(f"⏳ Chargement {self.model_name} sur {self.device}...")
        
        self._model = SentenceTransformer(
            self.model_name,
            device=self.device,
            model_kwargs={"dtype": self.dtype},
            cache_folder=self.cache_dir,
        )
        
        # Vérifier les dimensions réelles
        test_emb = self._model.encode(["test"], convert_to_numpy=True)
        self._dims = test_emb.shape[1]
        
        vram_gb = torch.cuda.memory_allocated(0) / 1024**3 if self.device == "cuda" else 0
        elapsed = time.time() - t0
        
        logger.info(
            f"✅ {self.model_name} chargé en {elapsed:.1f}s "
            f"(dims={self._dims}, VRAM={vram_gb:.2f} GB)"
        )
        return self
    
    def unload(self):
        """Libère le modèle et la VRAM."""
        if self._model is not None:
            del self._model
            self._model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            logger.info(f"🗑️  {self.model_name} déchargé, VRAM libérée")
    
    # ── Embeddings ────────────────────────────────────────────────────────────
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Génère les embeddings dense pour une liste de textes.
        
        Args:
            texts: Liste de textes à encoder
            
        Returns:
            Liste de vecteurs (List[float]), dim=1024
        """
        if not texts:
            return []
        
        # Lazy load
        if self._model is None:
            self.load()
        
        # Tronquer les textes trop longs
        truncated = [t[:TRUNCATE_CHARS] if len(t) > TRUNCATE_CHARS else t for t in texts]
        
        # Encode en batch
        embeddings = self._model.encode(
            truncated,
            batch_size=self.batch_size,
            show_progress_bar=len(truncated) > 500,
            convert_to_numpy=True,
            normalize_embeddings=True,   # Cosine similarity → normaliser
        )
        
        return embeddings.tolist()
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embed une seule query. Raccourci pour embed([query])[0].
        
        Pour BGE-M3, il est recommandé de préfixer les queries
        avec une instruction, mais sentence-transformers le gère
        automatiquement si le modèle le requiert.
        """
        return self.embed([query])[0]
    
    # ── Utilitaires ──────────────────────────────────────────────────────────
    
    def is_available(self) -> bool:
        """Vérifie si le provider peut fonctionner."""
        try:
            if self.device == "cuda" and not torch.cuda.is_available():
                return False
            return True
        except Exception:
            return False
    
    def get_info(self) -> dict:
        """Retourne les infos du provider (pour debug/logs)."""
        vram_gb = torch.cuda.memory_allocated(0) / 1024**3 if self.device == "cuda" else 0
        return {
            "model": self.model_name,
            "device": self.device,
            "dtype": str(self.dtype),
            "dims": self._dims,
            "loaded": self.is_loaded,
            "vram_gb": round(vram_gb, 2),
            "batch_size": self.batch_size,
        }
    
    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return f"EmbeddingProvider({self.model_name}, {self.device}, {status})"
    
    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
