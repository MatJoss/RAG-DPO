"""
BM25 Index - Index de recherche par mots-clés pour retrieval hybride

Deux index :
1. Summary-level : BM25 sur les fiches synthétiques (pré-filtrage documents)
2. Chunk-level : BM25 sur les chunks (recherche fine)

Utilisé en combinaison avec la recherche sémantique ChromaDB pour le hybrid retrieval.
"""
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Stopwords français courants (non pertinents pour la recherche)
FRENCH_STOPWORDS = {
    "le", "la", "les", "de", "des", "du", "un", "une", "et", "en", "au", "aux",
    "ce", "ces", "cette", "qui", "que", "quoi", "dont", "où", "par", "pour",
    "dans", "sur", "avec", "sans", "sous", "entre", "vers", "chez", "est",
    "sont", "être", "avoir", "fait", "faire", "peut", "il", "elle", "ils",
    "elles", "nous", "vous", "on", "se", "ne", "pas", "plus", "très", "aussi",
    "mais", "ou", "donc", "car", "si", "ni", "je", "tu", "son", "sa", "ses",
    "leur", "leurs", "mon", "ma", "mes", "ton", "ta", "tes", "notre", "votre",
    "tout", "tous", "toute", "toutes", "même", "autre", "autres", "quel",
    "quelle", "quels", "quelles", "comme", "être", "été", "ayant", "après",
    "avant", "lors", "depuis", "pendant", "alors", "ainsi", "bien", "peu",
    "trop", "assez", "encore", "déjà", "jamais", "rien", "chaque",
    "cet", "aux", "à", "d", "l", "n", "s", "c", "j", "m", "t", "y",
}


def tokenize_french(text: str) -> List[str]:
    """
    Tokenisation simple adaptée au français juridique/DPO.
    Garde les termes techniques, acronymes, numéros d'articles.
    """
    # Lowercase
    text = text.lower()
    # Garder lettres, chiffres, tirets (pour sous-traitant, etc.)
    tokens = re.findall(r'[a-zàâäéèêëïîôùûüÿçœæ0-9]+(?:-[a-zàâäéèêëïîôùûüÿçœæ0-9]+)*', text)
    # Filtrer stopwords et tokens trop courts
    tokens = [t for t in tokens if t not in FRENCH_STOPWORDS and len(t) > 1]
    return tokens


@dataclass
class BM25Result:
    """Résultat d'une recherche BM25"""
    doc_key: str         # Clé du document (path ou ID)
    score: float         # Score BM25
    metadata: Dict       # Métadonnées associées


class SummaryBM25Index:
    """
    Index BM25 sur les fiches synthétiques (summaries).
    Permet de pré-filtrer les documents pertinents avant la recherche chunks.
    
    Exploite la structure NATURE/TYPE/SUJETS/USAGE DPO/SECTEUR/OBLIGATIONS.
    """
    
    def __init__(self, summaries_path: Optional[Path] = None):
        self.summaries_path = summaries_path or Path("data/keep/cnil/document_summaries.json")
        self.index: Optional[BM25Okapi] = None
        self.doc_keys: List[str] = []       # Index → doc_path
        self.doc_metadata: List[Dict] = []  # Index → metadata
        self.corpus_tokens: List[List[str]] = []
        self._is_built = False
    
    def build(self, summaries_path: Optional[str] = None) -> None:
        """Construit l'index BM25 à partir des summaries.
        
        Args:
            summaries_path: Chemin override (sinon utilise self.summaries_path)
        """
        if summaries_path:
            self.summaries_path = Path(summaries_path)
        
        logger.info("📚 Construction index BM25 sur summaries...")
        
        if not self.summaries_path.exists():
            raise FileNotFoundError(f"Fichier summaries introuvable: {self.summaries_path}")
        
        with open(self.summaries_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        
        self.doc_keys = []
        self.doc_metadata = []
        self.corpus_tokens = []
        
        skipped = 0
        for doc_path, entry in summaries.items():
            summary_text = entry.get("summary", "")
            if not summary_text or summary_text.startswith("ERREUR"):
                skipped += 1
                continue
            
            # Enrichir le texte avec titre et URL pour meilleur matching
            title = entry.get("document_title", "")
            url = entry.get("source_url", "")
            
            # Texte complet pour BM25 : titre + summary + URL
            full_text = f"{title} {summary_text} {url}"
            
            tokens = tokenize_french(full_text)
            if not tokens:
                skipped += 1
                continue
            
            self.doc_keys.append(doc_path)
            self.doc_metadata.append({
                "document_path": doc_path,
                "source_url": url,
                "document_title": title,
                "summary": summary_text,
            })
            self.corpus_tokens.append(tokens)
        
        # Construction BM25
        self.index = BM25Okapi(self.corpus_tokens)
        self._is_built = True
        
        logger.info(
            f"✅ Index BM25 summaries construit: {len(self.doc_keys)} documents "
            f"({skipped} ignorés)"
        )
    
    def search(self, query: str, top_k: int = 20) -> List[BM25Result]:
        """
        Recherche BM25 dans les summaries.
        
        Args:
            query: Question utilisateur
            top_k: Nombre de résultats
            
        Returns:
            Liste de BM25Result triés par score décroissant
        """
        if not self._is_built:
            raise RuntimeError("Index non construit. Appelez build() d'abord.")
        
        query_tokens = tokenize_french(query)
        if not query_tokens:
            logger.warning("⚠️ Query vide après tokenisation BM25")
            return []
        
        scores = self.index.get_scores(query_tokens)
        
        # Top-K avec scores > 0
        scored = [(i, scores[i]) for i in range(len(scores)) if scores[i] > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]
        
        results = []
        for idx, score in scored:
            results.append(BM25Result(
                doc_key=self.doc_keys[idx],
                score=score,
                metadata=self.doc_metadata[idx],
            ))
        
        return results
    
    def get_relevant_doc_paths(self, query: str, top_k: int = 20) -> Set[str]:
        """Raccourci : retourne juste les paths des documents pertinents."""
        results = self.search(query, top_k=top_k)
        return {r.doc_key for r in results}


class ChunkBM25Index:
    """
    Index BM25 sur les chunks ChromaDB.
    Utilisé pour le volet sparse du hybrid retrieval.
    """
    
    def __init__(self):
        self.index: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []
        self.chunk_texts: List[str] = []
        self.chunk_metadatas: List[Dict] = []
        self.corpus_tokens: List[List[str]] = []
        self._is_built = False
    
    def build_from_collection(self, collection, batch_size: int = 5000) -> None:
        """
        Construit l'index BM25 à partir d'une collection ChromaDB.
        
        Args:
            collection: Collection ChromaDB
            batch_size: Taille des batches pour extraction
        """
        logger.info("📚 Construction index BM25 sur chunks ChromaDB...")
        
        total = collection.count()
        logger.info(f"   Total chunks à indexer: {total}")
        
        self.chunk_ids = []
        self.chunk_texts = []
        self.chunk_metadatas = []
        self.corpus_tokens = []
        
        # Extraction par batches
        offset = 0
        while offset < total:
            batch = collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"]
            )
            
            for chunk_id, text, metadata in zip(
                batch["ids"], batch["documents"], batch["metadatas"]
            ):
                if not text or not text.strip():
                    continue
                
                tokens = tokenize_french(text)
                if not tokens:
                    continue
                
                self.chunk_ids.append(chunk_id)
                self.chunk_texts.append(text)
                self.chunk_metadatas.append(metadata)
                self.corpus_tokens.append(tokens)
            
            offset += batch_size
            logger.info(f"   Indexé {min(offset, total)}/{total} chunks...")
        
        # Construction BM25
        self.index = BM25Okapi(self.corpus_tokens)
        self._is_built = True
        
        logger.info(f"✅ Index BM25 chunks construit: {len(self.chunk_ids)} chunks")
    
    def search(
        self,
        query: str,
        top_k: int = 30,
        doc_filter: Optional[Set[str]] = None
    ) -> List[BM25Result]:
        """
        Recherche BM25 dans les chunks.
        
        Args:
            query: Question utilisateur
            top_k: Nombre max de résultats
            doc_filter: Si fourni, ne retourne que les chunks de ces documents
            
        Returns:
            Liste de BM25Result triés par score décroissant
        """
        if not self._is_built:
            raise RuntimeError("Index non construit. Appelez build_from_collection() d'abord.")
        
        query_tokens = tokenize_french(query)
        if not query_tokens:
            return []
        
        scores = self.index.get_scores(query_tokens)
        
        # Filtrer et trier
        scored = []
        for i in range(len(scores)):
            if scores[i] <= 0:
                continue
            if doc_filter is not None:
                doc_path = self.chunk_metadatas[i].get("document_path", "")
                if doc_path not in doc_filter:
                    continue
            scored.append((i, scores[i]))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        scored = scored[:top_k]
        
        results = []
        for idx, score in scored:
            results.append(BM25Result(
                doc_key=self.chunk_ids[idx],
                score=score,
                metadata={
                    **self.chunk_metadatas[idx],
                    "text": self.chunk_texts[idx],
                },
            ))
        
        return results
    
    @property
    def is_built(self) -> bool:
        return self._is_built
