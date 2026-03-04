"""
Tags RGPD — Vocabulaire guidé + rapprochement sémantique au runtime.

Architecture :
- ~25 catégories RGPD normalisées couvrant les thèmes DPO courants
- Le LLM est guidé pour choisir dans cette liste (mais peut proposer autre chose)
- parse_tags() normalise la réponse LLM vers le vocabulaire connu (fuzzy matching)
- TopicMatcher : boost sémantique au reranking via BGE-M3 (backup pour variantes)

Utilisé par :
- StructuralChunker (tags des chunks tableurs via LLM)
- tag_all_chunks.py (tags de TOUS les chunks via LLM)  
- IntentClassifier (topics des questions)
- Pipeline (topic-boost au reranking via similarité sémantique)
"""

import re
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# ── Vocabulaire RGPD normalisé (~25 catégories DPO) ──
# Couvre les thèmes principaux qu'un DPO rencontre au quotidien.
# Le LLM est guidé vers cette liste mais peut déborder si nécessaire.
RGPD_CATEGORIES = [
    "droits des personnes",
    "consentement",
    "sécurité des données",
    "durée de conservation",
    "sous-traitance",
    "base légale",
    "données sensibles",
    "transfert hors UE",
    "cookies",
    "violation de données",
    "transparence",
    "DPO",
    "vidéosurveillance",
    "finalité du traitement",
    "registre des traitements",
    "AIPD",
    "anonymisation",
    "minimisation",
    "responsable de traitement",
    "prospection commerciale",
    "conformité RGPD",
    "profilage",
    "sanctions CNIL",
    "données de santé",
    "information des personnes",
]

# Set lowercase pour matching rapide
_CATEGORIES_LOWER = {c.lower() for c in RGPD_CATEGORIES}

# Liste formatée pour injection dans les prompts
_CATEGORIES_STR = ", ".join(RGPD_CATEGORIES)

# Prompt de tagging : vocabulaire guidé
TAG_PROMPT = f"""Tu es un expert RGPD/CNIL. Lis cet extrait et identifie les thèmes RGPD couverts.

EXTRAIT :
{{text}}

Choisis 1 à 3 tags parmi cette liste (ou propose un tag court si aucun ne convient) :
{_CATEGORIES_STR}

Réponds UNIQUEMENT avec les tags séparés par des virgules, rien d'autre.

Tags :"""

# Prompt pour les tableurs (conversion table → texte + tags)
TAG_PROMPT_TABLE = f"""À la toute fin, ajoute une ligne commençant par [TAGS] suivie des thèmes RGPD couverts.
Choisis 1 à 3 tags parmi : {_CATEGORIES_STR}.
Si aucun ne convient, propose un tag court et descriptif.
"""


def _normalize_tag(tag: str) -> str:
    """Nettoyage basique d'un tag : lowercase, ponctuation trailing.
    
    Pas de fuzzy matching complexe — la convergence vers les catégories
    connues est assurée par le prompt qui guide le LLM.
    """
    clean = tag.strip().lower().rstrip('.;:!?')
    return clean if clean and len(clean) > 2 else ""


def parse_tags(raw_response: str) -> list[str]:
    """Parse la réponse LLM en liste de tags normalisés.
    
    Normalise vers le vocabulaire RGPD connu quand possible.
    
    Accepte des formats variés :
    - "droits des personnes, consentement"
    - "[TAGS] sécurité des données, AIPD"
    - "durée de conservation"
    """
    text = raw_response.strip()
    
    # Retirer préfixe [TAGS] si présent
    if text.upper().startswith('[TAGS]'):
        text = text[6:].strip()
    
    # Retirer lignes de texte avant les tags (si le LLM a bavardé)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 1:
        candidates = [l for l in lines if (',' in l or ';' in l) and len(l) < 200]
        if candidates:
            text = candidates[-1]
        else:
            text = lines[-1]
    
    # Splitter sur , et ;
    raw_tags = [t.strip().lower().rstrip('.;:!?') for t in re.split(r'[,;]', text)]
    
    # Normaliser chaque tag vers le vocabulaire connu
    tags = []
    seen = set()
    for t in raw_tags:
        if not t or len(t) < 3 or len(t) > 60:
            continue
        normalized = _normalize_tag(t)
        if normalized and normalized not in seen:
            tags.append(normalized)
            seen.add(normalized)
    
    # Max 3 tags
    return tags[:3]


class TopicMatcher:
    """Rapprochement sémantique entre topics de question et tags de chunks.
    
    Utilise les embeddings BGE-M3 pour comparer les tags par similarité
    cosinus, évitant tout problème de vocabulaire fermé.
    
    Cache les embeddings de tags déjà vus pour éviter les recalculs.
    """
    
    def __init__(self, embedding_provider=None):
        """
        Args:
            embedding_provider: EmbeddingProvider (BGE-M3). Si None, le boost
                sera désactivé silencieusement.
        """
        self._embedder = embedding_provider
        self._cache: dict[str, np.ndarray] = {}
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Embedding d'un tag/topic avec cache."""
        if self._embedder is None:
            return None
        if text not in self._cache:
            try:
                vec = self._embedder.embed([text])[0]
                self._cache[text] = np.array(vec)
            except Exception as e:
                logger.debug(f"Embedding tag '{text}' failed: {e}")
                return None
        return self._cache[text]
    
    def similarity(self, tag_a: str, tag_b: str) -> float:
        """Similarité cosinus entre deux tags/topics.
        
        Returns:
            Score entre 0.0 et 1.0. Retourne 0.0 si embedding indisponible.
        """
        vec_a = self._get_embedding(tag_a)
        vec_b = self._get_embedding(tag_b)
        if vec_a is None or vec_b is None:
            return 0.0
        # Cosine similarity (vecteurs déjà normalisés par BGE-M3)
        return float(np.dot(vec_a, vec_b))
    
    def topic_boost(
        self,
        question_topics: List[str],
        chunk_tags_str: str,
        threshold: float = 0.65,
    ) -> float:
        """Calcule un bonus de score pour un chunk dont les tags matchent les topics.
        
        Args:
            question_topics: Topics extraits de la question par l'intent classifier
            chunk_tags_str: Tags du chunk (string comma-separated depuis ChromaDB metadata)
            threshold: Seuil de similarité minimum pour considérer un match
            
        Returns:
            Bonus de score entre 0.0 (aucun match) et 0.15 (match parfait).
            Conçu pour être ajouté au score du reranker.
        """
        if not question_topics or not chunk_tags_str:
            return 0.0
        
        chunk_tags = [t.strip() for t in chunk_tags_str.split(',') if t.strip()]
        if not chunk_tags:
            return 0.0
        
        # Meilleur match parmi toutes les paires (topic, tag)
        best_sim = 0.0
        for topic in question_topics:
            for tag in chunk_tags:
                # Shortcut : match exact (pas besoin d'embedding)
                if topic.lower() == tag.lower():
                    best_sim = 1.0
                    break
                sim = self.similarity(topic, tag)
                if sim > best_sim:
                    best_sim = sim
            if best_sim >= 1.0:
                break
        
        if best_sim < threshold:
            return 0.0
        
        # Bonus linéaire : 0 au seuil, 0.15 à similarité 1.0
        max_boost = 0.15
        boost = max_boost * (best_sim - threshold) / (1.0 - threshold)
        return boost
