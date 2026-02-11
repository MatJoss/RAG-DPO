"""
Query Expansion LLM — Génère des reformulations de la question pour améliorer le recall.

Pattern multi-query : la question originale est reformulée en 2-3 variantes
qui capturent des formulations différentes, puis toutes les variantes sont
recherchées en parallèle et fusionnées.

Cela résout le problème de "gap sémantique" où le chunk contient la réponse
mais avec une formulation éloignée de la question utilisateur.
"""
import logging
import re
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Prompt minimaliste et directif pour éviter le bavardage
EXPANSION_PROMPT = """Tu es un expert RGPD/CNIL. Génère exactement 3 reformulations de la question ci-dessous.

Règles STRICTES :
- Chaque reformulation doit utiliser des mots-clés DIFFÉRENTS (synonymes, termes techniques, termes opérationnels)
- Pense aux termes qu'utiliserait un guide CNIL ou un texte réglementaire
- Format : une reformulation par ligne, numérotée 1. 2. 3.
- PAS d'explication, PAS de commentaire, JUSTE les 3 reformulations

Question : {question}"""


class QueryExpander:
    """
    Expand une question utilisateur en plusieurs reformulations via LLM.
    
    Stratégie :
    - Appel LLM léger (temperature élevée pour diversité)
    - Parse les 3 reformulations
    - Retourne [question_originale] + reformulations
    - Timeout court (5s) avec fallback gracieux
    """
    
    def __init__(
        self,
        llm_provider,
        enabled: bool = True,
        n_expansions: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 300,
        timeout: float = 10.0,
    ):
        """
        Args:
            llm_provider: Provider Ollama (doit avoir .generate())
            enabled: Active/désactive l'expansion
            n_expansions: Nombre de reformulations à générer
            temperature: Temperature LLM (élevée = plus de diversité)
            max_tokens: Tokens max pour la réponse
            timeout: Timeout en secondes
        """
        self.llm_provider = llm_provider
        self.enabled = enabled
        self.n_expansions = n_expansions
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
    
    def expand(self, question: str) -> List[str]:
        """
        Génère des reformulations de la question.
        
        Retourne toujours au minimum [question] (la question originale).
        En cas d'erreur ou timeout, retourne juste [question].
        
        Args:
            question: Question utilisateur originale
            
        Returns:
            Liste de queries : [question_originale, reformulation_1, ...]
        """
        if not self.enabled:
            return [question]
        
        start = time.time()
        
        try:
            prompt = EXPANSION_PROMPT.format(question=question)
            
            raw_response = self.llm_provider.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            elapsed = time.time() - start
            
            reformulations = self._parse_reformulations(raw_response)
            
            if reformulations:
                logger.info(
                    f"🔄 Query expansion: {len(reformulations)} reformulations en {elapsed:.1f}s"
                )
                for i, r in enumerate(reformulations, 1):
                    logger.debug(f"   {i}. {r[:120]}")
                
                # Question originale en premier (prioritaire) + reformulations
                return [question] + reformulations
            else:
                logger.warning(f"⚠️  Query expansion: aucune reformulation parsée ({elapsed:.1f}s)")
                return [question]
                
        except Exception as e:
            elapsed = time.time() - start
            logger.warning(f"⚠️  Query expansion échouée ({elapsed:.1f}s): {e}")
            return [question]
    
    def _parse_reformulations(self, raw: str) -> List[str]:
        """
        Parse la réponse LLM pour extraire les reformulations numérotées.
        
        Gère les formats :
        - "1. reformulation"
        - "1) reformulation"
        - "- reformulation"
        """
        lines = raw.strip().split('\n')
        reformulations = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Supprimer numérotation : "1. ", "1) ", "- ", "• "
            cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line)
            cleaned = re.sub(r'^[-•]\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            # Filtres de qualité
            if len(cleaned) < 10:  # Trop court
                continue
            if cleaned.lower().startswith(('voici', 'bien sûr', 'note', 'explication')):
                continue
            if cleaned == line:  # Pas de numérotation détectée → probablement du texte libre
                # Accepter quand même si ça ressemble à une question
                if not any(c in cleaned for c in ['?', 'comment', 'quel', 'quoi', 'quand', 'où', 'pourquoi', 'obligation', 'droit', 'donnée', 'RGPD', 'CNIL']):
                    continue
            
            reformulations.append(cleaned)
            
            if len(reformulations) >= self.n_expansions:
                break
        
        return reformulations
