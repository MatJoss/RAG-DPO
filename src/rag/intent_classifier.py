"""
Intent Classifier — Classification d'intention des questions DPO.

Analyse la question utilisateur AVANT le retrieval pour adapter le comportement
du pipeline (choix du system prompt, instructions de structuration).

Utilise le même Mistral-Nemo que le reste du pipeline (0 VRAM supplémentaire).
Appel court (~100 tokens) avec sortie JSON structurée.

Classes d'intent :
- factuel : question juridique précise (article, obligation, définition)
- methodologique : demande de méthodologie/processus/démarche
- organisationnel : question sur les acteurs, rôles, responsabilités
- comparaison : mise en parallèle de concepts
- cas_pratique : analyse d'un scénario concret
- liste_exhaustive : demande d'énumération complète
"""
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Prompt minimaliste pour classification rapide
CLASSIFY_PROMPT = """Analyse cette question RGPD et réponds UNIQUEMENT en JSON valide (pas de commentaire).

Question : {question}

Choisis l'intent parmi :
- "factuel" : définition, article de loi, délai, obligation précise (réponse courte, 100% sources)
- "methodologique" : "comment faire", "quelle démarche", étapes à suivre (réponse structurée en étapes)
- "organisationnel" : rôles, responsabilités, qui fait quoi, gouvernance (réponse centrée sur les acteurs)
- "comparaison" : "différence entre", "comparaison", pour/contre (réponse structurée en parallèle)
- "cas_pratique" : scénario concret, "mon entreprise veut...", "que faire si..." (analyse de cas)
- "liste_exhaustive" : "quels sont les X critères/droits/étapes", énumération complète demandée

Indices :
- "Quel est le rôle de X" / "Qui est responsable" → organisationnel
- "Quelle différence" / "Comparer" → comparaison
- "Mon entreprise veut..." / "Que faire si..." → cas_pratique
- "Comment" / "Quelle démarche" / "Quelles étapes" → methodologique
- "Qu'est-ce que" / "Quel délai" / "Quelle obligation" → factuel
- "Quels sont les N critères/droits" → liste_exhaustive

Réponds avec ce JSON exact :
{{
  "intent": "...",
  "scope_international": false,
  "needs_methodology": false,
  "expected_structure": "steps|list|comparison|analysis|definition",
  "topics": ["aipd"],
  "negative_topics": []
}}

Règles :
- scope_international : true SEULEMENT si pays étranger, transfert hors UE, ou clauses contractuelles types mentionnés
- needs_methodology : true si "comment faire", "démarche", "étapes"
- topics : 1-3 sujets RGPD de la question
- negative_topics : sujets à NE PAS aborder (ex: "transfert" si question purement nationale)

JSON :"""


@dataclass
class QuestionIntent:
    """Résultat de la classification d'intention."""
    intent: str = "factuel"
    scope_international: bool = False
    needs_methodology: bool = False
    expected_structure: str = "definition"
    topics: List[str] = field(default_factory=list)
    negative_topics: List[str] = field(default_factory=list)
    confidence: float = 1.0
    classification_time: float = 0.0

    # Intents valides
    VALID_INTENTS = {
        "factuel", "methodologique", "organisationnel",
        "comparaison", "cas_pratique", "liste_exhaustive"
    }
    VALID_STRUCTURES = {"steps", "list", "comparison", "analysis", "definition"}

    def __post_init__(self):
        """Validation et normalisation."""
        if self.intent not in self.VALID_INTENTS:
            self.intent = "factuel"
        if self.expected_structure not in self.VALID_STRUCTURES:
            self.expected_structure = "definition"

    @property
    def is_methodology(self) -> bool:
        """Vrai si la question demande une démarche structurée."""
        return self.intent == "methodologique" or self.needs_methodology

    @property
    def is_strict_sourcing(self) -> bool:
        """Vrai si la réponse doit être 100% sourcée (pas de connaissance générale)."""
        return self.intent in ("factuel", "liste_exhaustive")


class IntentClassifier:
    """
    Classifie l'intention d'une question DPO via un appel LLM court.

    Utilise le même modèle que le pipeline (Mistral-Nemo).
    Coût : ~0.3-0.5s, ~100 tokens output, 0 VRAM supplémentaire.
    """

    def __init__(self, llm_provider, timeout: float = 5.0):
        """
        Args:
            llm_provider: Provider Ollama (même que le pipeline)
            timeout: Timeout en secondes (fallback → factuel)
        """
        self.llm_provider = llm_provider
        self.timeout = timeout

    def classify(self, question: str) -> QuestionIntent:
        """
        Classifie une question DPO.

        Retourne toujours un QuestionIntent valide.
        En cas d'erreur → fallback factuel (comportement actuel du pipeline).

        Args:
            question: Question utilisateur brute

        Returns:
            QuestionIntent avec intent, topics, flags
        """
        start = time.time()

        try:
            prompt = CLASSIFY_PROMPT.format(question=question)

            raw = self.llm_provider.generate(
                prompt=prompt,
                temperature=0.0,  # Déterministe
                max_tokens=200,   # JSON court
            )

            elapsed = time.time() - start
            intent = self._parse_response(raw)
            intent.classification_time = elapsed

            logger.info(
                f"🎯 Intent: {intent.intent} | international={intent.scope_international} "
                f"| methodology={intent.needs_methodology} | topics={intent.topics} "
                f"| negative={intent.negative_topics} | {elapsed:.2f}s",
                extra={
                    "event": "intent_classification",
                    "intent": intent.intent,
                    "topics": intent.topics,
                    "time": elapsed,
                }
            )

            return intent

        except Exception as e:
            elapsed = time.time() - start
            logger.warning(
                f"⚠️  Intent classification failed ({elapsed:.1f}s): {e} — fallback factuel"
            )
            return QuestionIntent(
                intent="factuel",
                confidence=0.0,
                classification_time=elapsed,
            )

    def _parse_response(self, raw: str) -> QuestionIntent:
        """
        Parse la réponse LLM en QuestionIntent.

        Gère les cas courants :
        - JSON propre
        - JSON entouré de ```json ... ```
        - JSON avec trailing text
        - JSON invalide → fallback
        """
        text = raw.strip()

        # Extraire le JSON si entouré de code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        else:
            # Trouver le premier { et le dernier }
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                text = text[start_idx:end_idx + 1]

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Tenter de nettoyer les trailing commas
            cleaned = re.sub(r',\s*}', '}', text)
            cleaned = re.sub(r',\s*]', ']', cleaned)
            try:
                data = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(f"⚠️  JSON parsing failed: {text[:200]}")
                return QuestionIntent(confidence=0.5)

        # Construire QuestionIntent depuis le JSON
        return QuestionIntent(
            intent=data.get("intent", "factuel"),
            scope_international=bool(data.get("scope_international", False)),
            needs_methodology=bool(data.get("needs_methodology", False)),
            expected_structure=data.get("expected_structure", "definition"),
            topics=data.get("topics", []),
            negative_topics=data.get("negative_topics", []),
            confidence=1.0,
        )
