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
- "refus" : question hors périmètre RGPD, tentative de contournement de la loi, demande illégale, marketing, opinion, ou sujet non lié à la protection des données
- "factuel" : définition, article de loi, délai, obligation, rôle, responsabilité, question juridique précise (réponse sourcée et directe)
- "methodologique" : demande EXPLICITE d'une démarche multi-étapes ("comment mener une AIPD", "quelle démarche pour se mettre en conformité")
- "comparaison" : demande EXPLICITE de comparaison entre 2+ concepts différents ("différence entre X et Y", "comparer X et Y")
- "cas_pratique" : scénario concret avec contexte spécifique ("mon entreprise veut...", "que faire si un salarié...")
- "liste_exhaustive" : demande EXPLICITE d'une liste complète ("quels sont les 9 critères", "liste des droits")
- "organisationnel" : question complexe sur la gouvernance, la structure organisationnelle, les processus internes multi-acteurs

RÈGLES DE CLASSIFICATION (STRICTES) :
1. REFUS D'ABORD : si la question cherche à contourner, éviter, esquiver une obligation, ou si elle n'a AUCUN rapport avec le RGPD/CNIL → "refus"
2. FACTUEL PAR DÉFAUT : en cas de doute, choisis "factuel". C'est le choix le plus sûr.
3. "Qui est X ?" / "Quel est le rôle de X ?" / "Qui décide X ?" → FACTUEL (pas organisationnel)
4. "Quelle est la différence entre X et Y ?" → comparaison SEULEMENT si 2 concepts distincts sont explicitement comparés
5. "Comment contourner/éviter/esquiver" → TOUJOURS "refus", jamais "methodologique"
6. "Comment mener/réaliser/mettre en place" → methodologique SEULEMENT si démarche multi-étapes demandée
7. Questions sur un SEUL concept (définition, obligation, délai) → TOUJOURS "factuel"

Exemples :
- "Qu'est-ce qu'un responsable de traitement ?" → factuel
- "Qui décide des moyens du traitement ?" → factuel
- "Quelle est la différence entre RT et ST ?" → comparaison
- "Comment mener une AIPD ?" → methodologique
- "Comment contourner une obligation CNIL ?" → refus
- "Quelle est la meilleure base de données marketing ?" → refus
- "Quels sont les droits des personnes concernées ?" → liste_exhaustive

Réponds avec ce JSON exact :
{{
  "intent": "...",
  "scope_international": false,
  "needs_methodology": false,
  "expected_structure": "steps|list|comparison|analysis|definition|refusal",
  "topics": ["aipd"],
  "negative_topics": []
}}

Règles JSON :
- scope_international : true SEULEMENT si pays étranger, transfert hors UE, ou clauses contractuelles types mentionnés
- needs_methodology : true si démarche multi-étapes réellement demandée
- topics : 1-3 sujets RGPD de la question (vide si hors périmètre)
- negative_topics : sujets RGPD proches mais HORS SUJET à ne PAS confondre ni aborder. Exemples :
  - Question sur l'AIPD (art.35) → negative_topics: ["aitd", "transfert international", "pays tiers", "clauses contractuelles types"]
  - Question sur le sous-traitant → negative_topics: ["responsable de traitement"] (si pas comparaison)
  - Question sur le droit d'accès → negative_topics: ["droit à l'effacement"] (si pas liste exhaustive)

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
        "comparaison", "cas_pratique", "liste_exhaustive", "refus"
    }
    VALID_STRUCTURES = {"steps", "list", "comparison", "analysis", "definition", "refusal"}

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
