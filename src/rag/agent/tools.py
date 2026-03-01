"""
RAG Agent Tools — Outils locaux pour enrichir le raisonnement agent.

Tous les tools sont 100% locaux — aucune donnée ne sort du système.
Ils enrichissent le pipeline RAG sans compromettre la confidentialité.

Tools disponibles :
- DateCalculator : calcul de délais RGPD (72h notification, 2 ans conservation, etc.)
- ArticleLookup : index structuré des 99 articles du RGPD
- RegistrySearch : recherche dans les documents entreprise indexés
- IterativeRetrieval : reformulation + re-recherche si le 1er retrieval est insuffisant
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Date Calculator
# ═══════════════════════════════════════════════════════════════

# Délais RGPD courants (en jours sauf mention contraire)
RGPD_DEADLINES = {
    "notification_violation": {
        "delai": 72,  # heures
        "unite": "heures",
        "article": "art. 33 RGPD",
        "description": "Notification d'une violation de données à l'autorité de contrôle",
    },
    "reponse_droits": {
        "delai": 30,  # jours
        "unite": "jours",
        "article": "art. 12.3 RGPD",
        "description": "Réponse à une demande d'exercice de droits",
        "prolongation": "Prolongeable de 2 mois si complexité, sous réserve d'information",
    },
    "reponse_droits_prolongee": {
        "delai": 90,  # jours (30 + 60)
        "unite": "jours",
        "article": "art. 12.3 RGPD",
        "description": "Réponse prolongée (complexité ou nombre de demandes)",
    },
    "conservation_cv": {
        "delai": 730,  # 2 ans
        "unite": "jours",
        "article": "Recommandation CNIL",
        "description": "Conservation des CV après dernier contact candidat",
    },
    "conservation_prospection": {
        "delai": 1095,  # 3 ans
        "unite": "jours",
        "article": "Recommandation CNIL",
        "description": "Conservation des données de prospection commerciale après dernier contact",
    },
    "conservation_videosurveillance": {
        "delai": 30,  # jours
        "unite": "jours",
        "article": "Recommandation CNIL",
        "description": "Conservation des images de vidéosurveillance",
    },
    "tenue_registre": {
        "delai": None,  # permanent
        "unite": "permanent",
        "article": "art. 30 RGPD",
        "description": "Tenue du registre des activités de traitement (obligation continue)",
    },
    "aipd_avant_mise_en_oeuvre": {
        "delai": None,  # avant mise en œuvre
        "unite": "avant_traitement",
        "article": "art. 35 RGPD",
        "description": "L'AIPD doit être réalisée AVANT la mise en œuvre du traitement",
    },
}


@dataclass
class DateResult:
    """Résultat d'un calcul de délai."""
    deadline_type: str
    description: str
    article: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    remaining_days: Optional[int] = None
    is_expired: Optional[bool] = None
    note: str = ""


def calculate_deadline(
    deadline_type: str,
    start_date: Optional[datetime] = None,
) -> DateResult:
    """
    Calcule un délai RGPD à partir d'une date de début.
    
    Args:
        deadline_type: Clé dans RGPD_DEADLINES
        start_date: Date de début (défaut: maintenant)
    
    Returns:
        DateResult avec dates et statut
    """
    if deadline_type not in RGPD_DEADLINES:
        available = ", ".join(RGPD_DEADLINES.keys())
        return DateResult(
            deadline_type=deadline_type,
            description=f"Type inconnu. Disponibles : {available}",
            article="N/A",
        )
    
    info = RGPD_DEADLINES[deadline_type]
    
    if info["delai"] is None:
        return DateResult(
            deadline_type=deadline_type,
            description=info["description"],
            article=info["article"],
            note=f"Délai {info['unite']} — pas de date limite calculable",
        )
    
    if start_date is None:
        start_date = datetime.now()
    
    if info["unite"] == "heures":
        end_date = start_date + timedelta(hours=info["delai"])
    else:
        end_date = start_date + timedelta(days=info["delai"])
    
    now = datetime.now()
    remaining = (end_date - now).total_seconds()
    
    if info["unite"] == "heures":
        remaining_display = remaining / 3600
        remaining_label = f"{remaining_display:.1f} heures"
    else:
        remaining_display = remaining / 86400
        remaining_label = f"{remaining_display:.0f} jours"
    
    note = info.get("prolongation", "")
    
    return DateResult(
        deadline_type=deadline_type,
        description=info["description"],
        article=info["article"],
        start_date=start_date,
        end_date=end_date,
        remaining_days=int(remaining / 86400),
        is_expired=remaining < 0,
        note=f"Reste {remaining_label}" + (f". {note}" if note else ""),
    )


def list_deadlines() -> List[Dict[str, str]]:
    """Liste tous les délais RGPD disponibles."""
    return [
        {
            "type": key,
            "description": info["description"],
            "delai": f"{info['delai']} {info['unite']}" if info['delai'] else info['unite'],
            "article": info["article"],
        }
        for key, info in RGPD_DEADLINES.items()
    ]


# ═══════════════════════════════════════════════════════════════
# Article RGPD Lookup
# ═══════════════════════════════════════════════════════════════

# Index structuré des articles clés du RGPD
# (les plus utilisés en pratique DPO)
RGPD_ARTICLES = {
    4: "Définitions (donnée personnelle, traitement, responsable, sous-traitant, consentement, violation, etc.)",
    5: "Principes relatifs au traitement (licéité, finalité, minimisation, exactitude, limitation conservation, intégrité, responsabilité)",
    6: "Licéité du traitement — les 6 bases légales (consentement, contrat, obligation légale, intérêts vitaux, mission publique, intérêt légitime)",
    7: "Conditions applicables au consentement",
    8: "Consentement des enfants (services de la société de l'information)",
    9: "Traitement de catégories particulières de données (données sensibles)",
    10: "Traitement des données relatives aux condamnations pénales",
    12: "Transparence — modalités d'exercice des droits des personnes",
    13: "Informations à fournir lors de la collecte directe",
    14: "Informations à fournir lors de la collecte indirecte",
    15: "Droit d'accès de la personne concernée",
    16: "Droit de rectification",
    17: "Droit à l'effacement (« droit à l'oubli »)",
    18: "Droit à la limitation du traitement",
    19: "Obligation de notification en cas de rectification, effacement ou limitation",
    20: "Droit à la portabilité des données",
    21: "Droit d'opposition",
    22: "Décision individuelle automatisée, y compris le profilage",
    24: "Responsabilité du responsable du traitement",
    25: "Protection des données dès la conception et par défaut (privacy by design)",
    26: "Responsables conjoints du traitement",
    27: "Représentants des responsables hors UE",
    28: "Sous-traitant — obligations contractuelles",
    30: "Registre des activités de traitement",
    32: "Sécurité du traitement",
    33: "Notification d'une violation de données à l'autorité de contrôle (72h)",
    34: "Communication d'une violation de données à la personne concernée",
    35: "Analyse d'impact relative à la protection des données (AIPD/DPIA)",
    36: "Consultation préalable de l'autorité de contrôle",
    37: "Désignation du DPO — cas obligatoires",
    38: "Fonction du DPO",
    39: "Missions du DPO",
    44: "Principe général des transferts hors UE",
    45: "Transferts fondés sur une décision d'adéquation",
    46: "Transferts avec garanties appropriées (CCT, BCR)",
    47: "Règles d'entreprise contraignantes (BCR)",
    49: "Dérogations pour des situations particulières (transferts)",
    58: "Pouvoirs de l'autorité de contrôle (enquête, correctifs, sanctions)",
    77: "Droit d'introduire une réclamation auprès d'une autorité de contrôle",
    82: "Droit à réparation et responsabilité",
    83: "Conditions générales pour les amendes administratives",
    85: "Notification d'une violation de données à l'autorité de contrôle — délais",
    88: "Traitement dans le contexte de l'emploi",
    99: "Entrée en vigueur et application du règlement (25 mai 2018)",
}


def lookup_article(article_number: int) -> Dict[str, Any]:
    """
    Recherche un article RGPD par son numéro.
    
    Args:
        article_number: Numéro de l'article (1-99)
    
    Returns:
        Dict avec numéro, description, et articles connexes
    """
    if article_number < 1 or article_number > 99:
        return {
            "found": False,
            "error": f"Le RGPD contient 99 articles. Article {article_number} n'existe pas.",
        }
    
    if article_number in RGPD_ARTICLES:
        # Trouver les articles connexes (même chapitre)
        related = _find_related_articles(article_number)
        return {
            "found": True,
            "article": article_number,
            "titre": f"Article {article_number} RGPD",
            "description": RGPD_ARTICLES[article_number],
            "articles_connexes": related,
        }
    else:
        return {
            "found": True,
            "article": article_number,
            "titre": f"Article {article_number} RGPD",
            "description": "(Article non indexé dans la base de connaissances rapide — consulter le texte officiel)",
            "note": "Cet article existe mais n'est pas parmi les articles les plus fréquemment utilisés en pratique DPO.",
        }


def _find_related_articles(article_number: int) -> List[Dict[str, str]]:
    """Trouve les articles thématiquement proches."""
    # Groupes thématiques
    groups = {
        "droits_personnes": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        "principes": [5, 6, 7, 8, 9, 10],
        "securite_violations": [32, 33, 34, 85],
        "dpo": [37, 38, 39],
        "transferts": [44, 45, 46, 47, 49],
        "sous_traitance": [26, 28],
        "aipd": [35, 36],
        "sanctions": [58, 77, 82, 83],
    }
    
    related = []
    for group_name, articles in groups.items():
        if article_number in articles:
            for art in articles:
                if art != article_number and art in RGPD_ARTICLES:
                    related.append({
                        "article": art,
                        "description": RGPD_ARTICLES[art][:80],
                    })
    
    return related[:5]  # Max 5 articles connexes


def search_articles_by_topic(topic: str) -> List[Dict[str, Any]]:
    """
    Recherche des articles RGPD par sujet/mot-clé.
    
    Args:
        topic: Mot-clé à chercher (ex: "consentement", "violation", "DPO")
    
    Returns:
        Liste d'articles correspondants
    """
    topic_lower = topic.lower()
    results = []
    
    for num, desc in RGPD_ARTICLES.items():
        if topic_lower in desc.lower():
            results.append({
                "article": num,
                "titre": f"Article {num} RGPD",
                "description": desc,
            })
    
    return results


# ═══════════════════════════════════════════════════════════════
# Question Decomposer
# ═══════════════════════════════════════════════════════════════

def decompose_question(question: str, llm_provider) -> List[str]:
    """
    Décompose une question complexe en sous-questions atomiques.
    
    Utile pour les questions multi-aspects :
    "Quels sont les droits des personnes ET leurs limites ?"
    → ["Quels sont les droits des personnes concernées ?",
       "Quelles sont les limites aux droits des personnes ?"]
    
    Args:
        question: Question utilisateur
        llm_provider: Provider LLM pour la décomposition
    
    Returns:
        Liste de sous-questions (1 si pas décomposable)
    """
    prompt = f"""Analyse cette question RGPD. Si elle contient PLUSIEURS aspects distincts, décompose-la en sous-questions.
Si la question est simple et porte sur UN SEUL aspect, retourne-la telle quelle.

Question : {question}

Réponds UNIQUEMENT avec une liste JSON de questions (1 à 3 max) :
["question 1", "question 2"]

Si la question est simple :
["{question}"]

JSON :"""

    try:
        raw = llm_provider.generate(prompt, temperature=0.0, max_tokens=200)
        
        # Parser la liste JSON
        text = raw.strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            import json
            questions = json.loads(match.group())
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                # Max 3 sous-questions
                result = questions[:3]
                if len(result) > 1:
                    logger.info(f"🔀 Question décomposée en {len(result)} sous-questions")
                return result
        
        return [question]
    
    except Exception as e:
        logger.warning(f"⚠️  Décomposition échouée: {e}")
        return [question]


# ═══════════════════════════════════════════════════════════════
# Completeness Checker
# ═══════════════════════════════════════════════════════════════

def check_answer_completeness(
    question: str,
    answer: str,
    llm_provider,
) -> Dict[str, Any]:
    """
    Vérifie si la réponse couvre tous les aspects de la question.
    
    Utile pour le retry loop de l'agent : si la réponse est incomplète,
    identifier ce qui manque pour cibler la re-recherche.
    
    Args:
        question: Question originale
        answer: Réponse générée
        llm_provider: Provider LLM
    
    Returns:
        Dict avec is_complete, missing_aspects, suggested_queries
    """
    prompt = f"""Évalue si cette réponse couvre TOUS les aspects de la question.

Question : {question}

Réponse : {answer}

Réponds en JSON :
{{
  "is_complete": true/false,
  "coverage_pct": 0-100,
  "missing_aspects": ["aspect manquant 1", "aspect manquant 2"],
  "suggested_queries": ["requête ciblée pour trouver l'info manquante"]
}}

JSON :"""

    try:
        raw = llm_provider.generate(prompt, temperature=0.0, max_tokens=300)
        
        text = raw.strip()
        match = re.search(r'\{.*?\}', text, re.DOTALL)
        if match:
            import json
            result = json.loads(match.group())
            return {
                "is_complete": result.get("is_complete", True),
                "coverage_pct": result.get("coverage_pct", 100),
                "missing_aspects": result.get("missing_aspects", []),
                "suggested_queries": result.get("suggested_queries", []),
            }
        
        return {"is_complete": True, "coverage_pct": 100, "missing_aspects": [], "suggested_queries": []}
    
    except Exception as e:
        logger.warning(f"⚠️  Completeness check échoué: {e}")
        return {"is_complete": True, "coverage_pct": 100, "missing_aspects": [], "suggested_queries": []}
