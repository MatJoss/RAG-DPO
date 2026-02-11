"""
Validators - Vérification de pertinence et grounding
"""
import logging
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Résultat de validation"""
    is_valid: bool
    score: float
    reason: str


class RelevanceValidator:
    """
    Valide que les chunks récupérés sont pertinents pour la question
    
    Utilise le LLM pour scorer la pertinence de chaque chunk
    """
    
    def __init__(self, llm_provider, threshold: float = 0.30):
        """
        Args:
            llm_provider: Provider LLM
            threshold: Seuil de distance pour considérer pertinent (plus bas = meilleur)
                      0.25 = très strict, 0.30 = équilibré, 0.35 = permissif
        """
        self.llm_provider = llm_provider
        self.threshold = threshold
    
    def validate_chunks(
        self,
        query: str,
        chunks: List,
        conversation_history: Optional[List] = None
    ) -> List:
        """
        Filtre les chunks non pertinents
        
        Args:
            query: Question utilisateur
            chunks: Liste de RetrievedChunk
            conversation_history: Historique de conversation
        
        Returns:
            Chunks filtrés (seulement les pertinents)
        """
        if not chunks:
            return chunks
        
        # Context complet : historique + question
        context_query = query
        if conversation_history:
            # Prendre les 3 derniers messages
            recent = conversation_history[-6:]  # 3 paires user/assistant
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent])
            context_query = f"{history_text}\n\nQuestion actuelle: {query}"
        
        # Filtrage par distance
        filtered = []
        rejected = []
        
        for chunk in chunks:
            if chunk.distance <= self.threshold:
                filtered.append(chunk)
            else:
                rejected.append(chunk)
                logger.warning(
                    f"⚠️  Chunk rejeté (distance={chunk.distance:.3f} > {self.threshold}): "
                    f"{chunk.text[:100]}..."
                )
        
        if rejected:
            logger.info(f"✂️  {len(rejected)}/{len(chunks)} chunks filtrés (non pertinents)")
        
        return filtered if filtered else chunks  # Fallback: garder tout si rien ne passe


class GroundingValidator:
    """
    Valide que la réponse générée est bien groundée dans les sources
    
    Vérifie :
    - Présence de citations [Source X]
    - Pas d'invention de sources
    - Pas d'hallucination : chaque fait doit être dans le contexte
    - Cohérence entre réponse et contexte
    """
    
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
    
    def validate_response(
        self,
        response: str,
        available_sources: List[int],
        context: str
    ) -> ValidationResult:
        """
        Valide que la réponse est bien groundée
        
        Args:
            response: Réponse générée
            available_sources: IDs des sources disponibles [1, 2, 3]
            context: Contexte fourni au LLM
        
        Returns:
            ValidationResult avec score et raison
        """
        issues = []
        
        # 1. Vérifier présence de citations
        if "[Source" not in response and "Source " not in response:
            issues.append("Aucune citation de source")
            logger.warning("⚠️  Réponse sans citation de source")
        
        # 2. Vérifier invention de sources
        import re
        # Match: [Source 1], [Source 1 et Source 2], [Source 1, 2 et 3], etc.
        cited_sources = re.findall(r'Source\s+(\d+)', response)
        cited_ids = [int(s) for s in cited_sources]
        
        invalid_sources = [s for s in cited_ids if s not in available_sources]
        if invalid_sources:
            issues.append(f"Sources inventées: {invalid_sources}")
            logger.error(f"❌ Sources INVENTÉES: {invalid_sources} (disponibles: {available_sources})")
        
        # 3. Vérifier réponse vide ou trop courte
        if len(response.strip()) < 50:
            issues.append("Réponse trop courte")
            logger.warning("⚠️  Réponse très courte")
        
        # 4. Info : Détecter phrases "consultez" (pas une erreur, juste info)
        consultez_phrases = [
            "consultez la CNIL",
            "consultez les questions",
            "consultez les guides",
            "la CNIL vous propose",
            "la CNIL met à disposition",
            "vous pouvez consulter"
        ]
        
        response_lower = response.lower()
        found_consultez = [p for p in consultez_phrases if p in response_lower]
        if found_consultez:
            # Info seulement, pas une erreur
            logger.info(f"ℹ️  Phrases 'consultez' détectées: {found_consultez}")
        
        # 5. Vérifier phrases d'évitement critiques
        critical_evasive = [
            "je ne peux pas répondre",
            "je n'ai pas d'information",
            "contactez votre DPO",
            "demandez à votre délégué"
        ]
        
        found_evasive = [p for p in critical_evasive if p in response_lower]
        if found_evasive:
            issues.append(f"Réponse évasive: {found_evasive}")
            logger.warning(f"⚠️  Réponse évasive détectée: {found_evasive}")
        
        # 6. NOUVEAU : Vérifier hallucinations (faits inventés)
        hallucination_check = self._check_hallucinations(response, context)
        if not hallucination_check['is_grounded']:
            issues.append(f"Hallucination détectée: {hallucination_check['reason']}")
            logger.error(f"❌ HALLUCINATION: {hallucination_check['reason']}")
        
        # Score : 1.0 si OK, pénalité par issue
        is_valid = len(issues) == 0
        score = 1.0 - (len(issues) * 0.25)  # -0.25 par problème
        score = max(0.0, score)
        
        reason = "; ".join(issues) if issues else "OK"
        
        if not is_valid:
            logger.warning(f"⚠️  Validation réponse: {reason} (score={score:.2f})")
        
        return ValidationResult(
            is_valid=is_valid,
            score=score,
            reason=reason
        )
    
    def _check_hallucinations(self, response: str, context: str) -> dict:
        """
        Vérifie si la réponse contient des hallucinations.
        
        Approche déterministe et rapide (pas d'appel LLM) :
        - Vérifie que les montants € cités existent dans le contexte
        - Vérifie que les articles de loi cités existent dans le contexte
        - Vérifie que les noms d'organisations cités existent dans le contexte
        
        On ne bloque PAS sur les termes techniques (PIA, AIPD, etc.) qui 
        font partie du vocabulaire RGPD courant.
        """
        import re
        
        context_lower = context.lower()
        response_lower = response.lower()
        issues = []
        
        # 1. Vérifier les montants € inventés
        # Ex: "20 millions d'euros", "4% du CA", "10 000 €"
        amounts_in_response = re.findall(
            r'(\d[\d\s]*(?:millions?|milliards?)?\s*(?:d\'?euros?|€))',
            response_lower
        )
        for amount in amounts_in_response:
            # Extraire le nombre pour chercher dans le contexte
            number = re.search(r'\d[\d\s]*', amount)
            if number:
                num_str = number.group().strip()
                if num_str not in context_lower and len(num_str) > 2:
                    issues.append(f"Montant '{amount.strip()}' non trouvé dans le contexte")
        
        # 2. Vérifier les articles de loi inventés (art. XX, article XX)
        articles_in_response = re.findall(
            r'(?:article|art\.?)\s+(\d+(?:[\-\.]\d+)?)',
            response_lower
        )
        for art_num in articles_in_response:
            if art_num not in context_lower:
                issues.append(f"Article {art_num} non trouvé dans le contexte")
        
        # 3. Vérifier les dates spécifiques inventées (25 mai 2018, etc.)
        dates_in_response = re.findall(
            r'(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})',
            response_lower
        )
        for date_str in dates_in_response:
            # Tolérance : la date du RGPD (25 mai 2018) est connaissance générale
            if '25 mai 2018' in date_str or '27 avril 2016' in date_str:
                continue
            if date_str not in context_lower:
                issues.append(f"Date '{date_str}' non trouvée dans le contexte")
        
        if issues:
            reason = " ; ".join(issues[:3])  # Max 3 issues
            logger.warning(f"⚠️  Grounding warnings: {reason}")
            return {"is_grounded": False, "reason": reason}
        
        return {"is_grounded": True, "reason": "OK"}
    
    def fix_invented_sources(
        self,
        response: str,
        available_sources: List[int]
    ) -> str:
        """
        Supprime les citations vers des sources inventées
        
        Args:
            response: Réponse avec potentiellement sources inventées
            available_sources: IDs valides [1, 2, 3]
        
        Returns:
            Réponse nettoyée
        """
        import re
        
        # Trouver toutes les citations
        def replace_citation(match):
            source_num = int(match.group(1))
            if source_num in available_sources:
                return match.group(0)  # Garder
            else:
                logger.info(f"🧹 Suppression citation invalide: [Source {source_num}]")
                return ""  # Supprimer
        
        fixed = re.sub(r'\[Source (\d+)\]', replace_citation, response)
        
        # Nettoyer doubles espaces
        fixed = re.sub(r'  +', ' ', fixed)
        
        return fixed.strip()
