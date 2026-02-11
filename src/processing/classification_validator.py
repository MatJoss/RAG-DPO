"""
Classification Validation
Validation et auto-correction des classifications LLM
"""

from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ClassificationValidator:
    """Valide et corrige automatiquement les classifications douteuses"""
    
    # Règles de détection
    SANCTION_KEYWORDS = ['san-', 'délibération', 'amende', 'sanction', 'contentieux', 'mise en demeure']
    
    PRACTICAL_KEYWORDS = [
        'comment', 'étapes', 'procédure', "mode d'emploi", 
        'tutoriel', 'guide pratique', 'checklist', 'modèle'
    ]
    
    LEGAL_KEYWORDS = [
        'article', 'chapitre', 'section', 'alinéa', 
        'règlement (ue)', 'directive', 'loi n°'
    ]
    
    SECTEUR_KEYWORDS = {
        'Santé': ['santé', 'hôpital', 'médical', 'patient', 'soins'],
        'RH': ['ressources humaines', 'employé', 'salarié', 'recrutement', 'paie'],
        'Marketing': ['marketing', 'publicité', 'prospection', 'démarchage'],
        'Éducation': ['éducation', 'école', 'université', 'étudiant', 'élève'],
        'Banque': ['banque', 'crédit', 'compte bancaire', 'transaction'],
        'Assurance': ['assurance', 'assuré', "police d'assurance", 'sinistre'],
    }
    
    def validate(self, result: Dict, preview: str, title: str) -> Dict:
        """Valide et corrige automatiquement une classification
        
        VERSION CONSERVATIVE : Ne force un changement QUE si vraiment sûr
        
        Args:
            result: Résultat classification LLM
            preview: Preview du document
            title: Titre du document
            
        Returns:
            Résultat corrigé
        """
        
        preview_lower = preview.lower()
        title_lower = title.lower()
        combined = (preview_lower + " " + title_lower)[:2000]
        
        original_index = result.get('primary_index')
        corrections = []
        
        # RÈGLE 1 : Délibérations SANCTIONS (STRICTE)
        # Seulement si "SAN-" dans le texte OU ("délibération" + "amende/sanction")
        is_sanction = (
            'san-' in combined or 
            ('délibération' in combined and any(kw in combined for kw in ['amende', 'mise en demeure', 'condamnation']))
        )
        
        if is_sanction and result['primary_index'] != 'SANCTIONS':
            corrections.append("Sanction CNIL confirmée (SAN- ou délibération+amende)")
            result['primary_index'] = 'SANCTIONS'
            
            # Ajouter ancien index en secondaire
            if original_index and original_index not in result.get('secondary_indexes', []):
                result.setdefault('secondary_indexes', []).append(original_index)
        
        # RÈGLE 2 : Guides pratiques (SEULEMENT si flagrant)
        # Doit avoir "comment" OU "étapes" OU "guide pratique" dans le titre/début
        title_and_start = (title_lower + " " + preview_lower[:500])
        
        has_practical_title = any(kw in title_and_start for kw in [
            'comment', 'guide pratique', 'mode d\'emploi', 'tutoriel'
        ])
        
        if has_practical_title and result['primary_index'] == 'FONDAMENTAUX':
            corrections.append("Guide pratique détecté dans le titre")
            result['primary_index'] = 'OPERATIONNEL'
            
            if 'FONDAMENTAUX' not in result.get('secondary_indexes', []):
                result.setdefault('secondary_indexes', []).append('FONDAMENTAUX')
        
        # RÈGLE 3 : Textes de loi (STRICTE)
        # Seulement si BEAUCOUP d'articles de loi et PAS de guide
        article_count = combined.count('article ')
        has_regulation = 'règlement (ue)' in combined or 'directive' in combined
        is_not_guide = 'comment' not in title_and_start and 'guide' not in title_and_start
        
        if (article_count >= 5 or has_regulation) and is_not_guide:
            if result['primary_index'] == 'OPERATIONNEL':
                corrections.append(f"Texte de loi détecté ({article_count} articles)")
                result['primary_index'] = 'FONDAMENTAUX'
                
                if 'OPERATIONNEL' not in result.get('secondary_indexes', []):
                    result.setdefault('secondary_indexes', []).append('OPERATIONNEL')
        
        # RÈGLE 4 : Secteurs (AJOUT SEULEMENT, pas de changement d'index)
        detected_sectors = self._detect_sectors(combined)
        if detected_sectors:
            current_sectors = result.get('secteurs', [])
            for sect in detected_sectors:
                if sect not in current_sectors:
                    result.setdefault('secteurs', []).append(sect)
        
        # RÈGLE 5 : Importance minimale sanctions (gardée)
        if result['primary_index'] == 'SANCTIONS':
            if result.get('importance', 0) < 7:
                old_imp = result.get('importance', 0)
                result['importance'] = max(old_imp, 7)
                corrections.append(f"Importance sanctions: {old_imp}→7")
        
        # Logger et annoter
        if corrections:
            logger.debug(f"  🔧 Auto-corrections pour {title[:50]}:")
            for corr in corrections:
                logger.debug(f"     - {corr}")
            
            # Ajouter note dans raison SEULEMENT si changement d'index
            if original_index != result['primary_index']:
                result['raison'] = (
                    result.get('raison', '') + 
                    f" [Validé: {original_index}→{result['primary_index']}]"
                )
        
        return result
    
    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        """Vérifie si le texte contient au moins un mot-clé"""
        return any(kw in text for kw in keywords)
    
    def _detect_sectors(self, text: str) -> List[str]:
        """Détecte les secteurs mentionnés dans le texte"""
        detected = []
        
        for secteur, keywords in self.SECTEUR_KEYWORDS.items():
            if self._contains_any(text, keywords):
                detected.append(secteur)
        
        return detected


if __name__ == "__main__":
    # Tests
    validator = ClassificationValidator()
    
    # Test 1 : Sanction mal classée
    result1 = {
        'primary_index': 'OPERATIONNEL',
        'importance': 5
    }
    preview1 = "Délibération SAN-2024-001 - Amende de 50M€"
    
    corrected1 = validator.validate(result1, preview1, "Sanction CNIL")
    print(f"Test 1: {corrected1['primary_index']} (importance: {corrected1['importance']})")
    
    # Test 2 : Guide pratique mal classé
    result2 = {
        'primary_index': 'FONDAMENTAUX',
    }
    preview2 = "Comment faire une AIPD - Guide pratique en 10 étapes"
    
    corrected2 = validator.validate(result2, preview2, "Guide AIPD")
    print(f"Test 2: {corrected2['primary_index']}")