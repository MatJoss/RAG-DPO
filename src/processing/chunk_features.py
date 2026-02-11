"""
Chunk Features Extractor
Extraction de signaux lexicaux et structurels sans LLM
"""

import re
from typing import Dict


class ChunkFeaturesExtractor:
    """Extracteur de features heuristiques sur chunks"""
    
    # Patterns regex
    ARTICLE_RGPD = re.compile(r'article\s+\d+', re.IGNORECASE)
    STEPS_PATTERN = re.compile(r'étape\s+\d+|^\s*\d+\.|procédure|démarche|suivre\s+les\s+étapes', re.IGNORECASE | re.MULTILINE)
    TEMPLATE_PATTERN = re.compile(r'modèle|template|formulaire|tableau|téléchargeable|à\s+remplir', re.IGNORECASE)
    SANCTION_PATTERN = re.compile(r'san-\d+|amende|mise\s+en\s+demeure|délibération\s+n°|condamnation|sanction', re.IGNORECASE)
    TECHNICAL_PATTERN = re.compile(r'chiffrement|cryptographie|tls|ssl|https|pseudonymisation|anonymisation|hash|chiffrer', re.IGNORECASE)
    LEGAL_PATTERN = re.compile(r'règlement\s+\(ue\)|directive|loi\s+n°|code\s+civil|jurisprudence', re.IGNORECASE)
    
    # Mots-clés doctrine
    DOCTRINE_KEYWORDS = [
        'principe', 'définition', 'notion', 'portée', 'interprétation',
        'clarification', 'précision', 'lignes directrices', 'recommandation'
    ]
    
    # Mots-clés opérationnels
    OPERATIONAL_KEYWORDS = [
        'checklist', 'vérifier', 'contrôler', 'mettre en place',
        'mesure', 'outil', 'exemple', 'cas pratique'
    ]
    
    def extract(self, chunk_text: str) -> Dict:
        """Extrait les features d'un chunk"""
        
        text_lower = chunk_text.lower()
        word_count = len(chunk_text.split())
        
        # Features binaires (présence/absence)
        has_article_rgpd = bool(self.ARTICLE_RGPD.search(chunk_text))
        has_steps = bool(self.STEPS_PATTERN.search(chunk_text))
        has_template = bool(self.TEMPLATE_PATTERN.search(chunk_text))
        has_sanction = bool(self.SANCTION_PATTERN.search(chunk_text))
        has_technical = bool(self.TECHNICAL_PATTERN.search(chunk_text))
        has_legal_ref = bool(self.LEGAL_PATTERN.search(chunk_text))
        
        # Comptage keywords
        doctrine_score = sum(1 for kw in self.DOCTRINE_KEYWORDS if kw in text_lower)
        operational_score = sum(1 for kw in self.OPERATIONAL_KEYWORDS if kw in text_lower)
        
        # Detection listes / enumerations
        has_numbered_list = bool(re.search(r'^\s*\d+[\.)]\s+', chunk_text, re.MULTILINE))
        has_bullet_list = bool(re.search(r'^\s*[•\-\*]\s+', chunk_text, re.MULTILINE))
        
        # Detection tableaux (approximatif)
        has_table_markers = chunk_text.count('|') > 5 or chunk_text.count('\t') > 3
        
        return {
            # Signaux forts (règles dures)
            'has_article_rgpd': has_article_rgpd,
            'has_steps': has_steps,
            'has_template': has_template,
            'has_sanction': has_sanction,
            'has_technical': has_technical,
            'has_legal_ref': has_legal_ref,
            
            # Signaux faibles (scoring)
            'doctrine_score': doctrine_score,
            'operational_score': operational_score,
            
            # Structure
            'has_numbered_list': has_numbered_list,
            'has_bullet_list': has_bullet_list,
            'has_table': has_table_markers,
            
            # Méta
            'word_count': word_count,
            'char_count': len(chunk_text)
        }
    
    def classify_by_heuristics(self, features: Dict) -> Dict:
        """Classification purement heuristique (sans LLM)
        
        Returns:
            dict avec 'nature' et 'confidence'
        """
        
        # Règles dures (confidence = 1.0)
        if features['has_sanction']:
            return {'nature': 'SANCTION', 'confidence': 1.0, 'rule': 'sanction_keyword'}
        
        if features['has_template'] or (features['has_steps'] and features['has_numbered_list']):
            return {'nature': 'GUIDE', 'confidence': 0.95, 'rule': 'template_or_procedure'}
        
        if features['has_technical'] and features['word_count'] > 200:
            return {'nature': 'TECHNIQUE', 'confidence': 0.90, 'rule': 'technical_keywords'}
        
        # Règles mixtes (confidence moyenne)
        if features['has_article_rgpd'] and not features['has_steps']:
            # Article RGPD sans procédure = doctrine
            return {'nature': 'DOCTRINE', 'confidence': 0.85, 'rule': 'article_without_steps'}
        
        if features['doctrine_score'] >= 3:
            return {'nature': 'DOCTRINE', 'confidence': 0.75, 'rule': 'doctrine_keywords'}
        
        if features['operational_score'] >= 2 or features['has_numbered_list']:
            return {'nature': 'GUIDE', 'confidence': 0.70, 'rule': 'operational_keywords'}
        
        # Ambigu (confidence faible = besoin LLM)
        return {'nature': None, 'confidence': 0.0, 'rule': 'ambiguous'}


if __name__ == "__main__":
    # Tests
    extractor = ChunkFeaturesExtractor()
    
    # Test 1 : Doctrine
    text1 = """
    Le principe de minimisation, défini à l'article 5 du RGPD, impose aux responsables 
    de traitement de limiter la collecte de données personnelles au strict nécessaire.
    Cette notion doit être interprétée de manière stricte.
    """
    
    features1 = extractor.extract(text1)
    result1 = extractor.classify_by_heuristics(features1)
    print(f"Test 1 (Doctrine): {result1}")
    
    # Test 2 : Guide
    text2 = """
    Pour réaliser une AIPD, suivez les étapes suivantes :
    1. Identifier le traitement
    2. Évaluer les risques
    3. Mettre en place des mesures
    Utilisez le modèle Excel téléchargeable.
    """
    
    features2 = extractor.extract(text2)
    result2 = extractor.classify_by_heuristics(features2)
    print(f"Test 2 (Guide): {result2}")
    
    # Test 3 : Sanction
    text3 = """
    Délibération SAN-2024-001 : La CNIL a prononcé une amende de 50 000 € 
    à l'encontre de la société X pour manquement à l'article 13 du RGPD.
    """
    
    features3 = extractor.extract(text3)
    result3 = extractor.classify_by_heuristics(features3)
    print(f"Test 3 (Sanction): {result3}")
