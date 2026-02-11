"""
Dictionnaire exhaustif des acronymes RGPD/CNIL pour l'expansion de requêtes.
"""

# Acronymes RGPD/Protection des données
RGPD_ACRONYMS = {
    # Analyses et évaluations
    "AIPD": "Analyse d'Impact relative à la Protection des Données",
    "PIA": "Privacy Impact Assessment",
    "EIPD": "Étude d'Impact sur la Protection des Données",
    "DPIA": "Data Protection Impact Assessment",
    
    # Acteurs et fonctions
    "DPO": "Délégué à la Protection des Données",
    "DPD": "Délégué à la Protection des Données",
    "RSSI": "Responsable de la Sécurité des Systèmes d'Information",
    "DSI": "Direction des Systèmes d'Information",
    "AFPD": "Agent chargé de la Protection des Données",
    
    # Réglementations et textes
    "RGPD": "Règlement Général sur la Protection des Données",
    "GDPR": "General Data Protection Regulation",
    "LIL": "Loi Informatique et Libertés",
    "ePrivacy": "Règlement vie privée et communications électroniques",
    
    # Autorités
    "CNIL": "Commission Nationale de l'Informatique et des Libertés",
    "CEPD": "Comité Européen de la Protection des Données",
    "EDPB": "European Data Protection Board",
    "EDPS": "European Data Protection Supervisor",
    "APD": "Autorité de Protection des Données",
    "ICO": "Information Commissioner's Office",
    
    # Principes et concepts clés
    "DCP": "Données à Caractère Personnel",
    "DPO": "Durée de conservation des données",
    "BCR": "Binding Corporate Rules",
    "RIG": "Règles Internes Contraignantes",
    "SCC": "Standard Contractual Clauses",
    "CTC": "Clauses Types Contractuelles",
    "CCT": "Clauses Contractuelles Types",
    
    # Droits des personnes
    "DSAR": "Data Subject Access Request",
    "RTBF": "Right To Be Forgotten",
    "DDAC": "Droit d'Accès aux Données",
    
    # Sécurité et incidents
    "VPD": "Violation de Protection des Données",
    "DPB": "Data Protection Breach",
    "FSD": "Fuite de Sécurité des Données",
    "PSSI": "Politique de Sécurité des Systèmes d'Information",
    
    # Transferts internationaux
    "TIP": "Transfert International de données Personnelles",
    "SCT": "Schéma de Certification des Transferts",
    "GAD": "Garanties Appropriées pour les Données",
    
    # Registres et documentation
    "RTD": "Registre des Traitements de Données",
    "ROPA": "Record of Processing Activities",
    "RVD": "Registre des Violations de Données",
    
    # Technologies et méthodes
    "PET": "Privacy Enhancing Technologies",
    "PbD": "Privacy by Design",
    "PbDf": "Privacy by Default",
    "VPC": "Vie Privée dès la Conception",
    "VPD": "Vie Privée par Défaut",
    "IA": "Intelligence Artificielle",
    "AI": "Artificial Intelligence",
    "ML": "Machine Learning",
    "IoT": "Internet of Things",
    "IdO": "Internet des Objets",
    
    # Conformité et audit
    "MPC": "Mise en conformité",
    "POC": "Preuve de Conformité",
    "ACC": "Accountability Conformité Continue",
    "DPO": "Data Protection Officer",
    
    # Bases légales
    "BLT": "Base Légale du Traitement",
    "CI": "Consentement Informé",
    "IM": "Intérêt Légitime",
    "IL": "Intérêt Légitime",
    "OL": "Obligation Légale",
    "EP": "Exécution d'un contrat ou de mesures Précontractuelles",
    "MI": "Mission d'Intérêt public",
    "SV": "Sauvegarde de la Vie",
    
    # Sous-traitance
    "ST": "Sous-Traitant",
    "RT": "Responsable de Traitement",
    "RST": "Responsable et Sous-Traitant",
    "CST": "Contrat de Sous-Traitance",
    "SST": "Sous-Sous-Traitant",
    
    # Cookies et traceurs
    "CNIL": "Cookies et autres traceurs",
    "SDK": "Software Development Kit",
    "API": "Application Programming Interface",
    "CMP": "Consent Management Platform",
    "PGC": "Plateforme de Gestion du Consentement",
    
    # Données spécifiques
    "DSS": "Données Sensibles de Santé",
    "DDS": "Données de Santé",
    "DBG": "Données Biométriques et Génétiques",
    "DPC": "Données de Profilage et Catégorisation",
    "DGE": "Données de Géolocalisation Exacte",
    
    # Hébergement et cloud
    "HDS": "Hébergement de Données de Santé",
    "HDH": "Hébergeur de Données de Santé",
    "PaaS": "Platform as a Service",
    "SaaS": "Software as a Service",
    "IaaS": "Infrastructure as a Service",
    
    # Certifications et labels
    "CC": "Certification CNIL",
    "LC": "Label CNIL",
    "ISO": "International Organization for Standardization",
    "ANSSI": "Agence Nationale de la Sécurité des Systèmes d'Information",
    
    # Secteurs spécifiques
    "RH": "Ressources Humaines",
    "GTA": "Gestion des Temps et Activités",
    "CRM": "Customer Relationship Management",
    "GRC": "Gestion de la Relation Client",
    "SIRH": "Système d'Information de Ressources Humaines",
    "ATS": "Applicant Tracking System",
    
    # Principes RGPD détaillés
    "LF": "Licéité Finalité",
    "MT": "Minimisation des données et Transparence",
    "EQD": "Exactitude et Qualité des Données",
    "LCD": "Limitation de la Conservation des Données",
    "IS": "Intégrité et Sécurité",
    "CR": "Confidentialité et Responsabilité",
    
    # Sanctions et procédures
    "MSC": "Mise en demeure CNIL",
    "SE": "Sanction Éducative",
    "SP": "Sanction Pécuniaire",
    "AME": "Amende",
    "REC": "Recours",
    "CIL": "Correspondant Informatique et Libertés",
}

def expand_query_with_acronyms(query: str) -> str:
    """
    Expanse une requête en remplaçant/ajoutant les définitions des acronymes détectés.
    Ajoute également un enrichissement contextuel spécifique pour certains termes.
    
    Args:
        query: Requête utilisateur originale
        
    Returns:
        Requête enrichie avec les définitions des acronymes
        
    Examples:
        >>> expand_query_with_acronyms("Comment faire une AIPD ?")
        "Comment faire une AIPD Analyse d'Impact relative à la Protection des Données ?"
        
        >>> expand_query_with_acronyms("Rôle du DPO dans le RGPD")
        "Rôle du DPO Délégué à la Protection des Données dans le RGPD Règlement Général sur la Protection des Données"
    """
    expanded = query
    query_lower = query.lower()
    
    # Enrichissement contextuel spécifique pour AIPD méthodologie
    if "aipd" in query_lower and any(keyword in query_lower for keyword in ["méthodologie", "comment", "étape", "faire", "réaliser", "procéder"]):
        expanded += " 3 parties description traitement évaluation nécessité proportionnalité analyse risques sécurité vie privée"
    
    # Recherche des acronymes dans la requête (mots en majuscules de 2+ lettres)
    words = query.split()
    for word in words:
        # Nettoyer les caractères de ponctuation
        clean_word = word.strip('.,;:!?()[]{}"\'-')
        
        # Vérifier si c'est un acronyme connu
        if clean_word in RGPD_ACRONYMS:
            definition = RGPD_ACRONYMS[clean_word]
            
            # Ajouter la définition après l'acronyme (sans dupliquer si déjà présent)
            if definition not in expanded:
                # Remplacer la première occurrence de l'acronyme
                expanded = expanded.replace(
                    f" {clean_word} ", 
                    f" {clean_word} {definition} ",
                    1
                )
                # Gérer aussi le cas où l'acronyme est en début de phrase
                if expanded.startswith(clean_word):
                    expanded = f"{clean_word} {definition} " + expanded[len(clean_word):].lstrip()
    
    return expanded


def get_all_acronyms() -> dict:
    """Retourne le dictionnaire complet des acronymes."""
    return RGPD_ACRONYMS.copy()


def add_acronym(acronym: str, definition: str) -> None:
    """
    Ajoute un nouvel acronyme au dictionnaire.
    
    Args:
        acronym: L'acronyme (ex: "AIPD")
        definition: Sa définition complète
    """
    RGPD_ACRONYMS[acronym] = definition


def search_acronym(acronym: str) -> str | None:
    """
    Recherche la définition d'un acronyme.
    
    Args:
        acronym: L'acronyme à rechercher
        
    Returns:
        La définition si trouvée, None sinon
    """
    return RGPD_ACRONYMS.get(acronym.upper())
