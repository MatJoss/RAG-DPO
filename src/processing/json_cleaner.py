"""
JSON Cleaning Utilities
Nettoyage ultra-robuste des réponses LLM pour parsing JSON
"""

import re
import json
from typing import Dict, Any


def fix_json_quotes(json_str: str) -> str:
    """Échappe les guillemets non échappés dans les valeurs JSON
    
    Gère le cas : "raison": "texte avec " guillemets " dedans"
    → Transforme en : "raison": "texte avec \\" guillemets \\" dedans"
    """
    
    result = []
    in_string = False
    in_key = True
    i = 0
    
    while i < len(json_str):
        char = json_str[i]
        
        if char == '"' and (i == 0 or json_str[i-1] != '\\'):
            if not in_string:
                # Début d'une string
                in_string = True
                result.append(char)
            else:
                # Fin potentielle d'une string
                next_chars = json_str[i+1:i+10].strip()
                
                if (next_chars.startswith(':') or 
                    next_chars.startswith(',') or 
                    next_chars.startswith('}') or 
                    next_chars.startswith(']') or 
                    i == len(json_str) - 1):
                    # Vraie fin de string
                    in_string = False
                    result.append(char)
                    
                    if next_chars.startswith(':'):
                        in_key = False
                    elif next_chars.startswith(',') or next_chars.startswith('}'):
                        in_key = True
                else:
                    # Guillemet à l'intérieur d'une valeur → échapper
                    if not in_key:
                        result.append('\\"')
                    else:
                        result.append(char)
        else:
            result.append(char)
        
        i += 1
    
    return ''.join(result)


def convert_pipes_to_array(match) -> str:
    """Convertit "A" | "B" | "C" en ["A", "B", "C"]"""
    values = match.group(1)
    items = re.findall(r'"([^"]+)"', values)
    array = '["' + '", "'.join(items) + '"]'
    return f'"secteurs": {array}'


def clean_llm_json_response(response: str) -> str:
    """Pipeline complet de nettoyage d'une réponse LLM JSON
    
    Args:
        response: Réponse brute du LLM
        
    Returns:
        JSON nettoyé prêt à parser
    """
    
    response_clean = response.strip()
    
    # 1. Retirer balises markdown
    if response_clean.startswith('```json'):
        response_clean = response_clean[7:]
    elif response_clean.startswith('```'):
        response_clean = response_clean[3:]
    if response_clean.endswith('```'):
        response_clean = response_clean[:-3]
    response_clean = response_clean.strip()
    
    # 2. Nettoyage caractères typographiques
    response_clean = response_clean.replace('"', '"').replace('"', '"')
    response_clean = response_clean.replace(''', "'").replace(''', "'")
    response_clean = response_clean.replace('«', '"').replace('»', '"')
    
    # 3. Extraction JSON si texte autour
    if not response_clean.startswith('{'):
        start_idx = response_clean.find('{')
        if start_idx != -1:
            response_clean = response_clean[start_idx:]
        else:
            # Pas de JSON du tout (LLM a répondu en texte libre)
            # Retourner un JSON vide qui sera rejeté proprement
            return '{}'
    if not response_clean.endswith('}'):
        end_idx = response_clean.rfind('}')
        if end_idx != -1:
            response_clean = response_clean[:end_idx+1]
    
    # 3b. Fix doubles accolades {{ ... }} → { ... }
    #     Mistral Nemo escape parfois les accolades comme dans un template
    if response_clean.startswith('{{') and response_clean.endswith('}}'):
        response_clean = response_clean[1:-1]
    
    # 4. Échapper guillemets dans les valeurs
    response_clean = fix_json_quotes(response_clean)
    
    # 5. Conversion pipes → tableaux pour secteurs
    response_clean = re.sub(
        r'"secteurs?"\s*:\s*("(?:[^"]|\\")+"(?:\s*\|\s*"(?:[^"]|\\")+")*)',
        convert_pipes_to_array,
        response_clean
    )
    
    # 6. Normaliser secteur → secteurs
    response_clean = response_clean.replace('"secteur":', '"secteurs":')
    
    return response_clean


def safe_parse_json(json_str: str) -> Dict[str, Any]:
    """Parse JSON avec fallback json5 si disponible
    
    Args:
        json_str: String JSON à parser
        
    Returns:
        Dict parsé
        
    Raises:
        json.JSONDecodeError: Si parsing échoue
    """
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Tentative avec json5 (plus permissif)
        try:
            import json5
            return json5.loads(json_str)
        except (ImportError, Exception):
            # Re-lever l'erreur originale
            raise


if __name__ == "__main__":
    # Tests
    test_cases = [
        # Cas 1 : Guillemets dans valeur
        '{"raison": "Le texte contient " des guillemets " dedans"}',
        
        # Cas 2 : Pipes dans secteur
        '{"secteur": "Santé" | "RH" | "Marketing"}',
        
        # Cas 3 : Guillemets typographiques
        '{"secteur": "Santé", "raison": "C\'est un guide"}',
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n=== Test {i} ===")
        print(f"Input : {test}")
        
        try:
            cleaned = clean_llm_json_response(test)
            print(f"Clean : {cleaned}")
            
            parsed = safe_parse_json(cleaned)
            print(f"Parse : ✅ {parsed}")
        except Exception as e:
            print(f"Error : ❌ {e}")
