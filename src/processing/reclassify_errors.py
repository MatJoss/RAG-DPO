"""
Re-classification des Documents en Erreur
Relance UNIQUEMENT les documents qui ont échoué lors de la Phase 5A
"""

import json
from pathlib import Path
import sys
import logging
from typing import Dict, List
from tqdm import tqdm
import time

# Ajouter chemins
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'utils'))

from llm_provider import RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


def find_errors_in_results(results_file: Path) -> List[Dict]:
    """Identifie les documents en erreur dans les résultats"""
    
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    
    errors = []
    
    for file_path, result in metadata.items():
        is_error = False
        error_type = None
        
        # Type 1: Erreur explicite
        if result.get('error', False):
            is_error = True
            error_type = 'explicit_error'
        
        # Type 2: Classification par défaut (raison contient "erreur" ou "défaut")
        raison = result.get('raison', '').lower()
        if any(keyword in raison for keyword in ['erreur', 'défaut', 'error', 'échec', 'failed']):
            is_error = True
            error_type = 'default_classification'
        
        # Type 3: JSON parsing errors
        if 'Erreur parsing' in result.get('raison', ''):
            is_error = True
            error_type = 'json_error'
        
        # Type 4: Importance très basse (< 3) peut indiquer un problème
        if result.get('importance', 10) < 3:
            is_error = True
            error_type = 'low_importance'
        
        if is_error:
            errors.append({
                'file_path': file_path,
                'error_type': error_type,
                'current_result': result
            })
    
    return errors


def reclassify_errors(project_root: str = '.'):
    """Re-classifie uniquement les documents en erreur"""
    
    project_root = Path(project_root)
    data_path = project_root / 'data'
    
    results_file = data_path / 'document_metadata.json'
    cache_file = data_path / 'document_classification_cache.json'
    manifest_file = data_path / 'keep_manifest.json'
    
    print("=" * 70)
    print("🔄 RE-CLASSIFICATION DES ERREURS - PHASE 5A")
    print("=" * 70)
    
    # Vérifier fichiers
    if not results_file.exists():
        print("\n❌ document_metadata.json introuvable")
        print("   Lancez classify_documents.py d'abord")
        return
    
    # Identifier erreurs
    print("\n🔍 Identification des erreurs...")
    errors = find_errors_in_results(results_file)
    
    if not errors:
        print("\n✅ Aucune erreur détectée !")
        print("   Tous les documents ont été classifiés correctement")
        return
    
    print(f"\n⚠️  {len(errors)} documents en erreur détectés :")
    
    # Grouper par type d'erreur
    error_types = {}
    for err in errors:
        err_type = err['error_type']
        error_types[err_type] = error_types.get(err_type, 0) + 1
    
    for err_type, count in error_types.items():
        print(f"   {err_type:25s} : {count:4d}")
    
    # Confirmation
    print(f"\n⏱️  Durée estimée : ~{len(errors) * 2 / 60:.0f} minutes")
    confirm = input("\n   Re-classifier ces documents ? (oui/non) : ")
    
    if confirm.lower() not in ['oui', 'yes', 'y', 'o']:
        print("❌ Annulé")
        return
    
    # Charger cache existant
    cache = {}
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    
    # Charger manifest pour récupérer info documents
    with open(manifest_file, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    # Créer mapping file_path → doc_info
    file_to_doc = {}
    
    for item in manifest.get('html', []):
        file_path = item['metadata']['file_path']
        file_to_doc[file_path] = {
            'url': item['url'],
            'file_path': file_path,
            'type': 'html',
            'title': item.get('metadata', {}).get('title', ''),
            'parent_url': None,
            'related_resources': manifest.get('relationships', {}).get(item['url'], []),
        }
    
    for item in manifest.get('pdfs', []):
        file_path = item['metadata']['file_path']
        file_to_doc[file_path] = {
            'url': item['url'],
            'file_path': file_path,
            'type': 'pdf',
            'title': '',
            'parent_url': item.get('parent_url'),
            'related_resources': [],
        }
    
    for item in manifest.get('docs', []):
        file_path = item['metadata']['file_path']
        file_to_doc[file_path] = {
            'url': item['url'],
            'file_path': file_path,
            'type': 'doc',
            'title': '',
            'parent_url': item.get('parent_url'),
            'related_resources': [],
        }
    
    # Initialiser LLM
    try:
        config = RAGConfig()
        llm = config.llm_provider
        mode = config.mode
        logger.info(f"🤖 LLM initialisé en mode : {mode}")
    except Exception as e:
        logger.error(f"❌ Erreur init LLM : {e}")
        return
    
    # Import classifier pour réutiliser les méthodes
    sys.path.insert(0, str(project_root / 'src' / 'processing'))
    from classify_documents import DocumentClassifier
    
    classifier = DocumentClassifier(str(project_root))
    
    # Re-classification
    print(f"\n🔄 Re-classification en cours...")
    
    fixed = 0
    still_errors = 0
    
    for err in tqdm(errors, desc="Re-classification"):
        file_path = err['file_path']
        
        # Retirer du cache pour forcer re-classification
        if file_path in classifier.cache:
            del classifier.cache[file_path]
        
        # Récupérer doc_info
        doc_info = file_to_doc.get(file_path)
        
        if not doc_info:
            logger.warning(f"⚠️  Document {file_path} non trouvé dans manifest, skip")
            still_errors += 1
            continue
        
        # Re-classifier
        try:
            new_result = classifier.classify_document(doc_info)
            
            # Vérifier si toujours en erreur
            if new_result.get('error', False):
                still_errors += 1
                logger.warning(f"⚠️  Toujours en erreur : {Path(file_path).name}")
            else:
                fixed += 1
                logger.info(f"✅ Corrigé : {Path(file_path).name}")
            
        except Exception as e:
            logger.error(f"❌ Échec re-classification {Path(file_path).name}: {e}")
            still_errors += 1
    
    # Sauvegarder cache mis à jour
    classifier._save_cache()
    
    # Charger résultats complets et mettre à jour
    with open(results_file, 'r', encoding='utf-8') as f:
        full_results = json.load(f)
    
    # Mettre à jour avec nouvelles classifications
    for file_path in cache:
        if file_path in full_results['metadata']:
            full_results['metadata'][file_path] = cache[file_path]
    
    # Sauvegarder résultats mis à jour
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    # Résumé
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ RE-CLASSIFICATION")
    print("=" * 70)
    
    print(f"\n⚠️  Erreurs initiales    : {len(errors)}")
    print(f"✅ Corrigées           : {fixed}")
    print(f"❌ Toujours en erreur  : {still_errors}")
    
    success_rate = (fixed / len(errors) * 100) if errors else 100
    print(f"\n🎯 Taux de correction   : {success_rate:.1f}%")
    
    print("\n" + "=" * 70)
    print(f"💾 Cache mis à jour    : {cache_file}")
    print(f"💾 Résultats mis à jour: {results_file}")
    print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Re-classification des erreurs Phase 5A')
    parser.add_argument('--project-root', type=str, default='.', help='Racine du projet')
    
    args = parser.parse_args()
    
    reclassify_errors(args.project_root)


if __name__ == "__main__":
    main()
