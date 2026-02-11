"""
Correction des Extensions et Re-analyse
Détecte les fichiers mal nommés (ODS→XLSX, etc.) et corrige
"""

import json
from pathlib import Path
import shutil
import hashlib
import logging
from typing import Dict, List, Tuple
from tqdm import tqdm
import mimetypes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ExtensionFixer:
    """Corrige les extensions de fichiers mal nommés"""
    
    # Mapping MIME type → extension correcte
    MIME_TO_EXT = {
        'application/vnd.oasis.opendocument.spreadsheet': '.ods',
        'application/vnd.oasis.opendocument.text': '.odt',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'application/vnd.ms-excel': '.xls',
        'application/msword': '.doc',
        'application/pdf': '.pdf',
    }
    
    # Signatures de fichiers (magic bytes)
    FILE_SIGNATURES = {
        b'PK\x03\x04': 'zip_based',  # ODS, ODT, XLSX, DOCX sont des ZIP
        b'%PDF': '.pdf',
        b'\xd0\xcf\x11\xe0': 'ole_based',  # XLS, DOC (old format)
    }
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.data_path = self.project_root / 'data'
        self.metadata_dir = self.data_path / 'metadata'
        
        # Fichiers
        self.resource_index_file = self.metadata_dir / 'resource_index_by_source.json'
        self.cache_file = self.data_path / 'resource_classification_cache.json'
        
        # Stats
        self.stats = {
            'total_files': 0,
            'mismatches': 0,
            'ods_renamed': 0,
            'xlsx_renamed': 0,
            'odt_renamed': 0,
            'docx_renamed': 0,
            'other_renamed': 0,
            'errors': 0,
        }
    
    def detect_real_extension(self, file_path: Path) -> str:
        """Détecte la vraie extension d'un fichier"""
        
        try:
            # Lire les premiers bytes (signature)
            with open(file_path, 'rb') as f:
                header = f.read(8)
            
            # Vérifier signatures
            for signature, file_type in self.FILE_SIGNATURES.items():
                if header.startswith(signature):
                    if file_type == 'zip_based':
                        # Pour ZIP, il faut aller plus loin
                        return self._detect_zip_based_format(file_path)
                    elif file_type == 'ole_based':
                        # XLS ou DOC (ancien format)
                        # Pour simplifier, on regarde l'extension déclarée
                        return file_path.suffix.lower()
                    else:
                        return file_type
            
            # Fallback : utiliser l'extension actuelle
            return file_path.suffix.lower()
        
        except Exception as e:
            logger.debug(f"⚠️  Erreur détection {file_path.name}: {e}")
            return file_path.suffix.lower()
    
    def _detect_zip_based_format(self, file_path: Path) -> str:
        """Détecte le format exact des fichiers ZIP (ODS, ODT, XLSX, DOCX)"""
        
        try:
            import zipfile
            
            with zipfile.ZipFile(file_path, 'r') as zf:
                namelist = zf.namelist()
                
                # ODS : contient content.xml et mimetype avec spreadsheet
                if 'mimetype' in namelist:
                    mimetype = zf.read('mimetype').decode('utf-8').strip()
                    
                    if 'spreadsheet' in mimetype:
                        return '.ods'
                    elif 'text' in mimetype and 'opendocument' in mimetype:
                        return '.odt'
                
                # XLSX : contient xl/workbook.xml
                if 'xl/workbook.xml' in namelist:
                    return '.xlsx'
                
                # DOCX : contient word/document.xml
                if 'word/document.xml' in namelist:
                    return '.docx'
            
            # Si on ne peut pas déterminer, garder l'extension actuelle
            return file_path.suffix.lower()
        
        except Exception as e:
            logger.debug(f"⚠️  Erreur ZIP {file_path.name}: {e}")
            return file_path.suffix.lower()
    
    def analyze_mismatches(self) -> Dict:
        """Analyse les incohérences extension déclarée vs réelle"""
        
        print("=" * 70)
        print("🔍 ANALYSE DES INCOHÉRENCES D'EXTENSIONS")
        print("=" * 70)
        
        if not self.resource_index_file.exists():
            print("\n❌ resource_index_by_source.json introuvable")
            return {}
        
        # Charger index
        with open(self.resource_index_file, 'r', encoding='utf-8') as f:
            resource_index = json.load(f)
        
        print(f"\n📊 Index chargé : {len(resource_index)} pages HTML")
        
        mismatches = []
        
        # Analyser chaque ressource
        print("\n🔍 Détection des incohérences...")
        
        for source_url, data in tqdm(resource_index.items(), desc="Analyse"):
            for resource in data.get('resources', []):
                url = resource['url']
                declared_type = resource['file_type']
                file_path = self.project_root / resource['file_path']
                
                if not file_path.exists():
                    continue
                
                self.stats['total_files'] += 1
                
                # Détecter vraie extension
                declared_ext = f".{declared_type}"
                actual_ext = self.detect_real_extension(file_path)
                
                # Comparer
                if declared_ext != actual_ext and actual_ext != file_path.suffix.lower():
                    self.stats['mismatches'] += 1
                    
                    mismatches.append({
                        'url': url,
                        'source_url': source_url,
                        'file_path': str(file_path),
                        'declared_ext': declared_ext,
                        'current_ext': file_path.suffix.lower(),
                        'real_ext': actual_ext,
                    })
        
        # Grouper par type d'incohérence
        by_type = {}
        for m in mismatches:
            key = f"{m['declared_ext']} → {m['real_ext']}"
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(m)
        
        # Afficher résumé
        print(f"\n📊 Résultats :")
        print(f"   Fichiers analysés   : {self.stats['total_files']}")
        print(f"   Incohérences        : {self.stats['mismatches']}")
        
        if by_type:
            print(f"\n📋 Types d'incohérences :")
            for key, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
                print(f"   {key:30s} : {len(items):4d} fichiers")
        
        return {
            'mismatches': mismatches,
            'by_type': {k: len(v) for k, v in by_type.items()},
            'stats': self.stats
        }
    
    def fix_extensions(self, dry_run: bool = True) -> Dict:
        """Corrige les extensions (renomme les fichiers)"""
        
        print("\n" + "=" * 70)
        if dry_run:
            print("🧪 SIMULATION - CORRECTION DES EXTENSIONS")
        else:
            print("🔧 CORRECTION DES EXTENSIONS")
        print("=" * 70)
        
        # Analyser d'abord
        analysis = self.analyze_mismatches()
        mismatches = analysis['mismatches']
        
        if not mismatches:
            print("\n✅ Aucune incohérence détectée")
            return analysis
        
        print(f"\n🔧 {len(mismatches)} fichiers à corriger")
        
        if dry_run:
            print("\n⚠️  MODE SIMULATION - Aucune modification réelle")
            print("   Relancez avec --execute pour appliquer les changements")
        else:
            confirm = input("\n⚠️  Confirmer la correction ? (oui/non) : ")
            if confirm.lower() not in ['oui', 'yes', 'y', 'o']:
                print("❌ Annulé")
                return analysis
        
        # Charger index pour mise à jour
        with open(self.resource_index_file, 'r', encoding='utf-8') as f:
            resource_index = json.load(f)
        
        # Vider cache pour forcer re-analyse
        if not dry_run and self.cache_file.exists():
            print("\n🗑️  Vidage du cache pour forcer re-analyse...")
            self.cache_file.unlink()
        
        # Corriger
        renamed_files = []
        
        for mismatch in tqdm(mismatches, desc="Correction"):
            old_path = Path(mismatch['file_path'])
            real_ext = mismatch['real_ext']
            
            # Nouveau chemin
            new_path = old_path.with_suffix(real_ext)
            
            if dry_run:
                logger.info(f"SIMULATION: {old_path.name} → {new_path.name}")
            else:
                try:
                    # Renommer fichier
                    shutil.move(str(old_path), str(new_path))
                    logger.info(f"✅ {old_path.name} → {new_path.name}")
                    
                    # Mettre à jour index
                    for source_url, data in resource_index.items():
                        for resource in data.get('resources', []):
                            if resource['url'] == mismatch['url']:
                                resource['file_path'] = str(new_path.relative_to(self.project_root))
                                resource['file_type'] = real_ext[1:]  # Sans le point
                                break
                    
                    renamed_files.append({
                        'old': str(old_path),
                        'new': str(new_path),
                        'url': mismatch['url']
                    })
                    
                    # Stats
                    ext_key = f"{real_ext[1:]}_renamed"
                    if ext_key in self.stats:
                        self.stats[ext_key] += 1
                
                except Exception as e:
                    logger.error(f"❌ Erreur renommage {old_path.name}: {e}")
                    self.stats['errors'] += 1
        
        # Sauvegarder index mis à jour
        if not dry_run and renamed_files:
            print("\n💾 Mise à jour de l'index...")
            with open(self.resource_index_file, 'w', encoding='utf-8') as f:
                json.dump(resource_index, f, indent=2, ensure_ascii=False)
        
        # Résumé
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ")
        print("=" * 70)
        
        if dry_run:
            print(f"\n🧪 SIMULATION - Modifications qui seraient appliquées :")
        else:
            print(f"\n✅ Modifications appliquées :")
        
        print(f"   ODS renommés  : {self.stats.get('ods_renamed', 0)}")
        print(f"   XLSX renommés : {self.stats.get('xlsx_renamed', 0)}")
        print(f"   ODT renommés  : {self.stats.get('odt_renamed', 0)}")
        print(f"   DOCX renommés : {self.stats.get('docx_renamed', 0)}")
        print(f"   Autres        : {self.stats.get('other_renamed', 0)}")
        print(f"   Erreurs       : {self.stats.get('errors', 0)}")
        
        if not dry_run:
            print(f"\n💾 Index mis à jour : {self.resource_index_file}")
            print(f"💾 Cache vidé pour re-analyse")
        
        print("=" * 70)
        
        return {
            'renamed_files': renamed_files if not dry_run else [],
            'stats': self.stats,
            'analysis': analysis
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Correction des extensions de fichiers')
    parser.add_argument('--project-root', type=str, default='.', help='Racine du projet')
    parser.add_argument('--execute', action='store_true', help='Appliquer les changements (sinon simulation)')
    parser.add_argument('--analyze-only', action='store_true', help='Analyser seulement, pas de correction')
    
    args = parser.parse_args()
    
    fixer = ExtensionFixer(args.project_root)
    
    if args.analyze_only:
        # Analyse seulement
        fixer.analyze_mismatches()
    else:
        # Correction (simulation ou réelle)
        fixer.fix_extensions(dry_run=not args.execute)
        
        if not args.execute:
            print("\n💡 Pour appliquer les changements : python fix_extensions.py --execute")


if __name__ == "__main__":
    main()
