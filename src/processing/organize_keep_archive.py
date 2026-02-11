"""
Séparation Keep/Archive
Organise les fichiers pertinents et archive les autres
"""

import json
from pathlib import Path
import shutil
import logging
from typing import Dict, Set
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class KeepArchiveOrganizer:
    """Sépare les fichiers à garder de ceux à archiver"""
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.data_path = self.project_root / 'data'
        self.cnil_path = self.data_path / 'raw' / 'cnil'
        
        # Dossiers source
        self.html_dir = self.cnil_path / 'html'
        self.pdf_dir = self.cnil_path / 'pdf'
        self.docs_dir = self.cnil_path / 'docs'
        self.images_dir = self.cnil_path / 'images'
        self.metadata_dir = self.data_path / 'metadata'
        
        # Dossiers destination
        self.keep_dir = self.data_path / 'keep' / 'cnil'
        self.archive_dir = self.data_path / 'archive'
        
        # Fichiers de classification
        self.html_classification = self.cnil_path / 'hybrid_classification.json'
        
        # Manifest final
        self.manifest_file = self.cnil_path / 'keep_manifest.json'
        
        # Seuils de taille minimale (filtrage icônes / PDFs cassés)
        self.MIN_PDF_SIZE = 3 * 1024       # 3 KB — en dessous = PDF vide/cassé
        self.MIN_IMAGE_SIZE = 476 * 1024   # 476 KB — en dessous = icône/picto
        
        # Stats
        self.stats = {
            'html_kept': 0,
            'html_archived': 0,
            'pdf_kept': 0,
            'pdf_archived': 0,
            'pdf_skipped_size': 0,
            'docs_kept': 0,
            'docs_archived': 0,
            'images_kept': 0,
            'images_archived': 0,
            'images_skipped_size': 0,
        }
    
    def load_classifications(self):
        """Charge les classifications HTML"""
        
        # HTML
        if not self.html_classification.exists():
            raise FileNotFoundError(f"Classification HTML manquante : {self.html_classification}")
        
        with open(self.html_classification, 'r', encoding='utf-8') as f:
            self.html_data = json.load(f)
        
        logger.info(f"📄 Classification HTML chargée")
    
    def get_kept_html_urls(self) -> Set[str]:
        """Retourne les URLs HTML à garder"""
        kept = set()
        for hash_id, info in self.html_data.get('llm_classified', {}).items():
            if info.get('pertinent', False):
                kept.add(info['url'])
        return kept
    
    def _is_resource_linked_to_kept_html(self, metadata_file: Path, kept_html_urls: Set[str]) -> bool:
        """Vérifie si une ressource (PDF/doc) est liée à un HTML gardé via source_url."""
        if not metadata_file.exists():
            return False
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            source_url = metadata.get('source_url', '')
            return source_url in kept_html_urls
        except Exception:
            return False
    
    def organize_files(self, dry_run: bool = False):
        """Organise les fichiers (mode dry-run par défaut)"""
        
        print("=" * 70)
        if dry_run:
            print("🔍 SIMULATION - Aucun fichier ne sera déplacé")
        else:
            print("📦 ORGANISATION KEEP/ARCHIVE")
        print("=" * 70)
        
        # Charger classifications
        self.load_classifications()
        
        # Obtenir URLs HTML gardées
        kept_html_urls = self.get_kept_html_urls()
        
        logger.info(f"📄 {len(kept_html_urls)} HTML à garder")
        
        # Créer structure destination
        if not dry_run:
            for subdir in ['html', 'pdf', 'docs', 'images', 'metadata']:
                (self.keep_dir / subdir).mkdir(parents=True, exist_ok=True)
                (self.archive_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Manifest des fichiers gardés
        manifest = {
            'html': [],
            'pdfs': [],
            'docs': [],
            'images': [],
            'relationships': {}  # HTML -> [resources]
        }
        
        # 1. Traiter HTML
        print("\n📄 Traitement des fichiers HTML...")
        
        for html_file in tqdm(list(self.html_dir.glob('*.html')), desc="HTML"):
            # Charger métadonnées
            metadata_file = self.metadata_dir / f"{html_file.stem}.json"
            if not metadata_file.exists():
                continue
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                url = metadata.get('url', '')
            
            # Garder ou archiver
            if url in kept_html_urls:
                dest_dir = self.keep_dir
                self.stats['html_kept'] += 1
                
                # Ajouter au manifest
                manifest['html'].append({
                    'url': url,
                    'file': f"html/{html_file.name}",
                    'metadata': metadata
                })
                manifest['relationships'][url] = []
            else:
                dest_dir = self.archive_dir
                self.stats['html_archived'] += 1
            
            # Copier
            if not dry_run:
                shutil.copy2(html_file, dest_dir / 'html' / html_file.name)
                shutil.copy2(metadata_file, dest_dir / 'metadata' / metadata_file.name)
        
        # 2. Traiter PDFs — garder si parent_url est un HTML gardé
        print("\n📄 Traitement des PDFs...")
        
        for pdf_file in tqdm(list(self.pdf_dir.glob('*.pdf')), desc="PDFs"):
            # Filtre taille : PDFs < 3 KB = vides/cassés
            if pdf_file.stat().st_size < self.MIN_PDF_SIZE:
                self.stats['pdf_skipped_size'] += 1
                continue
            
            metadata_file = self.metadata_dir / f"{pdf_file.stem}.json"
            
            is_kept = self._is_resource_linked_to_kept_html(metadata_file, kept_html_urls)
            
            if is_kept:
                dest_dir = self.keep_dir
                self.stats['pdf_kept'] += 1
                
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    manifest['pdfs'].append({
                        'url': metadata.get('url', ''),
                        'file': f"pdf/{pdf_file.name}",
                        'parent_url': metadata.get('source_url', ''),
                        'metadata': metadata
                    })
                    
                    parent = metadata.get('source_url')
                    if parent and parent in manifest['relationships']:
                        manifest['relationships'][parent].append({
                            'type': 'pdf',
                            'file': f"pdf/{pdf_file.name}"
                        })
            else:
                dest_dir = self.archive_dir
                self.stats['pdf_archived'] += 1
            
            if not dry_run:
                shutil.copy2(pdf_file, dest_dir / 'pdf' / pdf_file.name)
                if metadata_file.exists():
                    shutil.copy2(metadata_file, dest_dir / 'metadata' / metadata_file.name)
        
        # 3. Traiter Documents — garder si parent_url est un HTML gardé
        print("\n📝 Traitement des documents...")
        
        for doc_file in tqdm(list(self.docs_dir.glob('*')), desc="Docs"):
            if doc_file.is_file():
                metadata_file = self.metadata_dir / f"{doc_file.stem}.json"
                
                is_kept = self._is_resource_linked_to_kept_html(metadata_file, kept_html_urls)
                
                if is_kept:
                    dest_dir = self.keep_dir
                    self.stats['docs_kept'] += 1
                    
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        manifest['docs'].append({
                            'url': metadata.get('url', ''),
                            'file': f"docs/{doc_file.name}",
                            'parent_url': metadata.get('source_url', ''),
                            'metadata': metadata
                        })
                        
                        parent = metadata.get('source_url')
                        if parent and parent in manifest['relationships']:
                            manifest['relationships'][parent].append({
                                'type': 'doc',
                                'file': f"docs/{doc_file.name}"
                            })
                else:
                    dest_dir = self.archive_dir
                    self.stats['docs_archived'] += 1
                
                if not dry_run:
                    shutil.copy2(doc_file, dest_dir / 'docs' / doc_file.name)
                    if metadata_file.exists():
                        shutil.copy2(metadata_file, dest_dir / 'metadata' / metadata_file.name)
        
        # 4. Traiter Images — garder si parent_url est un HTML gardé
        #    Note: les images archivées restent dans raw/ (pas de copie inutile)
        print("\n🖼️  Traitement des images...")
        
        for img_file in tqdm(list(self.images_dir.glob('*')), desc="Images"):
            if img_file.is_file():
                # Filtre taille : images < 476 KB = icônes/pictos
                if img_file.stat().st_size < self.MIN_IMAGE_SIZE:
                    self.stats['images_skipped_size'] += 1
                    continue
                
                metadata_file = self.metadata_dir / f"{img_file.stem}.json"
                
                is_kept = self._is_resource_linked_to_kept_html(metadata_file, kept_html_urls)
                
                if is_kept:
                    self.stats['images_kept'] += 1
                    
                    if not dry_run:
                        shutil.copy2(img_file, self.keep_dir / 'images' / img_file.name)
                        if metadata_file.exists():
                            shutil.copy2(metadata_file, self.keep_dir / 'metadata' / metadata_file.name)
                    
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        manifest['images'].append({
                            'url': metadata.get('url', ''),
                            'file': f"images/{img_file.name}",
                            'parent_url': metadata.get('source_url', ''),
                            'metadata': metadata
                        })
                        
                        parent = metadata.get('source_url')
                        if parent and parent in manifest['relationships']:
                            manifest['relationships'][parent].append({
                                'type': 'image',
                                'file': f"images/{img_file.name}"
                            })
                else:
                    self.stats['images_archived'] += 1
                    # Pas de copie vers archive — les images restent dans raw/
        
        # Sauvegarder manifest
        if not dry_run:
            with open(self.manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Manifest sauvegardé : {self.manifest_file}")
        
        # Résumé
        self._print_summary(dry_run)
    
    def _print_summary(self, dry_run: bool):
        """Affiche le résumé"""
        print("\n" + "=" * 70)
        if dry_run:
            print("📊 RÉSUMÉ SIMULATION")
        else:
            print("📊 RÉSUMÉ ORGANISATION")
        print("=" * 70)
        
        print(f"\n📄 HTML :")
        print(f"   Gardés   : {self.stats['html_kept']}")
        print(f"   Archivés : {self.stats['html_archived']}")
        
        print(f"\n📄 PDFs :")
        print(f"   Gardés   : {self.stats['pdf_kept']}")
        print(f"   Archivés : {self.stats['pdf_archived']}")
        if self.stats['pdf_skipped_size'] > 0:
            print(f"   Ignorés  : {self.stats['pdf_skipped_size']} (< {self.MIN_PDF_SIZE // 1024} KB — vides/cassés)")
        
        print(f"\n📝 Documents :")
        print(f"   Gardés   : {self.stats['docs_kept']}")
        print(f"   Archivés : {self.stats['docs_archived']}")
        
        print(f"\n🖼️  Images :")
        print(f"   Gardées   : {self.stats['images_kept']}")
        print(f"   Archivées : {self.stats['images_archived']}")
        if self.stats['images_skipped_size'] > 0:
            print(f"   Ignorées  : {self.stats['images_skipped_size']} (< {self.MIN_IMAGE_SIZE // 1024} KB — icônes/pictos)")
        
        total_kept = (self.stats['html_kept'] + self.stats['pdf_kept'] + 
                     self.stats['docs_kept'] + self.stats['images_kept'])
        total_archived = (self.stats['html_archived'] + self.stats['pdf_archived'] + 
                         self.stats['docs_archived'] + self.stats['images_archived'])
        total_skipped = self.stats['pdf_skipped_size'] + self.stats['images_skipped_size']
        
        print(f"\n📊 TOTAL :")
        print(f"   Gardés   : {total_kept}")
        print(f"   Archivés : {total_archived}")
        if total_skipped > 0:
            print(f"   Ignorés  : {total_skipped} (trop petits)")
        total_processed = total_kept + total_archived
        print(f"   Ratio    : {total_kept/total_processed*100:.1f}% gardés" if total_processed > 0 else "   Ratio    : N/A")
        
        if not dry_run:
            print(f"\n📁 Dossiers :")
            print(f"   Keep    : {self.keep_dir}")
            print(f"   Archive : {self.archive_dir}")
            print(f"   Manifest: {self.manifest_file}")
        
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Organisation Keep/Archive')
    parser.add_argument('--project-root', type=str, default='.', help='Racine du projet')
    parser.add_argument('--execute', action='store_true', help='Exécuter vraiment (sinon simulation)')
    parser.add_argument('--yes', '-y', action='store_true', help='Pas de confirmation interactive')
    parser.add_argument('--clean', action='store_true', help='Nettoyer keep/ avant de copier (supprimer anciens fichiers)')
    
    args = parser.parse_args()
    
    organizer = KeepArchiveOrganizer(args.project_root)
    
    # Nettoyer keep/ si demandé
    if args.clean and args.execute:
        import shutil as _shutil
        keep_dir = organizer.keep_dir
        if keep_dir.exists():
            print(f"\n🗑️  Nettoyage de {keep_dir}...")
            for subdir in ['html', 'pdf', 'docs', 'images', 'metadata']:
                target = keep_dir / subdir
                if target.exists():
                    count = sum(1 for _ in target.iterdir())
                    _shutil.rmtree(target)
                    target.mkdir(parents=True, exist_ok=True)
                    print(f"   Supprimé {subdir}/ ({count} fichiers)")
    
    # Dry-run par défaut pour sécurité
    dry_run = not args.execute
    
    if dry_run:
        print("\n⚠️  MODE SIMULATION (aucun fichier ne sera déplacé)")
        print("   Pour exécuter vraiment, utilisez --execute\n")
    elif not args.yes:
        print("\n⚠️  MODE EXÉCUTION - Les fichiers seront COPIÉS")
        response = input("   Confirmer ? (O/N) [N] : ").strip().upper()
        if response != 'O':
            print("   ⏭️  Annulé")
            return
    
    organizer.organize_files(dry_run=dry_run)


if __name__ == "__main__":
    main()
