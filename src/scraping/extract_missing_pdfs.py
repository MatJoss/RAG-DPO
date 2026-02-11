"""
Extraction des Fichiers Manquants depuis HTML Existants
Analyse les HTML déjà téléchargés et récupère les PDF/ODT/XLSX/ODS/IMAGES
VERSION COMPLÈTE avec support images
"""

import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from tqdm import tqdm
import logging
import requests
import time
import mimetypes
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingFilesExtractor:
    """Extrait les fichiers manquants depuis les HTML existants"""
    
    FILE_EXTENSIONS = {
        '.pdf': 'pdf',
        '.odt': 'odt',
        '.ods': 'ods',  # ← Ajouté explicitement
        '.docx': 'docx',
        '.doc': 'doc',
        '.xlsx': 'xlsx',
        '.xls': 'xls',
    }
    
    IMAGE_EXTENSIONS = {
        '.jpg': 'jpg',
        '.jpeg': 'jpeg',
        '.png': 'png',
        '.gif': 'gif',
        '.webp': 'webp',
    }
    
    def __init__(self, project_root: str = '.', download_images: bool = True):
        self.project_root = Path(project_root)
        self.html_dir = self.project_root / 'data' / 'raw' / 'html'
        self.metadata_dir = self.project_root / 'data' / 'metadata'
        self.pdf_dir = self.project_root / 'data' / 'raw' / 'pdf'
        self.docs_dir = self.project_root / 'data' / 'raw' / 'docs'
        self.images_dir = self.project_root / 'data' / 'raw' / 'images'
        
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.download_images = download_images
        
        # Session HTTP
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'RAG-DPO-Extractor/1.0',
            'Accept': '*/*',
        })
        
        self.stats = {
            'html_analyzed': 0,
            'doc_links_found': 0,
            'image_links_found': 0,
            'pdf_downloaded': 0,
            'docs_downloaded': 0,
            'images_downloaded': 0,
            'already_exists': 0,
            'errors': 0,
        }
    
    def extract_file_links_from_html(self, html_file: Path, base_url: str) -> dict:
        """Extrait TOUS les liens vers fichiers d'un HTML"""
        try:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
            
            file_links = {}
            
            # Documents (PDF, ODT, XLSX, ODS, etc.)
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)
                url_lower = absolute_url.lower()
                
                for ext in self.FILE_EXTENSIONS.keys():
                    if ext in url_lower:
                        file_links[absolute_url] = base_url
                        break
            
            # Chercher aussi dans <embed>, <object>
            for tag in soup.find_all(['embed', 'object'], src=True):
                src = urljoin(base_url, tag['src'])
                url_lower = src.lower()
                for ext in self.FILE_EXTENSIONS.keys():
                    if ext in url_lower:
                        file_links[src] = base_url
                        break
            
            for tag in soup.find_all(['embed', 'object'], data=True):
                data = urljoin(base_url, tag['data'])
                url_lower = data.lower()
                for ext in self.FILE_EXTENSIONS.keys():
                    if ext in url_lower:
                        file_links[data] = base_url
                        break
            
            return file_links
        
        except Exception as e:
            logger.warning(f"⚠️  Erreur extraction docs {html_file.name}: {e}")
            return {}
    
    def extract_image_links_from_html(self, html_file: Path, base_url: str) -> dict:
        """Extrait les images PERTINENTES d'un HTML"""
        if not self.download_images:
            return {}
        
        try:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
            
            # Supprimer header, footer, nav, aside pour ne garder que le contenu
            for tag in soup(['header', 'footer', 'nav', 'aside', 'script', 'style']):
                tag.decompose()
            
            main_content = soup.find(['main', 'article', 'div']) or soup
            
            image_links = {}
            
            for img in main_content.find_all('img', src=True):
                src = img['src']
                alt = img.get('alt', '').lower()
                
                # URL absolue
                absolute_url = urljoin(base_url, src)
                
                # Filtrer icônes par URL
                if any(skip in absolute_url.lower() for skip in 
                       ['icon', 'logo', 'picto', 'bullet', 'arrow', 'sprite', 'thumb']):
                    continue
                
                # Filtrer par alt text
                if any(skip in alt for skip in ['logo', 'icône', 'pictogramme']):
                    continue
                
                # Filtrer petites images déclarées en HTML
                width = img.get('width', '')
                height = img.get('height', '')
                if width and height:
                    try:
                        if int(width) < 100 or int(height) < 100:
                            continue
                    except:
                        pass
                
                # Vérifier extension valide
                url_lower = absolute_url.lower()
                valid = False
                for ext in self.IMAGE_EXTENSIONS.keys():
                    if ext in url_lower:
                        valid = True
                        break
                
                if valid:
                    image_links[absolute_url] = base_url
            
            return image_links
        
        except Exception as e:
            logger.warning(f"⚠️  Erreur extraction images {html_file.name}: {e}")
            return {}
    
    def get_base_url_from_metadata(self, html_hash: str) -> str:
        """Récupère l'URL d'origine depuis les métadonnées"""
        metadata_file = self.metadata_dir / f"{html_hash}.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    return metadata.get('url', 'https://www.cnil.fr')
            except:
                pass
        
        return 'https://www.cnil.fr'
    
    def get_file_hash(self, url: str) -> str:
        """Hash pour nom de fichier"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def file_already_exists(self, url: str, is_image: bool = False) -> bool:
        """Vérifie si le fichier a déjà été téléchargé"""
        file_hash = self.get_file_hash(url)
        
        if is_image:
            # Chercher dans images/
            for ext in self.IMAGE_EXTENSIONS.keys():
                if (self.images_dir / f"{file_hash}{ext}").exists():
                    return True
        else:
            # Chercher dans pdf/
            if (self.pdf_dir / f"{file_hash}.pdf").exists():
                return True
            
            # Chercher dans docs/
            for ext in ['.odt', '.ods', '.docx', '.doc', '.xlsx', '.xls']:
                if (self.docs_dir / f"{file_hash}{ext}").exists():
                    return True
        
        return False
    
    def download_file(self, url: str, source_url: str, is_image: bool = False) -> bool:
        """Télécharge un fichier avec métadonnées complètes"""
        try:
            time.sleep(0.3)  # Rate limiting
            
            response = self.session.get(url, timeout=30, allow_redirects=True, stream=True)
            response.raise_for_status()
            
            content = response.content
            content_type = response.headers.get('Content-Type', '')
            
            # Déterminer type et extension
            url_lower = url.lower()
            
            if is_image:
                # Images
                if '.png' in url_lower or 'image/png' in content_type:
                    ext = '.png'
                    file_type = 'png'
                elif '.jpg' in url_lower or '.jpeg' in url_lower or 'image/jpeg' in content_type:
                    ext = '.jpg'
                    file_type = 'jpeg'
                elif '.gif' in url_lower or 'image/gif' in content_type:
                    ext = '.gif'
                    file_type = 'gif'
                elif '.webp' in url_lower or 'image/webp' in content_type:
                    ext = '.webp'
                    file_type = 'webp'
                else:
                    ext = '.jpg'  # Par défaut
                    file_type = 'jpeg'
                
                output_dir = self.images_dir
            
            else:
                # Documents
                if '.pdf' in url_lower or 'application/pdf' in content_type:
                    file_type = 'pdf'
                    ext = '.pdf'
                    output_dir = self.pdf_dir
                
                elif '.ods' in url_lower or 'opendocument.spreadsheet' in content_type:
                    file_type = 'ods'
                    ext = '.ods'
                    output_dir = self.docs_dir
                
                elif '.odt' in url_lower or 'opendocument.text' in content_type:
                    file_type = 'odt'
                    ext = '.odt'
                    output_dir = self.docs_dir
                
                elif '.docx' in url_lower or 'word' in content_type:
                    file_type = 'docx'
                    ext = '.docx'
                    output_dir = self.docs_dir
                
                elif '.doc' in url_lower:
                    file_type = 'doc'
                    ext = '.doc'
                    output_dir = self.docs_dir
                
                elif '.xlsx' in url_lower or 'excel' in content_type or 'spreadsheetml' in content_type:
                    file_type = 'xlsx'
                    ext = '.xlsx'
                    output_dir = self.docs_dir
                
                elif '.xls' in url_lower:
                    file_type = 'xls'
                    ext = '.xls'
                    output_dir = self.docs_dir
                
                else:
                    logger.warning(f"⚠️  Type inconnu: {url[:60]}... ({content_type})")
                    return False
            
            # Sauvegarder
            file_hash = self.get_file_hash(url)
            file_path = output_dir / f"{file_hash}{ext}"
            file_path.write_bytes(content)
            
            # Métadonnées COMPLÈTES
            metadata = {
                'url': url,
                'source_url': source_url,
                'file_type': file_type,
                'file_path': str(file_path.relative_to(self.project_root)),
                'file_size_bytes': len(content),
                'content_type': content_type,
                'downloaded_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            }
            
            metadata_path = self.metadata_dir / f"{file_hash}.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Stats
            if is_image:
                self.stats['images_downloaded'] += 1
            elif file_type == 'pdf':
                self.stats['pdf_downloaded'] += 1
            else:
                self.stats['docs_downloaded'] += 1
            
            return True
        
        except Exception as e:
            logger.debug(f"⚠️  Erreur download {url[:60]}...: {e}")
            self.stats['errors'] += 1
            return False
    
    def run(self):
        """Lance l'extraction complète"""
        
        print("=" * 70)
        print("📥 EXTRACTION DES FICHIERS MANQUANTS")
        print("=" * 70)
        
        # Charger tous les HTML
        html_files = list(self.html_dir.glob('*.html'))
        
        print(f"\n📄 {len(html_files)} fichiers HTML à analyser")
        
        if self.download_images:
            print(f"🖼️  Téléchargement des images : ✅ ACTIVÉ")
        else:
            print(f"🖼️  Téléchargement des images : ❌ DÉSACTIVÉ")
        
        print("\n🔍 Analyse des HTML et extraction des liens...")
        
        all_doc_links = {}
        all_image_links = {}
        
        for html_file in tqdm(html_files, desc="Analyse HTML"):
            html_hash = html_file.stem
            base_url = self.get_base_url_from_metadata(html_hash)
            
            # Extraire documents
            doc_links = self.extract_file_links_from_html(html_file, base_url)
            all_doc_links.update(doc_links)
            
            # Extraire images
            if self.download_images:
                image_links = self.extract_image_links_from_html(html_file, base_url)
                all_image_links.update(image_links)
            
            self.stats['html_analyzed'] += 1
        
        self.stats['doc_links_found'] = len(all_doc_links)
        self.stats['image_links_found'] = len(all_image_links)
        
        print(f"\n📊 Liens trouvés :")
        print(f"   Documents : {self.stats['doc_links_found']}")
        print(f"   Images    : {self.stats['image_links_found']}")
        
        # Télécharger documents
        if all_doc_links:
            print(f"\n📥 Téléchargement des documents...")
            
            for url, source_url in tqdm(all_doc_links.items(), desc="Documents"):
                if self.file_already_exists(url, is_image=False):
                    self.stats['already_exists'] += 1
                    continue
                
                self.download_file(url, source_url, is_image=False)
        
        # Télécharger images
        if all_image_links:
            print(f"\n🖼️  Téléchargement des images...")
            
            for url, source_url in tqdm(all_image_links.items(), desc="Images"):
                if self.file_already_exists(url, is_image=True):
                    self.stats['already_exists'] += 1
                    continue
                
                self.download_file(url, source_url, is_image=True)
        
        # Créer index par source
        self._create_resource_index()
        
        # Résumé
        self._print_summary()
    
    def _create_resource_index(self):
        """Crée un index : URL source -> liste ressources"""
        print("\n🗂️  Création de l'index des ressources...")
        
        index = {}
        
        # Parcourir toutes les métadonnées
        for metadata_file in self.metadata_dir.glob('*.json'):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                source_url = metadata.get('source_url')
                if not source_url:
                    continue
                
                if source_url not in index:
                    index[source_url] = {
                        'source_url': source_url,
                        'resources': []
                    }
                
                index[source_url]['resources'].append({
                    'url': metadata['url'],
                    'file_type': metadata['file_type'],
                    'file_path': metadata['file_path'],
                    'file_size_bytes': metadata.get('file_size_bytes', 0),
                })
            
            except:
                pass
        
        # Sauvegarder
        output_file = self.metadata_dir / 'resource_index_by_source.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Index sauvegardé : {output_file}")
        logger.info(f"📊 {len(index)} pages HTML avec ressources")
    
    def _print_summary(self):
        """Affiche le résumé"""
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ")
        print("=" * 70)
        
        print(f"\n📄 HTML analysés      : {self.stats['html_analyzed']}")
        print(f"\n📎 Liens trouvés      :")
        print(f"   Documents          : {self.stats['doc_links_found']}")
        print(f"   Images             : {self.stats['image_links_found']}")
        print(f"\n📥 Téléchargements    :")
        print(f"   PDFs               : {self.stats['pdf_downloaded']}")
        print(f"   Documents          : {self.stats['docs_downloaded']}")
        print(f"   Images             : {self.stats['images_downloaded']}")
        print(f"   Déjà existants     : {self.stats['already_exists']}")
        print(f"   Erreurs            : {self.stats['errors']}")
        
        print("\n" + "=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extraction fichiers manquants')
    parser.add_argument('--project-root', type=str, default='.', help='Racine du projet')
    parser.add_argument('--no-images', action='store_true', help='Ne pas télécharger les images')
    
    args = parser.parse_args()
    
    extractor = MissingFilesExtractor(
        project_root=args.project_root,
        download_images=not args.no_images
    )
    
    extractor.run()


if __name__ == "__main__":
    main()