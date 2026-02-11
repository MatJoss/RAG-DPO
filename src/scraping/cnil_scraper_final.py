"""
CNIL Scraper - Version FINALE Production
Corrige TOUS les bugs, gestion robuste, métadonnées complètes
"""

import os
import time
import json
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Set, Dict, List, Optional, Tuple
from datetime import datetime
import shutil

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm
import logging

# Setup logging avec rotation
from logging.handlers import RotatingFileHandler

def setup_logging(log_dir: Path):
    """Configure le logging avec rotation"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'scraping_final.log'
    
    # Handler avec rotation (10MB max, 5 backups)
    handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
    )
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler())
    
    return logger

load_dotenv()


class CNILScraperFinal:
    """Scraper CNIL version production - robuste et complet"""
    
    MIME_CATEGORIES = {
        'text/html': 'html',
        'application/xhtml+xml': 'html',
        'application/pdf': 'pdf',
        'application/vnd.oasis.opendocument.text': 'odt',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/msword': 'doc',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
        'application/vnd.ms-excel': 'xls',
        'application/vnd.oasis.opendocument.spreadsheet': 'ods',
        'image/jpeg': 'image',
        'image/png': 'image',
        'image/gif': 'image',
        'image/webp': 'image',
        'image/svg+xml': 'image',
    }
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.data_path = self.project_root / 'data'
        
        # Structure de sortie
        self.output_dirs = {
            'html': self.data_path / 'raw' / 'html',
            'pdf': self.data_path / 'raw' / 'pdf',
            'docs': self.data_path / 'raw' / 'docs',
            'images': self.data_path / 'raw' / 'images',
            'metadata': self.data_path / 'metadata',
            'logs': self.project_root / 'logs',
        }
        
        for dir_path in self.output_dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.logger = setup_logging(self.output_dirs['logs'])
        
        # Config
        self.delay = float(os.getenv('DELAY_BETWEEN_REQUESTS', 2))
        self.max_retries = int(os.getenv('MAX_RETRIES', 3))
        self.user_agent = os.getenv('USER_AGENT', 'RAG-DPO-Bot/Final')
        
        # Session HTTP
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept': '*/*',
            'Accept-Language': 'fr-FR,fr;q=0.9',
        })
        
        # État
        self.state_file = self.output_dirs['metadata'] / 'scraping_state_final.json'
        self.visited_urls: Set[str] = set()
        self.downloaded_resources: Dict[str, str] = {}
        self.failed_urls: List[Dict] = []
        
        # Stats (réinitialisées à chaque run, pas écrasées)
        self.session_stats = {
            'html': 0,
            'pdf': 0,
            'docs': 0,
            'images': 0,
            'errors': 0,
            'total_size_mb': 0,
            'start_time': datetime.now().isoformat(),
        }
        
        # Charger état précédent
        self._load_state()
        
        self.logger.info("🚀 Scraper Final initialisé")
        self.logger.info(f"   URLs déjà visitées : {len(self.visited_urls)}")
    
    def _load_state(self):
        """Charge l'état SANS écraser"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                
                # Charger ce qui existe
                self.visited_urls = set(state.get('visited_urls', []))
                self.downloaded_resources = state.get('downloaded_resources', {})
                self.failed_urls = state.get('failed_urls', [])
                
                # Les stats globales (cumulées)
                self.global_stats = state.get('global_stats', {
                    'total_html': 0,
                    'total_pdf': 0,
                    'total_docs': 0,
                    'total_images': 0,
                    'total_errors': 0,
                    'total_size_mb': 0,
                })
                
                self.logger.info(f"📂 État chargé : {len(self.visited_urls)} URLs déjà visitées")
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur chargement état : {e}")
                self.global_stats = {
                    'total_html': 0,
                    'total_pdf': 0,
                    'total_docs': 0,
                    'total_images': 0,
                    'total_errors': 0,
                    'total_size_mb': 0,
                }
        else:
            self.global_stats = {
                'total_html': 0,
                'total_pdf': 0,
                'total_docs': 0,
                'total_images': 0,
                'total_errors': 0,
                'total_size_mb': 0,
            }
    
    def _save_state(self):
        """Sauvegarde l'état de manière SÉCURISÉE"""
        # Calculer stats globales cumulées
        cumulated_stats = {
            'total_html': self.global_stats['total_html'] + self.session_stats['html'],
            'total_pdf': self.global_stats['total_pdf'] + self.session_stats['pdf'],
            'total_docs': self.global_stats['total_docs'] + self.session_stats['docs'],
            'total_images': self.global_stats['total_images'] + self.session_stats['images'],
            'total_errors': self.global_stats['total_errors'] + self.session_stats['errors'],
            'total_size_mb': self.global_stats['total_size_mb'] + self.session_stats['total_size_mb'],
        }
        
        state = {
            'visited_urls': list(self.visited_urls),
            'downloaded_resources': self.downloaded_resources,
            'failed_urls': self.failed_urls,
            'global_stats': cumulated_stats,
            'session_stats': self.session_stats,
            'last_update': datetime.now().isoformat(),
            'scraper_version': 'final',
        }
        
        # Sauvegarde atomique (évite corruption)
        temp_file = self.state_file.with_suffix('.json.tmp')
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            
            # Remplacer atomiquement
            shutil.move(str(temp_file), str(self.state_file))
            
        except Exception as e:
            self.logger.error(f"❌ Erreur sauvegarde état : {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def _get_url_hash(self, url: str) -> str:
        """Hash unique pour URL"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _detect_file_type(self, url: str, content_type: Optional[str] = None) -> str:
        """Détection multi-niveaux du type de fichier"""
        # 1. Content-Type HTTP (priorité)
        if content_type:
            ct_clean = content_type.split(';')[0].strip().lower()
            if ct_clean in self.MIME_CATEGORIES:
                return self.MIME_CATEGORIES[ct_clean]
        
        # 2. Extension URL
        url_lower = url.lower()
        ext_mapping = {
            '.pdf': 'pdf',
            '.odt': 'odt',
            '.docx': 'docx', '.doc': 'docx',
            '.xlsx': 'xlsx', '.xls': 'xlsx', '.ods': 'xlsx',
            '.jpg': 'image', '.jpeg': 'image', '.png': 'image',
            '.gif': 'image', '.webp': 'image', '.svg': 'image',
        }
        
        for ext, file_type in ext_mapping.items():
            if url_lower.endswith(ext):
                return file_type
        
        # 3. Mimetypes guess
        guessed, _ = mimetypes.guess_type(url)
        if guessed and guessed in self.MIME_CATEGORIES:
            return self.MIME_CATEGORIES[guessed]
        
        return 'html'
    
    def _download_file(self, url: str, retry: int = 0) -> Optional[Tuple[bytes, str]]:
        """Télécharge avec retry et retourne (content, content_type)"""
        try:
            time.sleep(self.delay)
            
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            return response.content, content_type
        
        except Exception as e:
            if retry < self.max_retries:
                self.logger.warning(f"⚠️  Retry {retry+1}/{self.max_retries} : {url}")
                time.sleep(self.delay * (retry + 1))
                return self._download_file(url, retry + 1)
            else:
                self.logger.error(f"❌ Échec : {url} - {str(e)}")
                self.session_stats['errors'] += 1
                self.failed_urls.append({
                    'url': url,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })
                return None
    
    def _save_file_with_metadata(self, url: str, content: bytes, file_type: str, 
                                 source_url: str = None) -> str:
        """Sauvegarde fichier + métadonnées complètes"""
        url_hash = self._get_url_hash(url)
        
        # Déterminer extension et dossier
        parsed = urlparse(url)
        ext = Path(parsed.path).suffix or {
            'pdf': '.pdf', 'odt': '.odt', 'docx': '.docx',
            'xlsx': '.xlsx', 'image': '.jpg', 'html': '.html'
        }.get(file_type, '.bin')
        
        # Dossier de sortie
        dir_map = {
            'pdf': self.output_dirs['pdf'],
            'odt': self.output_dirs['docs'],
            'docx': self.output_dirs['docs'],
            'xlsx': self.output_dirs['docs'],
            'image': self.output_dirs['images'],
            'html': self.output_dirs['html'],
        }
        output_dir = dir_map.get(file_type, self.output_dirs['html'])
        
        # Sauvegarder fichier
        file_path = output_dir / f"{url_hash}{ext}"
        file_path.write_bytes(content)
        
        # Métadonnées COMPLÈTES
        metadata = {
            'url': url,
            'url_hash': url_hash,
            'file_type': file_type,
            'extension': ext,
            'file_path': str(file_path.relative_to(self.project_root)),
            'file_size_bytes': len(content),
            'source_url': source_url,  # URL de la page qui contient ce fichier
            'scraped_at': datetime.now().isoformat(),
        }
        
        metadata_path = self.output_dirs['metadata'] / f"{url_hash}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Stats
        size_mb = len(content) / (1024 * 1024)
        self.session_stats['total_size_mb'] += size_mb
        
        if file_type == 'pdf':
            self.session_stats['pdf'] += 1
        elif file_type in ['odt', 'docx', 'xlsx']:
            self.session_stats['docs'] += 1
        elif file_type == 'image':
            self.session_stats['images'] += 1
        elif file_type == 'html':
            self.session_stats['html'] += 1
        
        self.logger.debug(f"✓ {file_type.upper()} : {url}")
        
        return str(file_path)
    
    def _extract_resources_from_html(self, soup: BeautifulSoup, base_url: str) -> Dict:
        """Extrait TOUTES les ressources d'un HTML"""
        resources = {'links': [], 'images': [], 'embeds': [], 'iframes': []}
        
        # Liens
        for link in soup.find_all('a', href=True):
            href = urljoin(base_url, link['href'])
            if 'cnil.fr' in href:
                resources['links'].append(href.split('#')[0])
        
        # Images
        for img in soup.find_all('img', src=True):
            src = urljoin(base_url, img['src'])
            if 'cnil.fr' in src:
                resources['images'].append(src)
        
        # Embeds et objects
        for embed in soup.find_all(['embed', 'object'], src=True):
            src = urljoin(base_url, embed['src'])
            if 'cnil.fr' in src:
                resources['embeds'].append(src)
        
        for embed in soup.find_all(['embed', 'object'], data=True):
            src = urljoin(base_url, embed['data'])
            if 'cnil.fr' in src:
                resources['embeds'].append(src)
        
        # iframes
        for iframe in soup.find_all('iframe', src=True):
            src = urljoin(base_url, iframe['src'])
            if 'cnil.fr' in src:
                resources['iframes'].append(src)
        
        return resources
    
    def scrape_url(self, url: str, source_url: str = None) -> Optional[BeautifulSoup]:
        """Scrape une URL avec gestion complète"""
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        
        result = self._download_file(url)
        if result is None:
            return None
        
        content, content_type = result
        file_type = self._detect_file_type(url, content_type)
        
        # Sauvegarder
        local_path = self._save_file_with_metadata(url, content, file_type, source_url)
        self.downloaded_resources[url] = local_path
        
        # Si HTML, extraire ressources
        if file_type == 'html':
            try:
                soup = BeautifulSoup(content, 'lxml')
                resources = self._extract_resources_from_html(soup, url)
                
                # Télécharger ressources embarquées
                for img_url in set(resources['images']):
                    if img_url not in self.visited_urls:
                        self.scrape_url(img_url, source_url=url)
                
                for embed_url in set(resources['embeds']):
                    if embed_url not in self.visited_urls:
                        self.scrape_url(embed_url, source_url=url)
                
                return soup
            except Exception as e:
                self.logger.error(f"❌ Erreur parsing HTML {url}: {e}")
                return None
        
        return None
    
    def scrape_recursive(self, start_url: str = None, max_depth: int = 5):
        """Scraping récursif avec gestion robuste"""
        if start_url is None:
            start_url = "https://www.cnil.fr/fr/professionnel"
        
        self.logger.info(f"🚀 Début scraping depuis : {start_url}")
        self.logger.info(f"📊 Profondeur max : {max_depth}")
        
        to_visit = [(start_url, 0)]
        
        with tqdm(desc="Scraping", unit="URL") as pbar:
            while to_visit:
                current_url, depth = to_visit.pop(0)
                
                if depth > max_depth or current_url in self.visited_urls:
                    continue
                
                pbar.set_description(
                    f"D{depth} - H:{self.session_stats['html']} "
                    f"P:{self.session_stats['pdf']} D:{self.session_stats['docs']}"
                )
                
                soup = self.scrape_url(current_url)
                
                if soup:
                    resources = self._extract_resources_from_html(soup, current_url)
                    for link in resources['links']:
                        if link not in self.visited_urls and '/fr/' in link:
                            to_visit.append((link, depth + 1))
                
                pbar.update(1)
                
                # Sauvegarde régulière
                if len(self.visited_urls) % 50 == 0:
                    self._save_state()
        
        self._save_state()
        self._print_summary()
    
    def _print_summary(self):
        """Résumé complet"""
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ DU SCRAPING")
        print("=" * 70)
        
        # Stats session
        print(f"\n📄 Cette session :")
        print(f"   HTML      : {self.session_stats['html']}")
        print(f"   PDF       : {self.session_stats['pdf']}")
        print(f"   Documents : {self.session_stats['docs']}")
        print(f"   Images    : {self.session_stats['images']}")
        print(f"   Erreurs   : {self.session_stats['errors']}")
        print(f"   Taille    : {self.session_stats['total_size_mb']:.2f} MB")
        
        # Stats globales
        cumul = {
            'html': self.global_stats['total_html'] + self.session_stats['html'],
            'pdf': self.global_stats['total_pdf'] + self.session_stats['pdf'],
            'docs': self.global_stats['total_docs'] + self.session_stats['docs'],
            'images': self.global_stats['total_images'] + self.session_stats['images'],
        }
        
        print(f"\n📊 TOTAL cumulé :")
        print(f"   HTML      : {cumul['html']}")
        print(f"   PDF       : {cumul['pdf']}")
        print(f"   Documents : {cumul['docs']}")
        print(f"   Images    : {cumul['images']}")
        
        print(f"\n🌐 URLs visitées : {len(self.visited_urls)}")
        print(f"📁 Ressources téléchargées : {len(self.downloaded_resources)}")
        
        if self.failed_urls:
            print(f"\n⚠️  {len(self.failed_urls)} URLs en échec (voir {self.state_file})")
        
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CNIL Scraper Final')
    parser.add_argument('--url', type=str, help='URL de départ')
    parser.add_argument('--depth', type=int, default=5, help='Profondeur max')
    parser.add_argument('--project-root', type=str, default='.', help='Racine projet')
    
    args = parser.parse_args()
    
    scraper = CNILScraperFinal(args.project_root)
    scraper.scrape_recursive(start_url=args.url, max_depth=args.depth)


if __name__ == "__main__":
    main()
