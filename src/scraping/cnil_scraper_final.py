"""
CNIL Scraper - Version FINALE Production
Corrige TOUS les bugs, gestion robuste, métadonnées complètes
"""

import os
import re
import time
import json
import hashlib
import mimetypes
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Set, Dict, List, Optional, Tuple
from datetime import datetime
from email.utils import parsedate_to_datetime
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
    
    def _download_file(self, url: str, retry: int = 0, 
                        if_modified_since: str = None) -> Optional[Tuple[bytes, str, Dict[str, str]]]:
        """Télécharge avec retry et retourne (content, content_type, http_headers)
        
        Args:
            url: URL à télécharger
            retry: Numéro du retry courant
            if_modified_since: Valeur du header If-Modified-Since pour requête conditionnelle.
                               Si le serveur renvoie 304, retourne None (page non modifiée).
        
        Returns:
            Tuple (content, content_type, headers_dict) ou None si échec/304.
            headers_dict contient les headers HTTP pertinents pour les metadata.
        """
        try:
            time.sleep(self.delay)
            
            # Headers conditionnels pour le mode update
            extra_headers = {}
            if if_modified_since:
                extra_headers['If-Modified-Since'] = if_modified_since
            
            response = self.session.get(url, timeout=30, allow_redirects=True,
                                        headers=extra_headers)
            
            # 304 Not Modified → page inchangée
            if response.status_code == 304:
                self.logger.debug(f"⏭️  304 Not Modified : {url}")
                return None
            
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '')
            
            # Capturer les headers pertinents pour les metadata
            http_headers = {
                'last_modified': response.headers.get('Last-Modified', ''),
                'etag': response.headers.get('ETag', ''),
                'content_length': response.headers.get('Content-Length', ''),
            }
            
            return response.content, content_type, http_headers
        
        except Exception as e:
            if retry < self.max_retries:
                self.logger.warning(f"⚠️  Retry {retry+1}/{self.max_retries} : {url}")
                time.sleep(self.delay * (retry + 1))
                return self._download_file(url, retry + 1, if_modified_since)
            else:
                self.logger.error(f"❌ Échec : {url} - {str(e)}")
                self.session_stats['errors'] += 1
                self.failed_urls.append({
                    'url': url,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                })
                return None
    
    def _compute_content_hash(self, content: bytes) -> str:
        """SHA256 du contenu pour détection de changements"""
        return hashlib.sha256(content).hexdigest()
    
    def _parse_http_date(self, http_date: str) -> Optional[str]:
        """Convertit une date HTTP (RFC 2822) en ISO 8601
        
        Ex: 'Sun, 01 Mar 2026 10:28:39 GMT' → '2026-03-01T10:28:39+00:00'
        """
        if not http_date:
            return None
        try:
            dt = parsedate_to_datetime(http_date)
            return dt.isoformat()
        except Exception:
            return None
    
    def _extract_page_dates(self, soup: BeautifulSoup) -> Dict[str, Optional[str]]:
        """Extrait les dates de publication/modification depuis le HTML CNIL
        
        Le site CNIL expose les dates de deux façons :
        - <span class="element-date">Publié le  26/01/2026</span>  (pages listing)
        - <p class="date">05 septembre 2024</p>                    (pages article)
        
        Returns:
            Dict avec 'published_at' (première date trouvée = date de publication principale)
            et 'page_dates' (liste de toutes les dates trouvées sur la page)
        """
        dates_found = []
        
        # Pattern 1 : <span class="element-date">Publié le  DD/MM/YYYY</span>
        for span in soup.find_all('span', class_='element-date'):
            text = span.get_text(strip=True)
            match = re.search(r'(\d{2})/(\d{2})/(\d{4})', text)
            if match:
                day, month, year = match.groups()
                try:
                    iso = f"{year}-{month}-{day}"
                    dates_found.append(iso)
                except ValueError:
                    pass
        
        # Pattern 2 : <p class="date">DD mois YYYY</p>
        mois_fr = {
            'janvier': '01', 'février': '02', 'mars': '03', 'avril': '04',
            'mai': '05', 'juin': '06', 'juillet': '07', 'août': '08',
            'septembre': '09', 'octobre': '10', 'novembre': '11', 'décembre': '12'
        }
        for p in soup.find_all('p', class_='date'):
            text = p.get_text(strip=True).lower()
            match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', text)
            if match:
                day, mois_str, year = match.groups()
                month_num = mois_fr.get(mois_str)
                if month_num:
                    iso = f"{year}-{month_num}-{day.zfill(2)}"
                    dates_found.append(iso)
        
        # Pattern 3 : <time datetime="..."> (si jamais CNIL l'ajoute)
        for time_tag in soup.find_all('time'):
            dt_val = time_tag.get('datetime', '')
            if dt_val:
                dates_found.append(dt_val[:10])  # YYYY-MM-DD
        
        return {
            'published_at': dates_found[0] if dates_found else None,
            'page_dates': dates_found if dates_found else None,
        }
    
    def _save_file_with_metadata(self, url: str, content: bytes, file_type: str, 
                                 source_url: str = None,
                                 http_headers: Dict[str, str] = None,
                                 page_dates: Dict[str, Optional[str]] = None) -> str:
        """Sauvegarde fichier + métadonnées complètes
        
        Schéma metadata unifié (HTML et PDF/docs) :
        - Identité : url, url_hash, file_type, extension, file_path
        - Contenu : file_size_bytes, content_hash (SHA256)
        - Provenance : source_url
        - Dates serveur : http_last_modified (ISO 8601 depuis Last-Modified HTTP)
        - Dates page : published_at (date de publication CNIL), page_dates (toutes)
        - Scraping : scraped_at, scraper_version
        """
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
        
        # Préparer headers HTTP
        if http_headers is None:
            http_headers = {}
        
        # Préparer dates page
        if page_dates is None:
            page_dates = {}
        
        # Métadonnées COMPLÈTES — schéma unifié v2
        metadata = {
            # Identité
            'url': url,
            'url_hash': url_hash,
            'file_type': file_type,
            'extension': ext,
            'file_path': str(file_path.relative_to(self.project_root)),
            'file_size_bytes': len(content),
            
            # Détection de changement
            'content_hash': self._compute_content_hash(content),
            
            # Provenance
            'source_url': source_url,
            
            # Dates serveur HTTP
            'http_last_modified': self._parse_http_date(http_headers.get('last_modified', '')),
            
            # Dates extraites de la page HTML
            'published_at': page_dates.get('published_at'),
            'page_dates': page_dates.get('page_dates'),
            
            # Scraping
            'scraped_at': datetime.now().isoformat(),
            'scraper_version': 'v2',
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
    
    def scrape_url(self, url: str, source_url: str = None,
                    if_modified_since: str = None) -> Optional[BeautifulSoup]:
        """Scrape une URL avec gestion complète
        
        Args:
            url: URL à scraper
            source_url: URL de la page parent (pour PDFs/docs embarqués)
            if_modified_since: Pour le mode update — skip si 304
        
        Returns:
            BeautifulSoup si HTML, None sinon ou si échec/304
        """
        if url in self.visited_urls:
            return None
        
        self.visited_urls.add(url)
        
        result = self._download_file(url, if_modified_since=if_modified_since)
        if result is None:
            return None
        
        content, content_type, http_headers = result
        file_type = self._detect_file_type(url, content_type)
        
        # Extraire dates de la page HTML AVANT de sauvegarder
        page_dates = {}
        soup = None
        if file_type == 'html':
            try:
                soup = BeautifulSoup(content, 'lxml')
                page_dates = self._extract_page_dates(soup)
            except Exception as e:
                self.logger.warning(f"⚠️  Erreur extraction dates {url}: {e}")
        
        # Sauvegarder avec metadata enrichies
        local_path = self._save_file_with_metadata(
            url, content, file_type, source_url,
            http_headers=http_headers,
            page_dates=page_dates,
        )
        self.downloaded_resources[url] = local_path
        
        # Si HTML, extraire ressources
        if file_type == 'html' and soup:
            try:
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


    def update_existing(self):
        """Mode incrémental : re-scrape uniquement les pages modifiées
        
        Pour chaque document déjà scrapé :
        1. Envoie une requête conditionnelle (If-Modified-Since)
        2. Si 304 → skip (page inchangée)
        3. Si 200 → compare content_hash
           - Hash identique → skip (contenu identique malgré 200)
           - Hash différent → re-sauvegarde avec nouvelles metadata
        
        Retourne un rapport des changements détectés.
        """
        self.logger.info("🔄 Mode UPDATE : vérification des modifications...")
        
        # Charger toutes les metadata existantes
        metadata_dir = self.output_dirs['metadata']
        metadata_files = list(metadata_dir.glob('*.json'))
        
        # Exclure le fichier d'état
        metadata_files = [f for f in metadata_files if f.name != 'scraping_state_final.json']
        
        stats = {
            'checked': 0,
            'unchanged_304': 0,
            'unchanged_hash': 0,
            'modified': 0,
            'errors': 0,
            'new_fields_added': 0,
            'modified_urls': [],
        }
        
        self.logger.info(f"📄 {len(metadata_files)} documents à vérifier")
        
        for meta_file in tqdm(metadata_files, desc="Vérification modifications"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                url = meta.get('url', '')
                if not url:
                    continue
                
                stats['checked'] += 1
                old_hash = meta.get('content_hash', '')
                
                # Requête conditionnelle avec If-Modified-Since
                if_modified = meta.get('http_last_modified')
                # Convertir ISO → HTTP date pour le header
                http_if_modified = None
                if if_modified:
                    try:
                        dt = datetime.fromisoformat(if_modified)
                        http_if_modified = dt.strftime('%a, %d %b %Y %H:%M:%S GMT')
                    except Exception:
                        pass
                
                result = self._download_file(url, if_modified_since=http_if_modified)
                
                if result is None:
                    # 304 Not Modified ou erreur
                    if http_if_modified:  # On avait un If-Modified-Since → c'est un 304
                        stats['unchanged_304'] += 1
                    else:
                        stats['errors'] += 1
                    continue
                
                content, content_type, http_headers = result
                new_hash = self._compute_content_hash(content)
                file_type = self._detect_file_type(url, content_type)
                
                # Comparer le hash du contenu
                if old_hash and new_hash == old_hash:
                    # Contenu identique — mettre à jour les metadata manquantes si besoin
                    updated = False
                    if not meta.get('content_hash'):
                        meta['content_hash'] = new_hash
                        updated = True
                    if not meta.get('http_last_modified') and http_headers.get('last_modified'):
                        meta['http_last_modified'] = self._parse_http_date(http_headers['last_modified'])
                        updated = True
                    if not meta.get('published_at') and file_type == 'html':
                        try:
                            soup = BeautifulSoup(content, 'lxml')
                            page_dates = self._extract_page_dates(soup)
                            if page_dates.get('published_at'):
                                meta['published_at'] = page_dates['published_at']
                                meta['page_dates'] = page_dates.get('page_dates')
                                updated = True
                        except Exception:
                            pass
                    if not meta.get('scraper_version'):
                        meta['scraper_version'] = 'v2'
                        updated = True
                    
                    if updated:
                        stats['new_fields_added'] += 1
                        with open(meta_file, 'w', encoding='utf-8') as f:
                            json.dump(meta, f, indent=2, ensure_ascii=False)
                    
                    stats['unchanged_hash'] += 1
                    continue
                
                # Contenu modifié → re-sauvegarder
                self.logger.info(f"📝 Modifié : {url}")
                stats['modified'] += 1
                stats['modified_urls'].append(url)
                
                # Extraire dates de la page
                page_dates = {}
                if file_type == 'html':
                    try:
                        soup = BeautifulSoup(content, 'lxml')
                        page_dates = self._extract_page_dates(soup)
                    except Exception:
                        pass
                
                # Ne pas ajouter à visited_urls (on ne veut pas perturber l'état de crawl)
                self._save_file_with_metadata(
                    url, content, file_type, meta.get('source_url'),
                    http_headers=http_headers,
                    page_dates=page_dates,
                )
                
            except Exception as e:
                stats['errors'] += 1
                self.logger.warning(f"⚠️  Erreur update {meta_file.name}: {e}")
        
        # Rapport
        self._save_state()
        
        print("\n" + "=" * 70)
        print("📊 RAPPORT DE MISE À JOUR")
        print("=" * 70)
        print(f"   Documents vérifiés  : {stats['checked']}")
        print(f"   Inchangés (304)     : {stats['unchanged_304']}")
        print(f"   Inchangés (hash)    : {stats['unchanged_hash']}")
        print(f"   ✏️  Modifiés          : {stats['modified']}")
        print(f"   🏷️  Metadata enrichies: {stats['new_fields_added']}")
        print(f"   ❌ Erreurs           : {stats['errors']}")
        
        if stats['modified_urls']:
            print(f"\n📝 URLs modifiées :")
            for u in stats['modified_urls']:
                print(f"   - {u}")
        
        print("=" * 70)
        
        return stats
    
    def backfill_metadata(self):
        """Enrichit les metadata existantes SANS re-télécharger
        
        Pour chaque fichier déjà présent localement :
        - Ajoute content_hash (SHA256) si manquant
        - Extrait les dates de la page HTML si manquantes
        - Unifie le schéma (scraped_at, scraper_version, etc.)
        
        Ne fait AUCUNE requête HTTP. Utile pour migrer les anciennes metadata.
        """
        self.logger.info("🏷️  Mode BACKFILL : enrichissement metadata local...")
        
        metadata_dir = self.output_dirs['metadata']
        metadata_files = [f for f in metadata_dir.glob('*.json') 
                         if f.name != 'scraping_state_final.json']
        
        stats = {'processed': 0, 'updated': 0, 'errors': 0}
        
        for meta_file in tqdm(metadata_files, desc="Backfill metadata"):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                
                file_path = self.project_root / meta.get('file_path', '')
                if not file_path.exists():
                    continue
                
                stats['processed'] += 1
                updated = False
                
                # 1. Ajouter content_hash si manquant
                if not meta.get('content_hash'):
                    content = file_path.read_bytes()
                    meta['content_hash'] = self._compute_content_hash(content)
                    updated = True
                
                # 2. Extraire dates de la page HTML si manquantes
                if not meta.get('published_at') and meta.get('file_type') == 'html':
                    try:
                        content = file_path.read_bytes()
                        soup = BeautifulSoup(content, 'lxml')
                        page_dates = self._extract_page_dates(soup)
                        if page_dates.get('published_at'):
                            meta['published_at'] = page_dates['published_at']
                            meta['page_dates'] = page_dates.get('page_dates')
                            updated = True
                    except Exception:
                        pass
                
                # 3. Harmoniser scraped_at (certains ont downloaded_at)
                if not meta.get('scraped_at') and meta.get('downloaded_at'):
                    meta['scraped_at'] = meta['downloaded_at']
                    updated = True
                
                # 4. Ajouter url_hash si manquant
                if not meta.get('url_hash') and meta.get('url'):
                    meta['url_hash'] = self._get_url_hash(meta['url'])
                    updated = True
                
                # 5. Marquer version schéma
                if meta.get('scraper_version') != 'v2':
                    meta['scraper_version'] = 'v2'
                    updated = True
                
                if updated:
                    stats['updated'] += 1
                    with open(meta_file, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, indent=2, ensure_ascii=False)
                
            except Exception as e:
                stats['errors'] += 1
                self.logger.warning(f"⚠️  Erreur backfill {meta_file.name}: {e}")
        
        print(f"\n🏷️  Backfill terminé : {stats['updated']}/{stats['processed']} metadata enrichies, {stats['errors']} erreurs")
        return stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='CNIL Scraper Final')
    parser.add_argument('--url', type=str, help='URL de départ')
    parser.add_argument('--depth', type=int, default=5, help='Profondeur max')
    parser.add_argument('--project-root', type=str, default='.', help='Racine projet')
    parser.add_argument('--update', action='store_true',
                        help='Mode incrémental : ne re-scrape que les pages modifiées')
    parser.add_argument('--backfill', action='store_true',
                        help='Enrichir les metadata existantes sans re-télécharger')
    
    args = parser.parse_args()
    
    scraper = CNILScraperFinal(args.project_root)
    
    if args.backfill:
        scraper.backfill_metadata()
    elif args.update:
        scraper.update_existing()
    else:
        scraper.scrape_recursive(start_url=args.url, max_depth=args.depth)


if __name__ == "__main__":
    main()
