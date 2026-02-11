"""
Analyse des Ressources Liées aux Documents Pertinents - VERSION ROBUSTE
- PDFs avec extraction multi-méthodes (PyPDF2, pdfplumber, PyMuPDF, OCR)
- Nettoyage robuste des caractères problématiques
- Support LLaVA pour analyse images
- Docs (ODT/XLSX)
"""

import os
import json
from pathlib import Path
import sys
import logging
from typing import Dict, List, Set, Optional, Tuple
from tqdm import tqdm
import time
import mimetypes
from bs4 import BeautifulSoup
import hashlib
import base64

# Ajouter le chemin utils pour llm_provider
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'utils'))

from llm_provider import LLMFactory, RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ResourceAnalyzer:
    """Analyse les ressources liées aux documents pertinents"""
    
    # Prompt pour documents (PDF, ODT, XLSX)
    DOCUMENT_PROMPT = """Tu es un expert RGPD jouant le rôle de DPO.

Évalue si ce document attaché est utile pour un DPO.

Un document attaché EST pertinent s'il :
- Fournit des modèles/templates pratiques (registre, AIPD, etc.)
- Détaille une méthodologie opérationnelle
- Contient des exemples concrets de mise en conformité
- Est un formulaire officiel CNIL
- Est un guide technique détaillé

Un document attaché N'EST PAS pertinent s'il :
- Est purement décoratif/marketing
- Répète des infos déjà dans le HTML parent
- Est obsolète ou non applicable

Réponds UNIQUEMENT en JSON :
{
  "pertinent": true/false,
  "score": 0-10,
  "categorie": "essential" | "useful" | "duplicate" | "obsolete",
  "raison": "courte explication",
  "tags": ["tag1", "tag2"]
}"""

    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.data_path = self.project_root / 'data'
        self.html_dir = self.data_path / 'raw' / 'html'
        self.pdf_dir = self.data_path / 'raw' / 'pdf'
        self.docs_dir = self.data_path / 'raw' / 'docs'
        self.images_dir = self.data_path / 'raw' / 'images'
        self.metadata_dir = self.data_path / 'metadata'
        
        # Fichiers de résultats
        self.classification_file = self.data_path / 'hybrid_classification.json'
        self.resource_index_file = self.metadata_dir / 'resource_index_by_source.json'
        self.cache_file = self.data_path / 'resource_classification_cache.json'
        self.results_file = self.data_path / 'resource_analysis.json'
        
        # Cache
        self.cache = self._load_cache()
        
        # LLM
        try:
            config = RAGConfig()
            self.llm = config.llm_provider
            self.mode = config.mode
            logger.info(f"🤖 LLM initialisé en mode : {self.mode}")
        except Exception as e:
            logger.error(f"❌ Erreur init LLM : {e}")
            raise
        
        # Vérifier disponibilité LLaVA
        self.llava_available = self._check_llava()
        
        # Stats
        self.stats = {
            'total_resources': 0,
            'pdfs_analyzed': 0,
            'docs_analyzed': 0,
            'images_analyzed': 0,
            'pdfs_kept': 0,
            'docs_kept': 0,
            'images_kept': 0,
            'cached': 0,
            'llava_used': 0,
            'heuristic_used': 0,
        }
    
    def _check_llava(self) -> bool:
        """Vérifie si LLaVA est disponible"""
        try:
            import requests
            response = requests.get('http://localhost:11434/api/tags', timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                for model in models:
                    if 'llava' in model.get('name', '').lower():
                        logger.info(f"✅ LLaVA disponible : {model['name']}")
                        return True
            logger.info(f"⚠️  LLaVA non disponible - utilisation heuristique pour images")
            return False
        except:
            logger.info(f"⚠️  Ollama non accessible - utilisation heuristique pour images")
            return False
    
    def _load_cache(self) -> Dict:
        """Charge le cache"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"📦 Cache chargé : {len(cache)} ressources")
                return cache
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Sauvegarde le cache"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Erreur sauvegarde cache : {e}")
    
    def _get_file_hash(self, url: str) -> str:
        """Hash MD5 d'une URL"""
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def clean_extracted_text(self, text: str) -> str:
        """Nettoie le texte extrait (caractères problématiques)"""
        if not text:
            return ""
        
        # Supprimer caractères null et BOM
        text = text.replace('\x00', '')
        text = text.replace('\ufeff', '')
        text = text.replace('\ufffd', '')  # Caractère de remplacement
        
        # Normaliser guillemets et apostrophes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace('«', '"').replace('»', '"')
        
        # Supprimer caractères de contrôle (sauf \n, \r, \t)
        text = ''.join(c for c in text if c.isprintable() or c in '\n\r\t')
        
        # Normaliser espaces multiples
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_text_from_pdf(self, file_path: Path) -> str:
        """Extrait texte d'un PDF avec fallbacks multiples"""
        
        # Méthode 1 : PyPDF2 (rapide, standard)
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                if len(pdf.pages) > 0:
                    text = pdf.pages[0].extract_text()
                    text = self.clean_extracted_text(text)
                    if len(text) > 100:  # Au moins 100 chars
                        logger.debug(f"✅ PyPDF2: {file_path.name}")
                        return text[:2000]
        except Exception as e:
            logger.debug(f"⚠️  PyPDF2 échec: {e}")
        
        # Méthode 2 : pdfplumber (meilleur pour PDFs complexes)
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) > 0:
                    text = pdf.pages[0].extract_text()
                    text = self.clean_extracted_text(text)
                    if len(text) > 100:
                        logger.debug(f"✅ pdfplumber: {file_path.name}")
                        return text[:2000]
        except ImportError:
            logger.debug(f"⚠️  pdfplumber non installé (pip install pdfplumber)")
        except Exception as e:
            logger.debug(f"⚠️  pdfplumber échec: {e}")
        
        # Méthode 3 : PyMuPDF/fitz (très robuste)
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            if len(doc) > 0:
                text = doc[0].get_text()
                text = self.clean_extracted_text(text)
                if len(text) > 100:
                    logger.debug(f"✅ PyMuPDF: {file_path.name}")
                    doc.close()
                    return text[:2000]
                doc.close()
        except ImportError:
            logger.debug(f"⚠️  PyMuPDF non installé (pip install pymupdf)")
        except Exception as e:
            logger.debug(f"⚠️  PyMuPDF échec: {e}")
        
        # Méthode 4 : OCR avec Tesseract (dernier recours)
        try:
            import fitz
            from PIL import Image
            import pytesseract
            import io
            
            doc = fitz.open(file_path)
            if len(doc) > 0:
                # Convertir première page en image
                page = doc[0]
                pix = page.get_pixmap(dpi=150)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # OCR
                text = pytesseract.image_to_string(img, lang='fra')
                text = self.clean_extracted_text(text)
                
                if len(text) > 100:
                    logger.debug(f"✅ OCR Tesseract: {file_path.name}")
                    doc.close()
                    return text[:2000]
                doc.close()
        except ImportError:
            logger.debug(f"⚠️  Tesseract non installé")
        except Exception as e:
            logger.debug(f"⚠️  OCR échec: {e}")
        
        # Échec total
        logger.warning(f"⚠️  Impossible d'extraire texte de {file_path.name}")
        return "[PDF - Extraction impossible]"
    
    def extract_text_from_document(self, file_path: Path) -> str:
        """Extrait texte d'un PDF/ODT/XLSX (preview)"""
        ext = file_path.suffix.lower()
        
        try:
            if ext == '.pdf':
                return self.extract_text_from_pdf(file_path)
            
            elif ext in ['.odt', '.docx']:
                if ext == '.odt':
                    from odf import text as odf_text
                    from odf.opendocument import load
                    doc = load(str(file_path))
                    paragraphs = doc.getElementsByType(odf_text.P)
                    text = '\n'.join([str(p) for p in paragraphs[:20]])
                    text = self.clean_extracted_text(text)
                    return text[:2000]
                else:  # .docx
                    from docx import Document
                    doc = Document(file_path)
                    paragraphs = [p.text for p in doc.paragraphs[:20]]
                    text = '\n'.join(paragraphs)
                    text = self.clean_extracted_text(text)
                    return text[:2000]
            
            elif ext in ['.ods', '.xlsx', '.xls']:
                # ODS (LibreOffice Calc)
                if ext == '.ods':
                    from odf import table as odf_table
                    from odf import text as odf_text
                    from odf.opendocument import load
                    doc = load(str(file_path))
                    sheets = doc.spreadsheet.getElementsByType(odf_table.Table)
                    
                    if sheets:
                        sheet = sheets[0]  # Première feuille
                        rows = sheet.getElementsByType(odf_table.TableRow)
                        text_rows = []
                        
                        for i, row in enumerate(rows[:10]):  # 10 premières lignes
                            cells = row.getElementsByType(odf_table.TableCell)
                            cell_values = []
                            for cell in cells:
                                # Extraire texte de chaque cellule
                                paragraphs = cell.getElementsByType(odf_text.P)
                                cell_text = ' '.join([str(p) for p in paragraphs])
                                if cell_text.strip():
                                    cell_values.append(cell_text.strip())
                            
                            if cell_values:
                                text_rows.append(' | '.join(cell_values))
                        
                        text = '\n'.join(text_rows)
                        text = self.clean_extracted_text(text)
                        return text[:2000]
                
                # XLSX/XLS (Excel)
                else:
                    import openpyxl
                    wb = openpyxl.load_workbook(file_path, read_only=True)
                    sheet = wb.active
                    rows = []
                    for i, row in enumerate(sheet.iter_rows(values_only=True)):
                        if i >= 10:
                            break
                        row_text = ' | '.join([str(cell) for cell in row if cell])
                        rows.append(row_text)
                    text = '\n'.join(rows)
                    text = self.clean_extracted_text(text)
                    return text[:2000]
        
        except Exception as e:
            logger.debug(f"⚠️  Erreur extraction {file_path.name}: {e}")
            return f"[Fichier {ext[1:].upper()} - Extraction échouée]"
        
        return f"[Fichier {ext[1:].upper()}]"
    
    def analyze_document(self, file_path: Path, url: str, parent_url: str) -> Dict:
        """Analyse un document (PDF, ODT, XLSX)"""
        
        # Cache
        cache_key = f"{url}"
        if cache_key in self.cache:
            result = self.cache[cache_key].copy()
            result['cached'] = True
            self.stats['cached'] += 1
            return result
        
        # Extraire preview
        text_preview = self.extract_text_from_document(file_path)
        
        # Construire prompt
        user_prompt = f"""Document attaché : {file_path.name}
Page HTML parente : {parent_url}

Aperçu du contenu :
{text_preview}

Évalue la pertinence de ce document."""

        full_prompt = f"{self.DOCUMENT_PROMPT}\n\n{user_prompt}"
        
        try:
            response = self.llm.generate(full_prompt, temperature=0.1, max_tokens=300)
            
            # Parser JSON
            response_clean = response.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            # Extraction JSON si texte autour
            if not response_clean.startswith('{'):
                start_idx = response_clean.find('{')
                if start_idx != -1:
                    response_clean = response_clean[start_idx:]
            if not response_clean.endswith('}'):
                end_idx = response_clean.rfind('}')
                if end_idx != -1:
                    response_clean = response_clean[:end_idx+1]
            
            result = json.loads(response_clean)
            result['cached'] = False
            
            logger.info(f"✅ {result.get('categorie', 'N/A'):12s} ({result.get('score', 0):4.1f}/10) - {file_path.name[:40]}...")
            
            # Cache
            self.cache[cache_key] = result
            
            time.sleep(0.5 if self.mode == 'local' else 1.0)
            
            return result
        
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON invalide pour {file_path.name}: {e}")
            return {
                "pertinent": True,  # Garder par défaut
                "score": 5.0,
                "categorie": "useful",
                "raison": f"Erreur parsing: {str(e)}",
                "tags": [],
                "cached": False,
                "error": str(e)
            }
        except Exception as e:
            logger.error(f"❌ Erreur analyse {file_path.name}: {e}")
            return {
                "pertinent": True,
                "score": 5.0,
                "categorie": "useful",
                "raison": f"Erreur: {str(e)}",
                "tags": [],
                "cached": False,
                "error": str(e)
            }
    
    def analyze_image_with_llava(self, image_path: Path, parent_url: str) -> Optional[Dict]:
        """Analyse image avec LLaVA"""
        
        if not self.llava_available:
            return None
        
        try:
            import requests
            
            # Encoder image en base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # Prompt vision
            prompt = f"""Tu analyses une image provenant d'une page CNIL pour un DPO.

Page source : {parent_url}

Détermine si cette image est utile pour un DPO dans l'exercice de ses missions :

UTILE si :
- Schéma/diagramme technique (flux de données, architecture, processus)
- Infographie pédagogique sur concepts RGPD
- Capture d'écran montrant exemple concret d'interface/formulaire conforme
- Workflow ou méthodologie illustrée
- Exemple visuel de bonnes pratiques

NON UTILE si :
- Logo, icône, pictogramme décoratif
- Photo de personne ou bâtiment
- Bandeau publicitaire ou promotionnel
- Header, footer, élément de navigation
- Élément purement graphique sans valeur informative

Réponds UNIQUEMENT en JSON :
{{
  "pertinent": true/false,
  "score": 0-10,
  "categorie": "diagram" | "infographic" | "example" | "decorative",
  "raison": "courte explication de ce que tu vois et pourquoi c'est pertinent/non pertinent",
  "tags": ["tag1", "tag2"]
}}"""

            # Appel LLaVA
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llava:7b',
                    'prompt': prompt,
                    'images': [image_data],
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 300
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                logger.debug(f"⚠️  LLaVA HTTP error {response.status_code}")
                return None
            
            data = response.json()
            response_text = data.get('response', '')
            
            logger.debug(f"📥 LLaVA réponse brute (100 premiers chars) : {response_text[:100]}")
            
            # Parser JSON
            response_clean = response_text.strip()
            if response_clean.startswith('```json'):
                response_clean = response_clean[7:]
            elif response_clean.startswith('```'):
                response_clean = response_clean[3:]
            if response_clean.endswith('```'):
                response_clean = response_clean[:-3]
            response_clean = response_clean.strip()
            
            # Extraction JSON
            if not response_clean.startswith('{'):
                start_idx = response_clean.find('{')
                if start_idx != -1:
                    response_clean = response_clean[start_idx:]
                    logger.debug(f"✂️  JSON extrait à partir de position {start_idx}")
            if not response_clean.endswith('}'):
                end_idx = response_clean.rfind('}')
                if end_idx != -1:
                    response_clean = response_clean[:end_idx+1]
            
            logger.debug(f"🧹 JSON nettoyé (100 premiers chars) : {response_clean[:100]}")
            
            result = json.loads(response_clean)
            result['method'] = 'llava'
            result['cached'] = False
            self.stats['llava_used'] += 1
            
            logger.debug(f"✅ LLaVA parsing réussi: {result.get('categorie')} - {result.get('score')}/10")
            
            time.sleep(1.0)  # Rate limiting pour vision
            
            return result
        
        except json.JSONDecodeError as e:
            logger.debug(f"⚠️  LLaVA JSON parsing échec: {e}")
            logger.debug(f"   Réponse brute: {response_text[:200] if 'response_text' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logger.debug(f"⚠️  LLaVA échec: {e}")
            return None
    
    def analyze_image_heuristic(self, image_path: Path) -> Dict:
        """Analyse heuristique d'image (taille + nom fichier)"""
        
        filename = image_path.name.lower()
        
        # Patterns pertinents
        if any(kw in filename for kw in ['schema', 'diagram', 'diagramme', 'process', 'workflow', 'architecture', 'flux', 'infographic', 'infographie']):
            result = {
                "pertinent": True,
                "score": 7.0,
                "categorie": "diagram",
                "raison": "Nom de fichier indique diagramme/schéma",
                "tags": ["diagram", "heuristic"],
                "method": "heuristic"
            }
        
        # Patterns non pertinents
        elif any(kw in filename for kw in ['icon', 'logo', 'bandeau', 'header', 'footer', 'picto', 'avatar', 'portrait', 'thumb']):
            result = {
                "pertinent": False,
                "score": 1.0,
                "categorie": "decorative",
                "raison": "Nom de fichier indique élément décoratif",
                "tags": ["decorative", "heuristic"],
                "method": "heuristic"
            }
        
        # Vérifier taille image
        else:
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    width, height = img.size
                    
                    if width < 100 or height < 100:
                        result = {
                            "pertinent": False,
                            "score": 2.0,
                            "categorie": "decorative",
                            "raison": f"Petite image ({width}x{height}), probablement décorative",
                            "tags": ["small", "heuristic"],
                            "method": "heuristic"
                        }
                    else:
                        result = {
                            "pertinent": True,
                            "score": 5.0,
                            "categorie": "example",
                            "raison": f"Image de taille significative ({width}x{height})",
                            "tags": ["medium-large", "heuristic"],
                            "method": "heuristic"
                        }
            except:
                result = {
                    "pertinent": True,
                    "score": 5.0,
                    "categorie": "example",
                    "raison": "Image potentiellement utile",
                    "tags": ["uncertain", "heuristic"],
                    "method": "heuristic"
                }
        
        self.stats['heuristic_used'] += 1
        return result
    
    def analyze_image(self, image_path: Path, url: str, parent_url: str) -> Dict:
        """Analyse une image (LLaVA ou heuristique)"""
        
        # Cache
        cache_key = f"{url}"
        if cache_key in self.cache:
            result = self.cache[cache_key].copy()
            result['cached'] = True
            self.stats['cached'] += 1
            logger.debug(f"💾 Cache hit pour image: {image_path.name}")
            return result
        
        # Essayer LLaVA d'abord si disponible
        result = None
        if self.llava_available:
            logger.debug(f"🤖 Tentative analyse LLaVA: {image_path.name}")
            result = self.analyze_image_with_llava(image_path, parent_url)
        
        # Fallback heuristique si LLaVA indisponible ou échec
        if result is None:
            if self.llava_available:
                logger.debug(f"⚠️  LLaVA échec, fallback heuristique: {image_path.name}")
            else:
                logger.debug(f"ℹ️  LLaVA indisponible, analyse heuristique: {image_path.name}")
            result = self.analyze_image_heuristic(image_path)
        
        result['cached'] = False
        
        method = result.get('method', 'unknown')
        logger.info(f"✅ {result['categorie']:12s} ({result['score']:4.1f}/10) - {image_path.name[:35]}... [{method}]")
        
        # Cache
        self.cache[cache_key] = result
        
        return result
    
    def load_kept_documents(self) -> Set[str]:
        """Charge les URLs HTML gardées"""
        if not self.classification_file.exists():
            raise FileNotFoundError("Lancez d'abord hybrid_filter.py")
        
        with open(self.classification_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kept_urls = set()
        for hash_id, info in data.get('llm_classified', {}).items():
            if info.get('pertinent', False):
                kept_urls.add(info['url'])
        
        logger.info(f"📄 {len(kept_urls)} documents HTML pertinents")
        return kept_urls
    
    def load_resource_index(self) -> Dict:
        """Charge l'index des ressources"""
        if not self.resource_index_file.exists():
            logger.warning(f"⚠️  Index ressources introuvable")
            return {}
        
        with open(self.resource_index_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_relevant_images_from_html(self, html_file: Path, url: str) -> List[str]:
        """Extrait images référencées (hors déco)"""
        try:
            with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f, 'lxml')
            
            # Supprimer header/footer/nav
            for tag in soup(['header', 'footer', 'nav', 'aside', 'script', 'style']):
                tag.decompose()
            
            main_content = soup.find(['main', 'article', 'div']) or soup
            
            relevant_images = []
            
            for img in main_content.find_all('img', src=True):
                src = img['src']
                alt = img.get('alt', '').lower()
                
                # Filtrer icônes
                if any(skip in src.lower() for skip in ['icon', 'logo', 'picto', 'bullet', 'arrow']):
                    continue
                
                if any(skip in alt for skip in ['logo', 'icône', 'pictogramme']):
                    continue
                
                # Filtrer petites images
                width = img.get('width', '')
                height = img.get('height', '')
                if width and height:
                    try:
                        if int(width) < 50 or int(height) < 50:
                            continue
                    except:
                        pass
                
                # Images dans figure ou avec légende = pertinentes
                if img.find_parent('figure') or alt:
                    relevant_images.append(src)
            
            return relevant_images
        
        except Exception as e:
            logger.warning(f"⚠️  Erreur extraction images: {e}")
            return []
    
    def run(self, max_resources: Optional[int] = None):
        """Exécute l'analyse complète"""
        
        print("=" * 70)
        print("📎 ANALYSE DES RESSOURCES LIÉES - VERSION ROBUSTE")
        print("=" * 70)
        
        if max_resources:
            print(f"🧪 MODE TEST : {max_resources} ressources maximum")
        
        # Charger documents pertinents
        print("\n🔍 Chargement des documents HTML pertinents...")
        kept_urls = self.load_kept_documents()
        
        # Charger index ressources
        print("🔍 Chargement de l'index des ressources...")
        resource_index = self.load_resource_index()
        
        # Collecter ressources
        print("\n📋 Collecte des ressources liées...")
        
        resources_to_analyze = {
            'pdfs': {},
            'docs': {},
            'images': {}
        }
        
        for url in tqdm(kept_urls, desc="Collecte"):
            # Ressources depuis l'index
            if url in resource_index:
                for resource in resource_index[url].get('resources', []):
                    res_url = resource['url']
                    res_type = resource['file_type']
                    res_path = self.project_root / resource['file_path']
                    
                    if not res_path.exists():
                        continue
                    
                    if res_type == 'pdf' and os.path.getsize(res_path) >= 45153 :
                        resources_to_analyze['pdfs'][res_url] = {
                            'path': res_path,
                            'parent_url': url
                        }
                    elif res_type in ['odt', 'ods', 'docx', 'xlsx', 'xls']:
                        resources_to_analyze['docs'][res_url] = {
                            'path': res_path,
                            'parent_url': url
                        }
            
            # Images référencées
            html_hash = self._get_file_hash(url)
            html_file = self.html_dir / f"{html_hash}.html"
            
            if html_file.exists():
                relevant_images = self.get_relevant_images_from_html(html_file, url)
                for img_url in relevant_images:
                    img_hash = self._get_file_hash(img_url)
                    
                    for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                        img_path = self.images_dir / f"{img_hash}{ext}"
                        if img_path.exists() and os.path.getsize(img_path) >= 437742:
                            resources_to_analyze['images'][img_url] = {
                                'path': img_path,
                                'parent_url': url
                            }
                            break
        
        # Limiter si mode test
        if max_resources:
            total_collected = (
                len(resources_to_analyze['pdfs']) +
                len(resources_to_analyze['docs']) +
                len(resources_to_analyze['images'])
            )
            
            if total_collected > max_resources:
                # Répartir équitablement
                ratio_pdf = len(resources_to_analyze['pdfs']) / max(1, total_collected)
                ratio_doc = len(resources_to_analyze['docs']) / max(1, total_collected)
                ratio_img = len(resources_to_analyze['images']) / max(1, total_collected)
                
                max_pdf = int(max_resources * ratio_pdf)
                max_doc = int(max_resources * ratio_doc)
                max_img = max_resources - max_pdf - max_doc
                
                resources_to_analyze['pdfs'] = dict(list(resources_to_analyze['pdfs'].items())[:max_pdf])
                resources_to_analyze['docs'] = dict(list(resources_to_analyze['docs'].items())[:max_doc])
                resources_to_analyze['images'] = dict(list(resources_to_analyze['images'].items())[:max_img])
        
        self.stats['total_resources'] = (
            len(resources_to_analyze['pdfs']) +
            len(resources_to_analyze['docs']) +
            len(resources_to_analyze['images'])
        )
        
        print(f"\n📊 Ressources à analyser :")
        print(f"   PDFs       : {len(resources_to_analyze['pdfs'])}")
        print(f"   Documents  : {len(resources_to_analyze['docs'])}")
        print(f"   Images     : {len(resources_to_analyze['images'])}")
        print(f"   TOTAL      : {self.stats['total_resources']}")
        
        if self.stats['total_resources'] == 0:
            print("\n✅ Aucune ressource à analyser.")
            return
        
        # Estimation
        est_min = self.stats['total_resources'] * 2 / 60
        print(f"\n⏱️  Durée estimée : ~{est_min:.0f} minutes ({est_min/60:.1f}h)")
        
        if not max_resources:  # Seulement demander confirmation si pas en mode test
            input("\n   Appuyez sur Entrée pour continuer...")
        
        # Analyse
        results = {
            'pdfs': {},
            'docs': {},
            'images': {}
        }
        
        save_counter = 0
        
        # PDFs
        if resources_to_analyze['pdfs']:
            print(f"\n📄 Analyse des PDFs...")
            for url, info in tqdm(resources_to_analyze['pdfs'].items(), desc="PDFs"):
                analysis = self.analyze_document(info['path'], url, info['parent_url'])
                results['pdfs'][url] = {
                    'file_path': str(info['path'].relative_to(self.project_root)),
                    'parent_url': info['parent_url'],
                    'analysis': analysis
                }
                if analysis.get('pertinent'):
                    self.stats['pdfs_kept'] += 1
                self.stats['pdfs_analyzed'] += 1
                
                save_counter += 1
                if save_counter % 10 == 0:
                    self._save_cache()
        
        # Documents
        if resources_to_analyze['docs']:
            print(f"\n📝 Analyse des documents...")
            for url, info in tqdm(resources_to_analyze['docs'].items(), desc="Docs"):
                analysis = self.analyze_document(info['path'], url, info['parent_url'])
                results['docs'][url] = {
                    'file_path': str(info['path'].relative_to(self.project_root)),
                    'parent_url': info['parent_url'],
                    'analysis': analysis
                }
                if analysis.get('pertinent'):
                    self.stats['docs_kept'] += 1
                self.stats['docs_analyzed'] += 1
                
                save_counter += 1
                if save_counter % 10 == 0:
                    self._save_cache()
        
        # Images
        if resources_to_analyze['images']:
            print(f"\n🖼️  Analyse des images...")
            for url, info in tqdm(resources_to_analyze['images'].items(), desc="Images"):
                analysis = self.analyze_image(info['path'], url, info['parent_url'])
                results['images'][url] = {
                    'file_path': str(info['path'].relative_to(self.project_root)),
                    'parent_url': info['parent_url'],
                    'analysis': analysis
                }
                if analysis.get('pertinent'):
                    self.stats['images_kept'] += 1
                self.stats['images_analyzed'] += 1
        
        # Sauvegarde finale
        self._save_cache()
        self._save_results(results)
        
        # Résumé
        self._print_summary()
    
    def _save_results(self, results: Dict):
        """Sauvegarde les résultats"""
        output = {
            'pdfs': results['pdfs'],
            'docs': results['docs'],
            'images': results['images'],
            'stats': self.stats,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        }
        
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Résultats : {self.results_file}")
    
    def _print_summary(self):
        """Affiche le résumé"""
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ - ANALYSE DES RESSOURCES")
        print("=" * 70)
        
        print(f"\n📄 PDFs :")
        print(f"   Analysés : {self.stats['pdfs_analyzed']}")
        print(f"   Gardés   : {self.stats['pdfs_kept']} ({self.stats['pdfs_kept']/max(1,self.stats['pdfs_analyzed'])*100:.1f}%)")
        
        print(f"\n📝 Documents :")
        print(f"   Analysés : {self.stats['docs_analyzed']}")
        print(f"   Gardés   : {self.stats['docs_kept']} ({self.stats['docs_kept']/max(1,self.stats['docs_analyzed'])*100:.1f}%)")
        
        print(f"\n🖼️  Images :")
        print(f"   Analysées : {self.stats['images_analyzed']}")
        print(f"   Gardées   : {self.stats['images_kept']} ({self.stats['images_kept']/max(1,self.stats['images_analyzed'])*100:.1f}%)")
        print(f"   LLaVA     : {self.stats['llava_used']}")
        print(f"   Heuristique: {self.stats['heuristic_used']}")
        
        print(f"\n💾 Cache : {self.stats['cached']}/{self.stats['total_resources']} ({self.stats['cached']/max(1,self.stats['total_resources'])*100:.1f}%)")
        
        print("\n" + "=" * 70)
        print(f"💾 Résultats : {self.results_file}")
        print(f"💾 Cache : {self.cache_file}")
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyse ressources liées - Version robuste')
    parser.add_argument('--project-root', type=str, default='.', help='Racine du projet')
    parser.add_argument('--test', type=int, help='Tester sur N ressources (ex: --test 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbose')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger("__main__").setLevel(logging.DEBUG)
    
    analyzer = ResourceAnalyzer(args.project_root)
    analyzer.run(max_resources=args.test)


if __name__ == "__main__":
    main()