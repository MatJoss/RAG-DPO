"""
Phase 4B : Classification des images par OCR + Vision (LLaVA)
Pipeline en 3 étapes :
  1. Tesseract OCR → détection de texte
  2. Si texte détecté → LLaVA décrit le contenu visuel
  3. Classification : SCHEMA_DPO / INFOGRAPHIE / PHOTO_DECO
  
Les images PHOTO_DECO sont retirées de keep/ et du manifest.
"""

import json
import base64
import requests
import time
import sys
import io
import logging
import shutil
from pathlib import Path
from typing import Dict, Tuple, Optional
from PIL import Image

Image.MAX_IMAGE_PIXELS = 200_000_000  # Grosses infographies CNIL

try:
    import pytesseract
except ImportError:
    print("❌ pytesseract non installé : pip install pytesseract")
    sys.exit(1)

from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ImageClassifier:
    """Classification d'images par OCR + Vision"""
    
    # Seuils OCR
    MIN_WORDS_SCHEMA = 20    # ≥20 mots = probablement schéma/infographie
    MIN_WORDS_VISION = 5     # ≥5 mots = envoyer à LLaVA pour trancher
    
    # Config LLaVA
    OLLAMA_URL = 'http://localhost:11434'
    VISION_MODEL = 'llava:7b'
    MAX_IMAGE_DIM = 1024     # Redimensionner avant envoi LLaVA (économie VRAM)
    
    LLAVA_PROMPT = """Tu analyses une image provenant du site de la CNIL (Commission Nationale Informatique et Libertés).

Décris BRIÈVEMENT le contenu visuel, puis classe l'image.

CATÉGORIES :
- SCHEMA_DPO : schéma, diagramme, organigramme, flowchart technique utile pour un DPO
- INFOGRAPHIE : infographie pédagogique avec données chiffrées ou étapes de conformité RGPD
- PHOTO_DECO : photo d'événement, portrait, visuel décoratif, bandeau communication, BD simpliste

Réponds UNIQUEMENT en JSON strict :
{
  "categorie": "SCHEMA_DPO",
  "description": "Schéma montrant les 6 étapes d'une AIPD",
  "pertinent_dpo": true,
  "raison": "Diagramme technique utile pour comprendre le processus AIPD"
}

Règles :
- pertinent_dpo = true UNIQUEMENT si contenu technique/juridique exploitable par un DPO
- Une photo avec du texte superposé reste PHOTO_DECO si c'est juste de la communication
- Un schéma même simple (flèches, boîtes) est SCHEMA_DPO s'il illustre un processus DPO"""

    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.data_path = self.project_root / 'data'
        self.keep_dir = self.data_path / 'keep' / 'cnil'
        self.images_dir = self.keep_dir / 'images'
        self.metadata_dir = self.keep_dir / 'metadata'
        self.cnil_path = self.data_path / 'raw' / 'cnil'
        self.manifest_file = self.cnil_path / 'keep_manifest.json'
        self.output_file = self.cnil_path / 'image_classification.json'
        self.cache_file = self.cnil_path / 'image_classification_cache.json'
        
        # Stats
        self.stats = {
            'total': 0,
            'schema_dpo': 0,
            'infographie': 0,
            'photo_deco': 0,
            'ocr_only': 0,       # Éliminées par OCR seul (< 5 mots)
            'vision_analyzed': 0, # Passées par LLaVA
            'errors': 0,
            'cached': 0,
            'removed_from_keep': 0,
        }
    
    def _load_cache(self) -> Dict:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                logger.info(f"📦 Cache chargé : {len(cache)} images")
                return cache
            except:
                return {}
        return {}
    
    def _save_cache(self, cache: Dict):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    
    def ocr_image(self, img_path: Path) -> Tuple[int, str]:
        """OCR Tesseract sur une image. Retourne (nb_mots, texte_extrait)."""
        try:
            img = Image.open(img_path)
            w, h = img.size
            
            # Redimensionner si trop grande pour Tesseract
            if w * h > 30_000_000:
                ratio = (30_000_000 / (w * h)) ** 0.5
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            
            text = pytesseract.image_to_string(img, lang='fra', timeout=30)
            text_clean = ' '.join(text.split())
            word_count = len(text_clean.split()) if text_clean else 0
            
            return word_count, text_clean
        except Exception as e:
            logger.debug(f"OCR erreur {img_path.name}: {e}")
            return 0, ""
    
    def describe_with_llava(self, img_path: Path) -> Optional[Dict]:
        """Envoie l'image à LLaVA pour description + classification."""
        try:
            # Charger et redimensionner pour VRAM
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            if max(w, h) > self.MAX_IMAGE_DIM:
                ratio = self.MAX_IMAGE_DIM / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            
            # Convertir en base64
            import io as _io
            buf = _io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            image_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Appel Ollama
            response = requests.post(
                f'{self.OLLAMA_URL}/api/generate',
                json={
                    'model': self.VISION_MODEL,
                    'prompt': self.LLAVA_PROMPT,
                    'images': [image_b64],
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'num_predict': 300,
                    }
                },
                timeout=60
            )
            
            if response.status_code != 200:
                logger.warning(f"LLaVA HTTP {response.status_code} pour {img_path.name}")
                return None
            
            raw_text = response.json().get('response', '')
            
            # Parser le JSON (tolérant)
            # Chercher le premier { et le dernier }
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            if start >= 0 and end > start:
                json_str = raw_text[start:end+1]
                result = json.loads(json_str)
                return result
            
            logger.warning(f"LLaVA pas de JSON pour {img_path.name}: {raw_text[:200]}")
            return None
            
        except json.JSONDecodeError as e:
            logger.warning(f"LLaVA JSON invalide pour {img_path.name}: {e}")
            return None
        except Exception as e:
            logger.warning(f"LLaVA erreur pour {img_path.name}: {e}")
            return None
    
    def classify_image(self, img_path: Path, cache: Dict) -> Dict:
        """Pipeline complet de classification d'une image."""
        cache_key = img_path.name
        
        if cache_key in cache:
            self.stats['cached'] += 1
            # Comptabiliser la catégorie même depuis le cache
            cached_cat = cache[cache_key].get('categorie', '')
            if cached_cat == 'SCHEMA_DPO':
                self.stats['schema_dpo'] += 1
            elif cached_cat == 'INFOGRAPHIE':
                self.stats['infographie'] += 1
            elif cached_cat == 'PHOTO_DECO':
                self.stats['photo_deco'] += 1
            cached_method = cache[cache_key].get('method', '')
            if cached_method == 'ocr_only':
                self.stats['ocr_only'] += 1
            elif cached_method in ('llava', 'ocr_fallback'):
                self.stats['vision_analyzed'] += 1
            return cache[cache_key]
        
        size_kb = img_path.stat().st_size / 1024
        
        # Étape 1 : OCR
        word_count, ocr_text = self.ocr_image(img_path)
        
        result = {
            'file': img_path.name,
            'size_kb': round(size_kb, 1),
            'ocr_words': word_count,
            'ocr_preview': ocr_text[:200] if ocr_text else '',
        }
        
        # Étape 2 : Décision basée sur OCR
        if word_count < self.MIN_WORDS_VISION:
            # Pas assez de texte → photo/déco, pas besoin de LLaVA
            result['categorie'] = 'PHOTO_DECO'
            result['pertinent_dpo'] = False
            result['method'] = 'ocr_only'
            result['description'] = 'Image sans texte significatif (photo/visuel décoratif)'
            result['raison'] = f'OCR: {word_count} mots < seuil {self.MIN_WORDS_VISION}'
            self.stats['ocr_only'] += 1
        else:
            # Texte détecté → LLaVA pour analyse visuelle
            self.stats['vision_analyzed'] += 1
            llava_result = self.describe_with_llava(img_path)
            
            if llava_result:
                categorie = llava_result.get('categorie', 'PHOTO_DECO').upper()
                # Normaliser
                if categorie not in ('SCHEMA_DPO', 'INFOGRAPHIE', 'PHOTO_DECO'):
                    categorie = 'PHOTO_DECO'  # Fallback sécuritaire
                
                result['categorie'] = categorie
                result['pertinent_dpo'] = llava_result.get('pertinent_dpo', False)
                result['method'] = 'llava'
                result['description'] = llava_result.get('description', '')
                result['raison'] = llava_result.get('raison', '')
            else:
                # LLaVA a échoué → heuristique OCR
                if word_count >= self.MIN_WORDS_SCHEMA:
                    result['categorie'] = 'SCHEMA_DPO'
                    result['pertinent_dpo'] = True
                    result['method'] = 'ocr_fallback'
                    result['description'] = f'Image avec {word_count} mots (LLaVA indisponible)'
                    result['raison'] = 'Fallback OCR: beaucoup de texte détecté'
                else:
                    result['categorie'] = 'PHOTO_DECO'
                    result['pertinent_dpo'] = False
                    result['method'] = 'ocr_fallback'
                    result['description'] = f'Image avec peu de texte ({word_count} mots)'
                    result['raison'] = 'Fallback OCR: peu de texte, LLaVA indisponible'
                self.stats['errors'] += 1
        
        # Compteurs
        cat = result['categorie']
        if cat == 'SCHEMA_DPO':
            self.stats['schema_dpo'] += 1
        elif cat == 'INFOGRAPHIE':
            self.stats['infographie'] += 1
        else:
            self.stats['photo_deco'] += 1
        
        # Sauvegarder dans le cache
        cache[cache_key] = result
        
        return result
    
    def check_llava_available(self) -> bool:
        """Vérifie que LLaVA est disponible dans Ollama."""
        try:
            response = requests.get(f'{self.OLLAMA_URL}/api/tags', timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                for m in models:
                    if 'llava' in m.lower():
                        logger.info(f"✅ LLaVA disponible : {m}")
                        return True
            logger.warning("⚠️  LLaVA non trouvé dans Ollama")
            return False
        except Exception as e:
            logger.warning(f"⚠️  Ollama non accessible : {e}")
            return False
    
    def run(self, fresh: bool = False, max_images: int = None):
        """Exécute la classification complète des images."""
        
        print("=" * 70)
        print("🖼️  PHASE 4B : CLASSIFICATION DES IMAGES (OCR + LLaVA)")
        print("=" * 70)
        
        # Vérifier images
        if not self.images_dir.exists():
            print("❌ Pas d'images dans keep/")
            return
        
        images = list(self.images_dir.glob('*'))
        images = [f for f in images if f.is_file()]
        self.stats['total'] = len(images)
        
        if not images:
            print("ℹ️  Aucune image à classifier")
            return
        
        print(f"\n📊 Images à classifier : {len(images)}")
        
        # Vérifier LLaVA
        llava_ok = self.check_llava_available()
        if not llava_ok:
            print("⚠️  LLaVA non disponible — classification OCR uniquement")
            print("   Pour installer : ollama pull llava:7b")
        
        # Cache
        if fresh:
            for f in [self.cache_file, self.output_file]:
                if f.exists():
                    f.unlink()
                    logger.info(f"🗑️  Supprimé (fresh) : {f.name}")
            cache = {}
        else:
            cache = self._load_cache()
        
        is_test = max_images is not None
        if is_test:
            images = images[:max_images]
            print(f"🧪 MODE TEST : {max_images} images (dry-run, pas de modification keep/manifest)")
        
        # Estimation
        uncached = sum(1 for img in images if img.name not in cache)
        # ~1s OCR + ~5s LLaVA par image avec texte (~30% ont du texte)
        est_s = uncached * 1.5  # Moyenne pondérée
        print(f"   À traiter : {uncached} (cache: {len(images) - uncached})")
        print(f"   Durée estimée : ~{est_s/60:.1f} min")
        
        # Classification
        results = {}
        save_counter = 0
        interrupted = False
        
        print(f"\n🔄 Classification en cours...")
        
        try:
            for img_path in tqdm(images, desc="Images"):
                result = self.classify_image(img_path, cache)
                results[img_path.name] = result
                
                save_counter += 1
                if save_counter % 20 == 0:
                    self._save_cache(cache)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrompu — sauvegarde du cache...")
            interrupted = True
        
        # Sauvegarde cache
        self._save_cache(cache)
        
        if interrupted:
            print(f"💾 Cache sauvegardé ({len(cache)} images)")
            print("   Relancez pour reprendre")
            return
        
        # Sauvegarder résultats
        output = {
            'classifications': results,
            'stats': self.stats,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'llava_available': llava_ok,
        }
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        # Nettoyer keep/ : retirer les PHOTO_DECO (sauf en mode test)
        removed = 0
        if not is_test:
            for img_name, classification in results.items():
                if not classification.get('pertinent_dpo', False):
                    img_file = self.images_dir / img_name
                    if img_file.exists():
                        img_file.unlink()
                        removed += 1
                        # Aussi la métadonnée
                        meta_file = self.metadata_dir / f"{Path(img_name).stem}.json"
                        if meta_file.exists():
                            meta_file.unlink()
            
            self.stats['removed_from_keep'] = removed
            
            # Mettre à jour le manifest
            self._update_manifest(results)
        else:
            self.stats['removed_from_keep'] = 0
            logger.info("🧪 Mode test : keep/ et manifest non modifiés")
        
        # Résumé
        self._print_summary()
    
    def _update_manifest(self, results: Dict):
        """Met à jour keep_manifest.json : retire les images non pertinentes."""
        if not self.manifest_file.exists():
            return
        
        with open(self.manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        
        original_count = len(manifest.get('images', []))
        
        # Filtrer les images
        kept_images = []
        for img_item in manifest.get('images', []):
            img_file = img_item.get('file', '')  # "images/xxx.png"
            img_name = Path(img_file).name
            
            classification = results.get(img_name, {})
            if classification.get('pertinent_dpo', False):
                # Enrichir avec la classification
                img_item['classification'] = {
                    'categorie': classification.get('categorie'),
                    'description': classification.get('description'),
                    'ocr_words': classification.get('ocr_words', 0),
                }
                kept_images.append(img_item)
        
        manifest['images'] = kept_images
        
        # Nettoyer aussi les relationships
        for url, resources in manifest.get('relationships', {}).items():
            manifest['relationships'][url] = [
                r for r in resources
                if r.get('type') != 'image' or 
                Path(r.get('file', '')).name in {Path(img['file']).name for img in kept_images}
            ]
        
        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📝 Manifest mis à jour : {original_count} → {len(kept_images)} images")
    
    def _print_summary(self):
        """Affiche le résumé."""
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ - CLASSIFICATION IMAGES")
        print("=" * 70)
        
        print(f"\n🖼️  Images analysées : {self.stats['total']}")
        print(f"   Cache utilisé : {self.stats['cached']}")
        
        print(f"\n📂 Classification :")
        print(f"   SCHEMA_DPO   : {self.stats['schema_dpo']:4d}  (schémas, diagrammes)")
        print(f"   INFOGRAPHIE  : {self.stats['infographie']:4d}  (infographies pédagogiques)")
        print(f"   PHOTO_DECO   : {self.stats['photo_deco']:4d}  (photos, visuels décoratifs)")
        
        pertinent = self.stats['schema_dpo'] + self.stats['infographie']
        print(f"\n✅ Pertinentes DPO : {pertinent}")
        print(f"❌ Éliminées       : {self.stats['photo_deco']}")
        
        print(f"\n🔬 Méthode :")
        print(f"   OCR seul (<{self.MIN_WORDS_VISION} mots) : {self.stats['ocr_only']}")
        print(f"   LLaVA (vision)         : {self.stats['vision_analyzed']}")
        if self.stats['errors']:
            print(f"   ⚠️  Erreurs LLaVA       : {self.stats['errors']}")
        
        print(f"\n🗑️  Retirées de keep/ : {self.stats['removed_from_keep']} fichiers")
        
        print(f"\n💾 Résultats : {self.output_file}")
        print(f"💾 Cache     : {self.cache_file}")
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 4B - Classification images OCR + LLaVA')
    parser.add_argument('--project-root', type=str, default='.', help='Racine du projet')
    parser.add_argument('--test', type=int, help='Tester sur N images')
    parser.add_argument('--fresh', action='store_true', help='Ignorer cache, tout reclassifier')
    
    args = parser.parse_args()
    
    classifier = ImageClassifier(args.project_root)
    classifier.run(fresh=args.fresh, max_images=args.test)


if __name__ == "__main__":
    main()
