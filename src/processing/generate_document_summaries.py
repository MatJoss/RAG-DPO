"""
Script de génération de résumés pour les documents CNIL.
Génère des fiches structurées (100-150 tokens) pour chaque document indexé dans ChromaDB.
"""
import json
import logging
import re
from pathlib import Path
import sys
from typing import Dict, List
from tqdm import tqdm
import chromadb
from chromadb import PersistentClient

# Ajouter le chemin du projet
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.llm_provider import OllamaProvider

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Désactive logs HTTP verbeux
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Paths
CHROMADB_PATH = project_root / "data" / "vectordb" / "chromadb"
METADATA_DIR = project_root / "data" / "keep" / "cnil" / "metadata"
SUMMARIES_OUTPUT = project_root / "data" / "keep" / "cnil" / "document_summaries.json"
KEEP_PATH = project_root / "data" / "keep" / "cnil"

# Seuil sous lequel le contenu texte est considéré trop pauvre → fallback vision
MIN_CONTENT_FOR_TEXT_SUMMARY = 500

SUMMARY_PROMPT_TEMPLATE = """Tu es un DPO senior indexant des documents pour ta base de connaissances professionnelle.

Produis une FICHE SYNTHÉTIQUE pour ce document. Cette fiche sera utilisée par un moteur de recherche sémantique pour retrouver le document quand un DPO pose une question.

FORMAT OBLIGATOIRE (150-200 tokens) :

NATURE: [DOCTRINE | GUIDE | SANCTION | TECHNIQUE]
TYPE: [guide | faq | referentiel | deliberation | recommandation | fiche | modele | article | autre]
SUJETS: [max 5 mots-clés discriminants, séparés par virgule]
USAGE DPO: [1 phrase : quand et pourquoi un DPO consulterait ce document]
SECTEUR: [générique | santé | RH | marketing | éducation | vidéosurveillance | banque | autre]
OBLIGATIONS CITÉES: [articles RGPD ou loi mentionnés, ex: art. 35, art. 28, art. 33]
CONTIENT:
- [élément concret 1 : obligation, procédure, critère, modèle...]
- [élément concret 2]
- [élément concret 3]
NE CONTIENT PAS:
- [ce qu'on pourrait croire y trouver mais qui n'y est pas]

RÈGLES :
- SUJETS doit être DISCRIMINANT (pas "RGPD" ni "données personnelles" qui sont dans tous les docs)
- USAGE DPO doit répondre à : "Je suis DPO et je cherche..." → ce document répond à quoi ?
- OBLIGATIONS CITÉES : lister les articles de loi/RGPD mentionnés (vide si aucun)
- Sois FACTUEL, pas de supposition

Document:
---
{content}
---

FICHE:"""


def load_document_from_chunks(collection, doc_path: str) -> str:
    """
    Reconstitue un document en récupérant tous ses chunks depuis ChromaDB.
    """
    try:
        # Récupère tous les chunks de ce document
        results = collection.get(
            where={"document_path": doc_path},
            include=["documents", "metadatas"]
        )
        
        if not results or not results.get("documents"):
            logger.warning(f"Aucun chunk trouvé pour {doc_path}")
            return ""
        
        # Trie les chunks par index
        chunks_with_index = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunk_index = meta.get("chunk_index", 0)
            chunks_with_index.append((chunk_index, doc))
        
        chunks_with_index.sort(key=lambda x: x[0])
        
        # Concatène les chunks (limite à 8000 premiers chars - Nemo 128K)
        full_text = "\n\n".join([chunk for _, chunk in chunks_with_index])
        
        # Limite à ~8000 chars (plus riche pour Nemo 12B)
        if len(full_text) > 8000:
            full_text = full_text[:8000] + "\n\n[...document tronqué...]"
        
        return full_text
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement de {doc_path}: {e}")
        return ""


def is_navigation_content(content: str, doc_path: str = "") -> bool:
    """
    Détecte si le contenu est principalement une page de navigation/menu HTML.
    
    Ne s'applique QU'AUX HTML. Les PDFs infographiques et images SCHEMA_DPO
    ont structurellement peu de texte extractible mais restent pertinents.
    """
    # Seuls les HTML peuvent être des pages de navigation
    is_html = doc_path.endswith('.html')
    
    # PDF et images : jamais considérés comme "navigation"
    # Leur contenu court (OCR, extraction graphique) EST leur contenu réel
    if not is_html:
        return False
    
    # HTML : contenu vraiment vide (< 100 chars) = pas de contenu
    if len(content.strip()) < 100:
        return True
    
    # Patterns typiques de pages-portail/hub CNIL
    nav_patterns = [
        r'En savoir plus',
        r'Lire la suite',
        r'Voir aussi',
        r'Consulter',
        r'En détail',
        r'Accéder',
        r'Découvrir',
        r'Toutes les actualités',
        r'Rechercher',
        r'Affiner la recherche',
        r'\d+ résultat',
        r'Page \d+ sur \d+',
        r'Suivant|Précédent',
    ]
    nav_count = sum(len(re.findall(p, content, re.IGNORECASE)) for p in nav_patterns)
    
    # Contenu substantiel (> 2000 chars) = jamais navigation, même avec des
    # mots-clés nav ("Consulter", "En savoir plus") dans du texte légitime.
    # Les chunks sont souvent des blocs monoligne, donc long_lines est un
    # mauvais proxy — on se base sur la longueur totale.
    if len(content.strip()) > 2000:
        return False
    
    # Contenu court (< 2000 chars) avec beaucoup de patterns nav = portail
    if nav_count >= 3 and len(content.strip()) < 1000:
        return True
    
    # Page HTML très courte (< 500 chars) avec au moins 1 pattern nav
    if len(content.strip()) < 500 and nav_count >= 1:
        return True
    
    return False


def generate_summary(llm: OllamaProvider, content: str, doc_url: str) -> Dict:
    """
    Génère un résumé du document avec Mistral.
    """
    if not content:
        return {
            "summary": "Document vide ou non accessible",
            "error": "empty_content"
        }
    
    try:
        prompt = SUMMARY_PROMPT_TEMPLATE.format(content=content)
        
        response = llm.generate(
            prompt=prompt,
            temperature=0.1,  # Très déterministe pour classification
            max_tokens=250    # Fiche enrichie (150-200 tokens cible)
        )
        
        summary = response.strip()
        
        if not summary or len(summary) < 30:
            return {
                "summary": f"Erreur: résumé trop court ({len(summary)} chars)",
                "error": "summary_too_short"
            }
        
        # Post-processing: normalisation du secteur
        summary = normalize_secteur(summary)
        
        return {
            "summary": summary,
            "length": len(summary),
            "source_url": doc_url
        }
        
    except Exception as e:
        logger.error(f"Erreur génération résumé: {e}")
        return {
            "summary": f"Erreur lors de la génération: {str(e)}",
            "error": str(e)
        }


VISUAL_SUMMARY_PROMPT = """Tu es un DPO senior. Tu analyses une image ou infographie provenant du site de la CNIL.

Produis une FICHE SYNTHÉTIQUE de ce document visuel. Cette fiche sera utilisée par un moteur de recherche.

FORMAT OBLIGATOIRE (150-200 tokens) :

NATURE: [DOCTRINE | GUIDE | SANCTION | TECHNIQUE]
TYPE: [infographie | schema | graphique | tableau | autre]
SUJETS: [max 5 mots-clés discriminants]
USAGE DPO: [1 phrase : quand un DPO consulterait ce document]
SECTEUR: [générique | santé | RH | marketing | éducation | vidéosurveillance | autre]
CONTIENT:
- [élément concret 1]
- [élément concret 2]
- [élément concret 3]

Décris TOUT le texte et les données visibles. Sois exhaustif sur les chiffres, légendes et titres."""


def generate_summary_visual(doc_path: str, doc_url: str) -> Dict:
    """Génère un résumé via LLaVA pour les documents visuels (images, PDFs infographiques).
    
    Envoie directement le fichier source à LLaVA au lieu de se baser sur le texte des chunks.
    """
    import base64
    import io
    import requests
    from PIL import Image
    
    OLLAMA_URL = "http://localhost:11434"
    VISION_MODEL = "llava:7b"
    MAX_DIM = 1024
    
    try:
        # Résoudre le chemin du fichier source
        file_path = _resolve_file_path(doc_path)
        if not file_path or not file_path.exists():
            return {"summary": "Fichier source introuvable", "error": "file_not_found"}
        
        images_b64 = []
        
        if file_path.suffix.lower() == '.pdf':
            # PDF → convertir pages en images
            import fitz
            doc = fitz.open(file_path)
            for page_idx in range(min(len(doc), 5)):  # Max 5 pages
                mat = fitz.Matrix(2.0, 2.0)
                pix = doc[page_idx].get_pixmap(matrix=mat)
                img = Image.open(io.BytesIO(pix.tobytes("jpeg")))
                w, h = img.size
                if max(w, h) > MAX_DIM:
                    ratio = MAX_DIM / max(w, h)
                    img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format='JPEG', quality=85)
                images_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
            doc.close()
        else:
            # Image directe
            img = Image.open(file_path).convert('RGB')
            w, h = img.size
            if max(w, h) > MAX_DIM:
                ratio = MAX_DIM / max(w, h)
                img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=85)
            images_b64.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
        
        if not images_b64:
            return {"summary": "Aucune image extractible", "error": "no_image"}
        
        # Envoyer la première image (principale) à LLaVA
        response = requests.post(
            f'{OLLAMA_URL}/api/generate',
            json={
                'model': VISION_MODEL,
                'prompt': VISUAL_SUMMARY_PROMPT,
                'images': [images_b64[0]],
                'stream': False,
                'options': {'temperature': 0.1, 'num_predict': 400}
            },
            timeout=120
        )
        
        if response.status_code != 200:
            return {"summary": f"LLaVA HTTP {response.status_code}", "error": "vision_http_error"}
        
        summary = response.json().get('response', '').strip()
        
        if not summary or len(summary) < 30:
            return {"summary": f"Résumé vision trop court ({len(summary)} chars)", "error": "vision_too_short"}
        
        summary = normalize_secteur(summary)
        
        return {
            "summary": summary,
            "length": len(summary),
            "source_url": doc_url,
            "method": "vision_llava"
        }
        
    except Exception as e:
        logger.error(f"Erreur résumé vision pour {doc_path}: {e}")
        return {"summary": f"Erreur vision: {str(e)}", "error": str(e)}


def _resolve_file_path(doc_path: str) -> Path:
    """Résout le chemin d'un document vers son fichier physique dans keep/."""
    # doc_path = 'data\\raw\\cnil\\pdf\\abc123.pdf'
    p = Path(doc_path)
    stem = p.stem
    suffix = p.suffix.lower()
    
    # Mapper vers keep/
    if suffix == '.html':
        candidate = KEEP_PATH / "html" / f"{stem}.html"
    elif suffix == '.pdf':
        candidate = KEEP_PATH / "pdf" / f"{stem}.pdf"
    elif suffix in ('.png', '.jpg', '.jpeg', '.gif', '.webp'):
        # Chercher dans keep/images/ avec n'importe quelle extension
        for img in KEEP_PATH.glob(f"images/{stem}.*"):
            return img
        candidate = KEEP_PATH / "images" / p.name
    else:
        candidate = KEEP_PATH / "docs" / p.name
    
    return candidate if candidate.exists() else None


def normalize_secteur(summary: str) -> str:
    """
    Normalise le champ SECTEUR pour n'avoir qu'une seule valeur.
    Si plusieurs secteurs détectés → générique
    """
    # Cherche ligne SECTEUR
    secteur_match = re.search(r'SECTEUR:\s*(.+)', summary, re.IGNORECASE)
    if not secteur_match:
        return summary
    
    secteur_line = secteur_match.group(1).strip()
    
    # Détecte plusieurs secteurs (virgule, "et", slash, etc.)
    if any(sep in secteur_line.lower() for sep in [',', ' et ', '/', '+']):
        # Remplace par "générique"
        normalized = re.sub(
            r'(SECTEUR:\s*)(.+)',
            r'\1générique',
            summary,
            flags=re.IGNORECASE
        )
        return normalized
    
    return summary


def get_all_unique_documents(collection) -> List[Dict]:
    """
    Récupère la liste de tous les documents uniques depuis ChromaDB.
    """
    logger.info("Récupération de tous les documents depuis ChromaDB...")
    
    # Récupère tous les chunks
    all_chunks = collection.get(
        include=["metadatas"]
    )
    
    # Extrait documents uniques avec leurs métadonnées
    docs_dict = {}
    for meta in all_chunks["metadatas"]:
        doc_path = meta.get("document_path")
        if doc_path and doc_path not in docs_dict:
            docs_dict[doc_path] = {
                "document_path": doc_path,
                "chunk_nature": meta.get("chunk_nature", "UNKNOWN"),
                "document_title": meta.get("document_title", "Sans titre"),
                "url_hash": meta.get("url_hash", "")
            }
    
    docs_list = list(docs_dict.values())
    logger.info(f"✅ {len(docs_list)} documents uniques trouvés")
    
    return docs_list


def load_metadata(url_hash: str) -> Dict:
    """Charge les métadonnées complètes depuis le fichier JSON."""
    metadata_path = METADATA_DIR / f"{url_hash}.json"
    
    if not metadata_path.exists():
        return {}
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Erreur lecture métadonnées {url_hash}: {e}")
        return {}


def main():
    """Génère les résumés pour tous les documents."""
    import argparse
    parser = argparse.ArgumentParser(description='Phase 6B : Génération résumés')
    parser.add_argument('--test', type=int, help='Limiter à N documents')
    args = parser.parse_args()
    max_docs = args.test
    
    logger.info("Demarrage generation resumes documents CNIL")
    logger.info(f"ChromaDB: {CHROMADB_PATH}")
    logger.info(f"Output: {SUMMARIES_OUTPUT}")
    
    # 1. Connexion ChromaDB
    logger.info("Connexion à ChromaDB...")
    client = PersistentClient(path=str(CHROMADB_PATH))
    collection = client.get_collection("rag_dpo_chunks")
    logger.info(f"✅ Collection chargée: {collection.count()} chunks")
    
    # 2. Initialisation LLM
    logger.info("Initialisation Mistral Nemo 12B...")
    llm = OllamaProvider(
        model="mistral-nemo",
        base_url="http://localhost:11434"
    )
    logger.info("✅ Mistral Nemo prêt")
    
    # 3. Récupération documents uniques
    documents = get_all_unique_documents(collection)
    
    # 4. Chargement résumés existants (pour reprise)
    existing_summaries = {}
    if SUMMARIES_OUTPUT.exists():
        logger.info(f"Chargement résumés existants depuis {SUMMARIES_OUTPUT}")
        with open(SUMMARIES_OUTPUT, "r", encoding="utf-8") as f:
            existing_summaries = json.load(f)
        logger.info(f"✅ {len(existing_summaries)} résumés existants")
    
    # 5. Génération résumés
    summaries = existing_summaries.copy()
    skipped = 0
    errors = 0
    
    if max_docs:
        documents = documents[:max_docs]
        logger.info(f"MODE TEST : {max_docs} documents")
    
    logger.info(f"Generation de {len(documents)} resumes...")
    
    for doc in tqdm(documents, desc="Generation resumes"):
        doc_path = doc["document_path"]
        url_hash = doc["url_hash"]
        
        # Skip si déjà fait
        if doc_path in summaries and "error" not in summaries[doc_path]:
            skipped += 1
            continue
        
        # Extraire url_hash depuis le chemin si vide
        if not url_hash:
            # Ex: data\\raw\\cnil\\html\\bf239ee48384.html -> bf239ee48384
            match = re.search(r'([a-f0-9]{12})\.(html|pdf)', doc_path)
            if match:
                url_hash = match.group(1)
        
        # Charge métadonnées complètes
        metadata = load_metadata(url_hash) if url_hash else {}
        doc_url = metadata.get("url", "URL inconnue")
        
        # Extraire titre du document depuis l'URL si possible
        doc_title = doc.get("document_title", "Sans titre")
        if doc_url != "URL inconnue" and doc_title == "Sans titre":
            # Extrait dernière partie de l'URL comme titre
            title_candidate = doc_url.rstrip('/').split('/')[-1]
            if title_candidate:
                doc_title = title_candidate.replace('-', ' ').title()[:100]
        
        # Reconstitue le document depuis ses chunks
        content = load_document_from_chunks(collection, doc_path)
        
        if not content:
            logger.warning(f"⚠️  Document vide: {doc_path}")
            summaries[doc_path] = {
                "summary": "Document vide ou inaccessible",
                "error": "empty_content",
                "source_url": doc_url,
                "document_title": doc_title,
                "metadata": doc
            }
            errors += 1
            continue
        
        # Détection contenu navigation/menu (HTML uniquement)
        if is_navigation_content(content, doc_path):
            logger.info(f"🚫 Navigation détectée, skip: {doc_title}")
            summaries[doc_path] = {
                "summary": "Page de navigation/menu — pas de contenu substantiel",
                "error": "navigation_content",
                "source_url": doc_url,
                "document_title": doc_title,
                "metadata": doc
            }
            skipped += 1
            continue
        
        # Fallback vision pour contenus trop courts (images, PDFs infographiques)
        is_visual = not doc_path.endswith('.html')
        if is_visual and len(content.strip()) < MIN_CONTENT_FOR_TEXT_SUMMARY:
            logger.info(f"📸 Fallback vision pour {doc_path} ({len(content)} chars)")
            summary_data = generate_summary_visual(doc_path, doc_url)
            summaries[doc_path] = {
                **summary_data,
                "document_title": doc_title,
                "metadata": doc
            }
            if "error" in summary_data:
                errors += 1
            if len(summaries) % 50 == 0:
                with open(SUMMARIES_OUTPUT, "w", encoding="utf-8") as f:
                    json.dump(summaries, f, indent=2, ensure_ascii=False)
            continue
        
        # Génère résumé (texte standard)
        try:
            summary_data = generate_summary(llm, content, doc_url)
            summaries[doc_path] = {
                **summary_data,
                "document_title": doc_title,
                "metadata": doc
            }
            
            if "error" in summary_data:
                errors += 1
            
            # Sauvegarde incrémentale tous les 50 documents
            if len(summaries) % 50 == 0:
                with open(SUMMARIES_OUTPUT, "w", encoding="utf-8") as f:
                    json.dump(summaries, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Sauvegarde incrémentale: {len(summaries)} résumés")
                
        except Exception as e:
            logger.error(f"❌ Erreur pour {doc_path}: {e}")
            summaries[doc_path] = {
                "summary": f"Erreur: {str(e)}",
                "error": str(e),
                "metadata": doc
            }
            errors += 1
    
    # 6. Sauvegarde finale
    logger.info(f"💾 Sauvegarde finale: {len(summaries)} résumés")
    with open(SUMMARIES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    
    logger.info("=" * 80)
    logger.info("✅ GÉNÉRATION TERMINÉE")
    logger.info(f"   Total documents: {len(documents)}")
    logger.info(f"   Résumés générés: {len(summaries) - skipped}")
    logger.info(f"   Résumés existants (skip): {skipped}")
    logger.info(f"   Erreurs: {errors}")
    logger.info(f"   Fichier output: {SUMMARIES_OUTPUT}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
