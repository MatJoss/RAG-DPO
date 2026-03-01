"""
Pipeline d'ingestion de documents entreprise dans le RAG-DPO.

Permet à un DPO d'ajouter ses propres documents internes (politiques, registres,
contrats, PIA, etc.) dans la même collection ChromaDB que les docs CNIL.
Les documents entreprise sont marqués source="enterprise" pour :
- Distinction dans les réponses ([CNIL] vs [Interne])
- Purge facile sans affecter le corpus CNIL
- Filtrage optionnel au retrieval

Usage :
    python src/processing/ingest_enterprise.py data/enterprise/    # Dossier entier
    python src/processing/ingest_enterprise.py mon_doc.pdf         # Fichier unique
    python src/processing/ingest_enterprise.py --purge             # Supprime tous les docs entreprise
    python src/processing/ingest_enterprise.py --list              # Liste les docs entreprise indexés
    python src/processing/ingest_enterprise.py --stats             # Statistiques de l'index
"""
import json
import sys
import time
import hashlib
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# Setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
for noisy in ("httpx", "ollama", "chromadb", "sentence_transformers", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────
COLLECTION_NAME = "rag_dpo_chunks"
CHROMA_PATH = project_root / "data" / "vectordb" / "chromadb"
TAGS_REGISTRY_PATH = project_root / "configs" / "enterprise_tags.json"
SUPPORTED_EXTENSIONS = {'.pdf', '.html', '.htm', '.docx', '.doc', '.xlsx', '.xls', '.odt', '.txt'}


def get_file_hash(file_path: Path) -> str:
    """Calcule le hash SHA256 d'un fichier pour déduplication."""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(8192), b''):
            h.update(block)
    return h.hexdigest()[:12]


def make_document_id(file_path: Path) -> str:
    """Génère un document_id stable basé sur le nom + hash du contenu."""
    file_hash = get_file_hash(file_path)
    return f"ent_{file_hash}"


def detect_file_type(file_path: Path) -> str:
    """Détecte le type de fichier."""
    ext = file_path.suffix.lower()
    type_map = {
        '.html': 'html', '.htm': 'html',
        '.pdf': 'pdf',
        '.docx': 'docx', '.doc': 'docx',
        '.xlsx': 'xlsx', '.xls': 'xlsx',
        '.odt': 'odt',
        '.txt': 'txt',
    }
    return type_map.get(ext, 'unknown')


def chunk_text_file(file_path: Path) -> List[Dict]:
    """Chunking simple pour fichiers texte brut."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().strip()
    
    if not text:
        return []
    
    return [{'text': text, 'heading': file_path.stem, 'page_info': ''}]


def extract_and_chunk(file_path: Path) -> List[Dict]:
    """Extrait le texte et découpe en chunks.
    
    Réutilise le StructuralChunker du pipeline CNIL pour un traitement
    identique (overlap, split sémantique, merge tiny, heading prefix).
    """
    from processing.process_and_chunk import StructuralChunker
    
    chunker = StructuralChunker()
    file_type = detect_file_type(file_path)
    
    try:
        if file_type == 'html':
            chunks = chunker.chunk_html(file_path)
        elif file_type == 'pdf':
            chunks = chunker.chunk_pdf(file_path)
        elif file_type in ('docx', 'odt', 'xlsx', 'xls'):
            chunks = chunker.chunk_doc(file_path, file_type)
        elif file_type == 'txt':
            raw_chunks = chunk_text_file(file_path)
            chunks = chunker._post_process(raw_chunks)
        else:
            logger.warning(f"⚠️  Type non supporté : {file_type} ({file_path.name})")
            return []
        
        return chunks
    except Exception as e:
        logger.error(f"❌ Erreur extraction {file_path.name}: {e}")
        return []


def ingest_files(
    paths: List[Path],
    collection,
    embedding_provider,
    tags: List[str] = None,
    batch_size: int = 50,
) -> Dict:
    """Ingère une liste de fichiers dans ChromaDB.
    
    Returns:
        Statistiques d'ingestion
    """
    stats = {
        'files_processed': 0,
        'files_skipped': 0,
        'files_error': 0,
        'chunks_indexed': 0,
        'chunks_total': 0,
    }
    
    # Récupérer les document_id entreprise déjà indexés
    existing_docs = set()
    try:
        existing = collection.get(
            where={"source": "ENTREPRISE"},
            include=[],
        )
        for mid in (existing.get('ids') or []):
            doc_id = mid.rsplit('_', 1)[0]  # chunk_id = docid_N
            existing_docs.add(doc_id)
    except Exception:
        pass
    
    all_chunks_to_index = []
    
    for file_path in tqdm(paths, desc="Extraction"):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            logger.info(f"⏭️  Ignoré (type non supporté) : {file_path.name}")
            stats['files_skipped'] += 1
            continue
        
        doc_id = make_document_id(file_path)
        
        # Skip si déjà indexé (même hash)
        if doc_id in existing_docs:
            logger.info(f"♻️  Déjà indexé : {file_path.name} ({doc_id})")
            stats['files_skipped'] += 1
            continue
        
        # Extraire et chunker
        chunks = extract_and_chunk(file_path)
        
        if not chunks:
            logger.warning(f"⚠️  Pas de contenu extractible : {file_path.name}")
            stats['files_error'] += 1
            continue
        
        file_type = detect_file_type(file_path)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            text = chunk.get('text', '').strip()
            heading = chunk.get('heading', '').strip()
            
            if not text or len(text.split()) < 10:
                continue
            
            full_text = f"[{heading}] {text}" if heading and not text.startswith(f"[{heading}]") else text
            
            metadata = {
                'document_id': doc_id,
                'document_path': str(file_path),
                'document_nature': 'INTERNE',
                'chunk_nature': 'INTERNE',
                'chunk_index': 'OPERATIONNEL',
                'heading': heading[:200] if heading else '',
                'page_info': chunk.get('page_info', ''),
                'confidence': 0.8,
                'method': 'structural',
                'word_count': len(text.split()),
                'sectors': '',
                'file_type': file_type,
                'title': file_path.stem[:300],
                'source': 'ENTREPRISE',
                'source_type': file_type,
                'is_priority': False,
                'source_url': str(file_path),
                'parent_url': '',
                'ingested_at': datetime.now().isoformat(),
            }
            
            # Champs booléens par tag (filtrables nativement par ChromaDB)
            for tag in (tags or []):
                metadata[f'tag_{tag}'] = True
            
            all_chunks_to_index.append({
                'id': chunk_id,
                'text': full_text,
                'metadata': metadata,
            })
        
        stats['files_processed'] += 1
    
    stats['chunks_total'] = len(all_chunks_to_index)
    
    if not all_chunks_to_index:
        print("ℹ️  Aucun nouveau chunk à indexer.")
        return stats
    
    # Générer embeddings + indexer par batch
    print(f"\n📐 Embedding + indexation de {len(all_chunks_to_index)} chunks...")
    
    for i in tqdm(range(0, len(all_chunks_to_index), batch_size), desc="Indexation"):
        batch = all_chunks_to_index[i:i + batch_size]
        
        ids = [c['id'] for c in batch]
        texts = [c['text'] for c in batch]
        metadatas = [c['metadata'] for c in batch]
        
        try:
            embeddings = embedding_provider.embed(texts)
            
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            stats['chunks_indexed'] += len(batch)
        except Exception as e:
            logger.error(f"❌ Erreur indexation batch {i // batch_size}: {e}")
    
    return stats


def purge_enterprise(collection) -> int:
    """Supprime tous les documents entreprise de la collection.
    
    Returns:
        Nombre de chunks supprimés
    """
    try:
        existing = collection.get(
            where={"source": "ENTREPRISE"},
            include=[],
        )
        ids = existing.get('ids', [])
        if not ids:
            return 0
        
        # ChromaDB delete par batch de 5000
        for i in range(0, len(ids), 5000):
            batch_ids = ids[i:i + 5000]
            collection.delete(ids=batch_ids)
        
        return len(ids)
    except Exception as e:
        logger.error(f"❌ Erreur purge : {e}")
        return 0


def purge_by_tag(collection, tag: str) -> int:
    """Supprime les chunks entreprise ayant un tag spécifique.
    
    Utilise le filtre natif ChromaDB sur le champ booléen tag_xxx.
    
    Returns:
        Nombre de chunks supprimés
    """
    tag_field = f'tag_{tag}'
    try:
        existing = collection.get(
            where={"$and": [
                {"source": "ENTREPRISE"},
                {tag_field: True}
            ]},
            include=[],
        )
        
        ids = existing.get('ids', [])
        if not ids:
            return 0
        
        for i in range(0, len(ids), 5000):
            batch_ids = ids[i:i + 5000]
            collection.delete(ids=batch_ids)
        
        return len(ids)
    except Exception as e:
        logger.error(f"❌ Erreur purge tag '{tag}' : {e}")
        return 0


def list_enterprise_docs(collection) -> List[Dict]:
    """Liste les documents entreprise indexés."""
    try:
        existing = collection.get(
            where={"source": "ENTREPRISE"},
            include=["metadatas"],
        )
        
        docs = {}
        for mid, meta in zip(existing.get('ids', []), existing.get('metadatas', [])):
            doc_id = meta.get('document_id', '')
            if doc_id not in docs:
                # Extraire les tags depuis les champs booléens tag_xxx
                doc_tags = [k[4:] for k, v in meta.items() if k.startswith('tag_') and v is True]
                docs[doc_id] = {
                    'document_id': doc_id,
                    'document_path': meta.get('document_path', ''),
                    'file_type': meta.get('file_type', ''),
                    'title': meta.get('title', ''),
                    'enterprise_tags': ','.join(sorted(doc_tags)),
                    'ingested_at': meta.get('ingested_at', ''),
                    'n_chunks': 0,
                }
            docs[doc_id]['n_chunks'] += 1
        
        return list(docs.values())
    except Exception as e:
        logger.error(f"❌ Erreur listing : {e}")
        return []


def get_stats(collection) -> Dict:
    """Statistiques de la collection."""
    total = collection.count()
    
    try:
        enterprise = collection.get(
            where={"source": "ENTREPRISE"},
            include=[],
        )
        n_enterprise = len(enterprise.get('ids', []))
    except Exception:
        n_enterprise = 0
    
    n_cnil = total - n_enterprise
    
    return {
        'total_chunks': total,
        'cnil_chunks': n_cnil,
        'enterprise_chunks': n_enterprise,
        'enterprise_pct': f"{n_enterprise / total * 100:.1f}%" if total > 0 else "0%",
    }


def load_tags_registry() -> Dict:
    """Charge le registre des tags depuis le fichier JSON."""
    if TAGS_REGISTRY_PATH.exists():
        with open(TAGS_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"default_tags": [], "active_tags": [], "labels": {}}


def save_tags_registry(registry: Dict):
    """Sauvegarde le registre des tags."""
    with open(TAGS_REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def update_tags_registry(new_tags: List[str]):
    """Ajoute des tags au registre s'ils n'y sont pas déjà."""
    registry = load_tags_registry()
    active = set(registry.get('active_tags', []))
    added = []
    for tag in new_tags:
        if tag not in active:
            active.add(tag)
            added.append(tag)
    if added:
        registry['active_tags'] = sorted(active)
        save_tags_registry(registry)
        logger.info(f"🏷️  Registre MAJ : +{added} → {sorted(active)}")


def refresh_tags_registry(collection):
    """Recalcule les tags actifs en scannant la collection ChromaDB.
    
    Appelé après purge pour retirer les tags orphelins.
    """
    registry = load_tags_registry()
    
    try:
        existing = collection.get(
            where={"source": "ENTREPRISE"},
            include=["metadatas"],
        )
        
        active = set()
        for meta in existing.get('metadatas', []):
            for key, val in meta.items():
                if key.startswith('tag_') and val is True:
                    active.add(key[4:])  # tag_registre → registre
        
        registry['active_tags'] = sorted(active)
        save_tags_registry(registry)
        logger.info(f"🏷️  Registre recalculé : {sorted(active)}")
    except Exception as e:
        logger.error(f"❌ Erreur refresh registre : {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Ingestion de documents entreprise dans le RAG-DPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python src/processing/ingest_enterprise.py docs/            # Dossier entier
  python src/processing/ingest_enterprise.py rapport.pdf      # Fichier unique
  python src/processing/ingest_enterprise.py --purge          # Supprime les docs entreprise
  python src/processing/ingest_enterprise.py --list           # Liste les docs indexés
  python src/processing/ingest_enterprise.py --stats          # Statistiques
        """
    )
    parser.add_argument("paths", nargs="*", help="Fichiers ou dossiers à ingérer")
    parser.add_argument("--tags", "-t", type=str, default="",
                        help="Tags métier séparés par des virgules (ex: registre,service_rh,pia)")
    parser.add_argument("--purge", action="store_true", help="Supprime tous les docs entreprise")
    parser.add_argument("--purge-tag", type=str, default="",
                        help="Supprime uniquement les docs avec ce tag")
    parser.add_argument("--list", action="store_true", help="Liste les docs entreprise indexés")
    parser.add_argument("--stats", action="store_true", help="Affiche les statistiques")
    parser.add_argument("--recursive", "-r", action="store_true", help="Parcours récursif des dossiers")
    
    args = parser.parse_args()
    
    # Init ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        print(f"❌ Collection '{COLLECTION_NAME}' introuvable.")
        print(f"   Lancez d'abord l'indexation CNIL.")
        sys.exit(1)
    
    # ── Commandes sans fichiers ──
    if args.purge:
        print("🗑️  Purge des documents entreprise...")
        n = purge_enterprise(collection)
        print(f"✅ {n} chunks entreprise supprimés.")
        # Vider le registre
        refresh_tags_registry(collection)
        return
    
    if args.purge_tag:
        tag = args.purge_tag.strip()
        print(f"🗑️  Purge des documents avec tag '{tag}'...")
        n = purge_by_tag(collection, tag)
        print(f"✅ {n} chunks supprimés (tag: {tag}).")
        # MAJ registre : recalculer les tags actifs
        refresh_tags_registry(collection)
        return
    
    if args.list:
        docs = list_enterprise_docs(collection)
        if not docs:
            print("ℹ️  Aucun document entreprise indexé.")
            return
        print(f"\n📄 Documents entreprise indexés ({len(docs)}) :")
        print(f"  {'ID':<15} {'Type':<6} {'Chunks':>7}  {'Tags':<25} {'Fichier'}")
        print(f"  {'-'*90}")
        for d in sorted(docs, key=lambda x: x.get('ingested_at', '')):
            tags_str = d.get('enterprise_tags', '') or '-'
            print(f"  {d['document_id']:<15} {d['file_type']:<6} {d['n_chunks']:>7}  {tags_str:<25} {d['document_path']}")
        return
    
    if args.stats:
        s = get_stats(collection)
        print(f"\n📊 Statistiques collection '{COLLECTION_NAME}' :")
        print(f"  Total chunks      : {s['total_chunks']}")
        print(f"  Chunks CNIL       : {s['cnil_chunks']}")
        print(f"  Chunks entreprise : {s['enterprise_chunks']} ({s['enterprise_pct']})")
        return
    
    # ── Ingestion ──
    if not args.paths:
        parser.print_help()
        sys.exit(1)
    
    # Collecter les fichiers
    files = []
    for p in args.paths:
        path = Path(p)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            if args.recursive:
                for ext in SUPPORTED_EXTENSIONS:
                    files.extend(path.rglob(f"*{ext}"))
            else:
                for ext in SUPPORTED_EXTENSIONS:
                    files.extend(path.glob(f"*{ext}"))
        else:
            print(f"⚠️  Chemin introuvable : {p}")
    
    if not files:
        print("❌ Aucun fichier supporté trouvé.")
        print(f"   Extensions supportées : {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        sys.exit(1)
    
    files = sorted(set(files))  # Dédupliquer
    
    print("=" * 70)
    print("📥 INGESTION DOCUMENTS ENTREPRISE")
    print("=" * 70)
    print(f"  Fichiers trouvés : {len(files)}")
    print(f"  Types : {', '.join(sorted(set(f.suffix.lower() for f in files)))}")
    print(f"  Collection : {COLLECTION_NAME}")
    print()
    
    # Parse tags
    tags = [t.strip() for t in args.tags.split(',') if t.strip()] if args.tags else []
    if tags:
        print(f"  Tags : {', '.join(tags)}")
    
    # Init embedding provider (BGE-M3)
    from utils.embedding_provider import EmbeddingProvider
    embedding_provider = EmbeddingProvider(
        cache_dir=str(project_root / "models" / "huggingface" / "hub"),
    )
    
    t_start = time.time()
    stats = ingest_files(files, collection, embedding_provider, tags=tags)
    elapsed = time.time() - t_start
    
    # MAJ automatique du registre des tags
    if tags and stats['chunks_indexed'] > 0:
        update_tags_registry(tags)
    
    # Résumé
    print(f"\n{'=' * 70}")
    print(f"📊 RÉSUMÉ INGESTION")
    print(f"{'=' * 70}")
    print(f"  📄 Fichiers traités  : {stats['files_processed']}")
    print(f"  ♻️  Fichiers ignorés  : {stats['files_skipped']}")
    if stats['files_error']:
        print(f"  ⚠️  Fichiers en erreur: {stats['files_error']}")
    print(f"  📦 Chunks indexés    : {stats['chunks_indexed']} / {stats['chunks_total']}")
    print(f"  ⏱️  Temps total      : {elapsed:.1f}s")
    
    # Stats globales
    s = get_stats(collection)
    print(f"\n  📊 Collection totale : {s['total_chunks']} chunks")
    print(f"     CNIL       : {s['cnil_chunks']}")
    print(f"     Entreprise : {s['enterprise_chunks']} ({s['enterprise_pct']})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
