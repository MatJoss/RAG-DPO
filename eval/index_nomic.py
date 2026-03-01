"""
Index les chunks dans une collection ChromaDB séparée avec nomic-embed-text (Ollama).

Crée la collection 'rag_dpo_chunks_nomic' (768 dims, cosine) à côté de
'rag_dpo_chunks' (BGE-M3, 1024 dims) pour permettre un benchmark comparatif.

Usage :
    python eval/index_nomic.py              # Indexation complète (~15 min)
    python eval/index_nomic.py --verify     # Vérifie l'index existant
"""
import json
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

# Setup
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
for noisy in ("httpx", "ollama", "chromadb", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────
COLLECTION_NAME = "rag_dpo_chunks_nomic"
CHROMA_PATH = project_root / "data" / "vectordb" / "chromadb"
CHUNKS_FILE = project_root / "data" / "raw" / "cnil" / "processed_chunks.jsonl"
KEEP_MANIFEST = project_root / "data" / "raw" / "cnil" / "keep_manifest.json"
NOMIC_MAX_CHARS = 2000  # nomic-embed-text context limit (~512 tokens)


def load_url_cache() -> Dict[str, str]:
    """Charge le cache URL depuis keep_manifest.json."""
    if not KEEP_MANIFEST.exists():
        logger.warning("⚠️  keep_manifest introuvable")
        return {}
    
    cache = {}
    with open(KEEP_MANIFEST, 'r', encoding='utf-8') as f:
        keep_data = json.load(f)
    
    for list_key in ['html', 'pdfs', 'docs']:
        for item in keep_data.get(list_key, []):
            metadata = item.get('metadata', {})
            file_path = metadata.get('file_path', '')
            url = metadata.get('url', '') or item.get('url', '')
            parent_url = item.get('parent_url', '') or metadata.get('source_url', '')
            
            if not file_path:
                continue
            normalized_path = file_path.replace('\\', '/')
            if url:
                cache[normalized_path] = url
            elif parent_url:
                cache[normalized_path] = parent_url
    
    logger.info(f"✅ Cache URLs chargé: {len(cache)} documents")
    return cache


def load_chunks() -> List[Dict]:
    """Charge les chunks depuis JSONL."""
    if not CHUNKS_FILE.exists():
        logger.error(f"❌ {CHUNKS_FILE} introuvable")
        return []
    
    chunks = []
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                chunks.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Erreur parsing: {e}")
    
    logger.info(f"📄 Chunks chargés : {len(chunks)}")
    return chunks


def embed_nomic(texts: List[str], client) -> List[List[float]]:
    """Génère les embeddings via nomic-embed-text (Ollama)."""
    embeddings = []
    for text in texts:
        # Troncature nomic-embed-text
        truncated = text[:NOMIC_MAX_CHARS]
        response = client.embeddings(
            model="nomic-embed-text",
            prompt=truncated
        )
        embeddings.append(response['embedding'])
    return embeddings


def detect_source_type(doc_path: str) -> str:
    """Détecte le type de fichier source."""
    ext = Path(doc_path).suffix.lower()
    type_map = {'.html': 'html', '.htm': 'html', '.pdf': 'pdf',
                '.docx': 'docx', '.doc': 'docx', '.xlsx': 'xlsx', '.xls': 'xlsx'}
    return type_map.get(ext, 'unknown')


def run_indexation():
    """Exécute l'indexation complète avec nomic-embed-text."""
    import ollama as ollama_lib
    
    print("=" * 70)
    print("🗄️  INDEXATION NOMIC-EMBED-TEXT → ChromaDB")
    print("=" * 70)
    
    # Vérifier Ollama + nomic
    print("\n⏳ Vérification Ollama + nomic-embed-text...")
    ollama_client = ollama_lib.Client(host="http://localhost:11434")
    try:
        test_emb = ollama_client.embeddings(model="nomic-embed-text", prompt="test")
        dims = len(test_emb['embedding'])
        print(f"✅ nomic-embed-text OK ({dims} dims)")
    except Exception as e:
        print(f"❌ Erreur nomic-embed-text: {e}")
        print("   Assurez-vous qu'Ollama tourne et que nomic-embed-text est installé:")
        print("   ollama pull nomic-embed-text")
        sys.exit(1)
    
    # Charger les chunks
    chunks = load_chunks()
    if not chunks:
        print("❌ Pas de chunks à indexer")
        sys.exit(1)
    
    url_cache = load_url_cache()
    
    # Init ChromaDB — collection séparée
    print(f"\n📂 ChromaDB: {CHROMA_PATH}")
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False, allow_reset=True)
    )
    
    # Supprimer collection existante si elle existe
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️  Collection '{COLLECTION_NAME}' supprimée")
    except Exception:
        pass
    
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Chunks RGPD/CNIL — nomic-embed-text (768 dims)",
            "hnsw:space": "cosine",
            "embedding_model": "nomic-embed-text",
            "embedding_dims": dims,
        }
    )
    print(f"✅ Collection '{COLLECTION_NAME}' créée ({dims} dims, cosine)")
    
    # Estimation temps
    est_minutes = len(chunks) / 100 * 1.5  # ~1.5 min par batch de 100 (Ollama séquentiel)
    print(f"\n📊 Chunks à indexer : {len(chunks)}")
    print(f"⏱️  Durée estimée : ~{est_minutes:.0f} minutes")
    print(f"   (nomic via Ollama = séquentiel, plus lent que BGE-M3 batch GPU)")
    
    # Indexation par batch
    batch_size = 50  # Plus petit que BGE-M3 car Ollama est séquentiel
    indexed = 0
    errors = 0
    t_start = time.time()
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Indexation nomic"):
        batch = chunks[i:i + batch_size]
        
        ids = []
        documents = []
        metadatas = []
        
        for chunk in batch:
            chunk_id = chunk.get('chunk_id', f"chunk_{i}")
            ids.append(chunk_id)
            
            text = chunk.get('text', '')
            heading = chunk.get('heading', '')
            full_text = f"{heading}\n\n{text}" if heading else text
            documents.append(full_text)
            
            doc_path = chunk.get('document_path', '')
            normalized = doc_path.replace('\\', '/')
            source_url = chunk.get('source_url', '') or url_cache.get(normalized, doc_path)
            
            metadata = {
                'document_id': chunk.get('document_id', ''),
                'document_path': doc_path,
                'document_nature': chunk.get('document_nature', 'GUIDE'),
                'chunk_nature': chunk.get('chunk_nature', 'GUIDE'),
                'chunk_index': chunk.get('chunk_index', 'OPERATIONNEL'),
                'heading': heading[:200] if heading else '',
                'page_info': chunk.get('page_info', ''),
                'confidence': chunk.get('confidence', 0.5),
                'method': chunk.get('method', 'unknown'),
                'word_count': len(text.split()),
                'sectors': ','.join(chunk.get('sectors', [])),
                'file_type': chunk.get('file_type', detect_source_type(doc_path)),
                'title': chunk.get('title', '')[:300],
                'source': 'CNIL',
                'source_type': detect_source_type(doc_path),
                'is_priority': False,
                'source_url': source_url,
                'parent_url': chunk.get('parent_url', ''),
            }
            metadatas.append(metadata)
        
        # Générer embeddings nomic
        try:
            embeddings = embed_nomic(documents, ollama_client)
            
            if len(embeddings) != len(documents):
                logger.warning(f"⚠️  Batch {i // batch_size}: embeddings incomplets")
                errors += len(batch)
                continue
            
            collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            indexed += len(batch)
            
        except Exception as e:
            logger.error(f"Erreur batch {i // batch_size}: {e}")
            errors += len(batch)
    
    elapsed = time.time() - t_start
    
    # Résumé
    print(f"\n{'=' * 70}")
    print(f"📊 RÉSUMÉ INDEXATION NOMIC")
    print(f"{'=' * 70}")
    print(f"  📄 Chunks chargés  : {len(chunks)}")
    print(f"  ✅ Chunks indexés  : {indexed}")
    if errors:
        print(f"  ⚠️  Erreurs        : {errors}")
    print(f"  ⏱️  Temps total    : {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"  📦 Collection      : {COLLECTION_NAME}")
    print(f"  📐 Dimensions      : {dims}")
    print(f"{'=' * 70}")


def verify_index():
    """Vérifie l'index nomic existant."""
    import ollama as ollama_lib
    
    print("=" * 70)
    print("🔍 VÉRIFICATION INDEX NOMIC")
    print("=" * 70)
    
    chroma_client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False)
    )
    
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        count = collection.count()
        print(f"\n✅ Collection '{COLLECTION_NAME}' : {count} chunks")
    except Exception as e:
        print(f"\n❌ Collection '{COLLECTION_NAME}' introuvable: {e}")
        print("   Lancez: python eval/index_nomic.py")
        return
    
    # Test query
    print("\n📋 Test query: 'Comment faire une AIPD ?'")
    ollama_client = ollama_lib.Client(host="http://localhost:11434")
    query_emb = ollama_client.embeddings(model="nomic-embed-text", prompt="Comment faire une AIPD ?")
    
    results = collection.query(
        query_embeddings=[query_emb['embedding']],
        n_results=5,
    )
    
    seen = set()
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        doc_path = meta.get('document_path', '')
        if doc_path not in seen:
            seen.add(doc_path)
            print(f"  📄 {doc_path[-60:]}")
            print(f"     Nature: {meta.get('chunk_nature')} | {doc[:100]}...")
    
    print(f"\n{'=' * 70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Indexation nomic-embed-text dans ChromaDB")
    parser.add_argument("--verify", action="store_true", help="Vérifie l'index existant")
    args = parser.parse_args()
    
    if args.verify:
        verify_index()
    else:
        run_indexation()


if __name__ == "__main__":
    main()
