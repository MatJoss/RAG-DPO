"""
Re-processing ciblé des fichiers tabulaires (ODS/XLSX) dans ChromaDB.

Ce script :
1. Identifie les chunks ODS/XLSX existants dans ChromaDB
2. Les supprime
3. Re-chunk les fichiers sources avec le nouveau chunker sémantique 
   (en-têtes + phrases naturelles)
4. Réindexe les nouveaux chunks

Usage :
    python rechunk_spreadsheets.py           # Exécuter le re-processing
    python rechunk_spreadsheets.py --dry-run  # Preview sans modifier
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'utils'))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'processing'))

import chromadb
from chromadb.config import Settings
from src.utils.embedding_provider import EmbeddingProvider
from src.processing.process_and_chunk import StructuralChunker, ChunkFeatureExtractor, ChunkClassifier
from src.utils.llm_provider import RAGConfig

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Re-chunk spreadsheets with semantic headers")
    parser.add_argument('--dry-run', action='store_true', help="Preview without modifying ChromaDB")
    args = parser.parse_args()
    
    print("=" * 70)
    print("RECHUNK SPREADSHEETS — En-têtes + Phrases Naturelles")
    print("=" * 70)
    
    # --- 1. Connect to ChromaDB ---
    chroma_path = PROJECT_ROOT / 'data' / 'vectordb' / 'chromadb'
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=False)
    )
    collection = client.get_collection("rag_dpo_chunks")
    
    initial_count = collection.count()
    print(f"\n📊 ChromaDB initial : {initial_count} chunks")
    
    # --- 2. Find all ODS/XLSX chunks ---
    all_data = collection.get(include=["metadatas"])
    
    spreadsheet_ids = []
    spreadsheet_docs = defaultdict(list)  # doc_id -> [chunk_ids]
    
    for chunk_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        ft = meta.get('file_type', '')
        if ft in ('ods', 'xlsx', 'xls'):
            spreadsheet_ids.append(chunk_id)
            doc_id = meta.get('document_id', '?')
            spreadsheet_docs[doc_id].append(chunk_id)
    
    print(f"📋 Chunks tabulaires trouvés : {len(spreadsheet_ids)} (de {len(spreadsheet_docs)} documents)")
    
    for doc_id, ids in spreadsheet_docs.items():
        print(f"   • {doc_id} : {len(ids)} chunks")
    
    if not spreadsheet_ids:
        print("\n✅ Aucun chunk tabulaire à re-traiter !")
        return
    
    # --- 3. Find source files ---
    # Load keep_manifest to find the ODS/XLSX files
    manifest_path = PROJECT_ROOT / 'data' / 'raw' / 'cnil' / 'keep_manifest.json'
    metadata_path = PROJECT_ROOT / 'data' / 'raw' / 'cnil' / 'document_metadata.json'
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        doc_classifications = json.load(f).get('metadata', {})
    
    # Build URL cache
    url_cache = {}
    for list_key in ['html', 'pdfs', 'docs', 'images']:
        for item in manifest.get(list_key, []):
            fp = item.get('metadata', {}).get('file_path', '')
            if fp:
                url_cache[fp] = {
                    'url': item.get('url', ''),
                    'parent_url': item.get('parent_url', item.get('metadata', {}).get('source_url', '') or ''),
                }
    
    # Find spreadsheet source files from docs list
    spreadsheet_files = []
    for doc_item in manifest.get('docs', []):
        file_path = doc_item['metadata']['file_path']
        fp = Path(file_path)
        if fp.suffix.lower() in ('.ods', '.xlsx', '.xls'):
            doc_meta = doc_classifications.get(file_path, {})
            if doc_meta and fp.stem in spreadsheet_docs:
                spreadsheet_files.append((file_path, doc_meta))
    
    print(f"\n📂 Fichiers source trouvés : {len(spreadsheet_files)}")
    for fp, meta in spreadsheet_files:
        print(f"   • {Path(fp).name} — {meta.get('title', meta.get('raison', '?'))[:60]}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN — Preview des nouveaux chunks :")
    
    # --- 4. Re-chunk with new semantic chunker ---
    chunker = StructuralChunker()
    feature_extractor = ChunkFeatureExtractor()
    
    config = RAGConfig()
    classifier = ChunkClassifier(config.llm_provider)
    
    nature_map = {
        'DOCTRINE': 'FONDAMENTAUX', 'FONDAMENTAUX': 'FONDAMENTAUX',
        'GUIDE': 'OPERATIONNEL', 'OPERATIONNEL': 'OPERATIONNEL',
        'SANCTION': 'SANCTIONS', 'SANCTIONS': 'SANCTIONS',
        'TECHNIQUE': 'TECHNIQUE',
    }
    
    new_chunks_all = []
    
    for file_path, doc_meta in spreadsheet_files:
        fp = Path(file_path)
        if not fp.exists():
            logger.warning(f"⚠️  Fichier introuvable : {file_path}")
            continue
        
        file_type = doc_meta.get('file_type', fp.suffix.lower()[1:])
        raw_chunks = chunker._chunk_spreadsheet(fp, file_type)
        
        doc_id = fp.stem
        doc_nature = doc_meta.get('nature', 'GUIDE')
        url_info = url_cache.get(file_path, {})
        
        print(f"\n   📄 {fp.name} : {len(raw_chunks)} nouveaux chunks sémantiques")
        
        for idx, raw_chunk in enumerate(raw_chunks):
            features = feature_extractor.extract(raw_chunk['text'])
            classification = classifier.classify(raw_chunk, features, doc_nature)
            chunk_nature = classification['chunk_nature']
            
            # Tags RGPD extraits par le LLM lors du chunking
            rgpd_topics = raw_chunk.get('rgpd_topics', [])
            
            chunk = {
                'id': f"{doc_id}_{idx}",
                'document': raw_chunk['text'][:5000],
                'metadata': {
                    'document_id': doc_id,
                    'document_path': file_path,
                    'document_nature': doc_nature,
                    'chunk_nature': chunk_nature,
                    'chunk_index': nature_map.get(chunk_nature, 'OPERATIONNEL'),
                    'heading': raw_chunk.get('heading', '')[:200],
                    'page_info': raw_chunk.get('page_info', ''),
                    'confidence': classification['confidence'],
                    'method': classification.get('method', 'unknown'),
                    'word_count': len(raw_chunk['text'].split()),
                    'sectors': ','.join(classification.get('sectors', [])),
                    'rgpd_topics': ','.join(rgpd_topics) if rgpd_topics else '',
                    'file_type': file_type,
                    'source_url': url_info.get('url', ''),
                    'parent_url': url_info.get('parent_url', ''),
                    'title': doc_meta.get('raison', '') or doc_meta.get('title', ''),
                    'source': 'CNIL',
                    'is_priority': False,
                    'source_type': 'odt',
                }
            }
            new_chunks_all.append(chunk)
            
            if args.dry_run and idx < 3:
                preview = raw_chunk['text'][:300].replace('\n', ' | ')
                tags_str = f" [TAGS: {', '.join(rgpd_topics)}]" if rgpd_topics else ""
                print(f"      chunk {idx}{tags_str}: {preview}")
    
    print(f"\n📊 Résumé :")
    print(f"   Anciens chunks à supprimer : {len(spreadsheet_ids)}")
    print(f"   Nouveaux chunks à indexer  : {len(new_chunks_all)}")
    print(f"   Delta                      : {len(new_chunks_all) - len(spreadsheet_ids):+d}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN terminé. Relancez sans --dry-run pour appliquer.")
        return
    
    # --- 5. Delete old chunks ---
    print(f"\n🗑️  Suppression de {len(spreadsheet_ids)} anciens chunks...")
    # ChromaDB delete en batches de 5000 max
    for i in range(0, len(spreadsheet_ids), 5000):
        batch_ids = spreadsheet_ids[i:i+5000]
        collection.delete(ids=batch_ids)
    print(f"   ✅ Supprimés")
    
    # --- 6. Embed and index new chunks ---
    print(f"\n🔄 Embedding + indexation de {len(new_chunks_all)} nouveaux chunks...")
    
    ep = EmbeddingProvider(cache_dir=str(PROJECT_ROOT / 'models' / 'huggingface' / 'hub'))
    
    BATCH_SIZE = 64
    for i in range(0, len(new_chunks_all), BATCH_SIZE):
        batch = new_chunks_all[i:i+BATCH_SIZE]
        
        texts = [c['document'] for c in batch]
        ids = [c['id'] for c in batch]
        metadatas = [c['metadata'] for c in batch]
        
        embeddings = ep.embed(texts)
        
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"   Batch {i//BATCH_SIZE + 1}/{(len(new_chunks_all)-1)//BATCH_SIZE + 1} indexé")
    
    final_count = collection.count()
    print(f"\n✅ Re-processing terminé !")
    print(f"   ChromaDB : {initial_count} → {final_count} chunks (delta {final_count - initial_count:+d})")
    
    # --- 7. Update JSONL ---
    print(f"\n📝 Mise à jour du JSONL...")
    jsonl_path = PROJECT_ROOT / 'data' / 'raw' / 'cnil' / 'processed_chunks.jsonl'
    
    if jsonl_path.exists():
        # Lire toutes les lignes, filtrer les anciens spreadsheet chunks, ajouter les nouveaux
        kept_lines = []
        removed = 0
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    doc_id = chunk.get('document_id', '')
                    file_type = chunk.get('file_type', '')
                    if file_type in ('ods', 'xlsx', 'xls') and doc_id in spreadsheet_docs:
                        removed += 1
                        continue
                    kept_lines.append(line)
                except json.JSONDecodeError:
                    kept_lines.append(line)
        
        # Ajouter les nouveaux chunks au format JSONL
        for c in new_chunks_all:
            jsonl_chunk = {
                'chunk_id': c['id'],
                'document_id': c['metadata']['document_id'],
                'document_path': c['metadata']['document_path'],
                'document_nature': c['metadata']['document_nature'],
                'chunk_nature': c['metadata']['chunk_nature'],
                'chunk_index': c['metadata']['chunk_index'],
                'sectors': c['metadata']['sectors'].split(',') if c['metadata']['sectors'] else [],
                'rgpd_topics': c['metadata']['rgpd_topics'].split(',') if c['metadata']['rgpd_topics'] else [],
                'text': c['document'],
                'heading': c['metadata']['heading'],
                'page_info': c['metadata']['page_info'],
                'confidence': c['metadata']['confidence'],
                'method': c['metadata']['method'],
                'file_type': c['metadata']['file_type'],
                'source_url': c['metadata']['source_url'],
                'parent_url': c['metadata']['parent_url'],
                'title': c['metadata']['title'],
            }
            kept_lines.append(json.dumps(jsonl_chunk, ensure_ascii=False))
        
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for line in kept_lines:
                f.write(line + '\n')
        
        print(f"   JSONL : -{removed} anciens, +{len(new_chunks_all)} nouveaux")
    
    print(f"\n🎉 Terminé !")


if __name__ == '__main__':
    main()
