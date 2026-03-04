"""
Tag RGPD unifié — Ajoute des tags RGPD à TOUS les chunks de ChromaDB via LLM.

Stratégie :
1. Lit chaque chunk depuis ChromaDB
2. Envoie le texte (tronqué à ~300 mots) à Nemo pour extraire 1-3 tags RGPD
   parmi ~25 catégories guidées (le LLM peut déborder si nécessaire)
3. Met à jour la metadata `rgpd_topics` via collection.update()

Le tagging est idempotent : les chunks déjà taggés sont skippés (sauf --force).
Sauvegarde incrémentale toutes les 100 chunks pour pouvoir reprendre après crash.

Usage :
    python tag_all_chunks.py                # Tagger les chunks sans tags
    python tag_all_chunks.py --force        # Re-tagger tous les chunks
    python tag_all_chunks.py --retry-failed # Re-tagger uniquement les chunks sans tags
    python tag_all_chunks.py --dry-run      # Preview sans modifier
    python tag_all_chunks.py --batch 500    # Limiter à N chunks (pour tester)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import json
import time
import argparse
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'utils'))

import chromadb
from chromadb.config import Settings
from src.utils.llm_provider import RAGConfig
from src.utils.rgpd_topics import TAG_PROMPT, parse_tags, RGPD_CATEGORIES

from tqdm import tqdm

import logging
# Logger silencieux : pas de spam dans le terminal (tqdm gère l'affichage)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# Checkpoint file pour reprendre après crash
CHECKPOINT_FILE = PROJECT_ROOT / 'tasks' / '_tag_checkpoint.json'


def load_checkpoint() -> set:
    """Charge la liste des chunk_ids déjà traités."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return set(data.get('tagged_ids', []))
    return set()


def save_checkpoint(tagged_ids: set):
    """Sauvegarde la liste des chunk_ids traités."""
    with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
        json.dump({'tagged_ids': list(tagged_ids)}, f)


def tag_chunk_text(llm, text: str, max_words: int = 300) -> list[str]:
    """Envoie le texte à Nemo et retourne les tags RGPD.
    
    Tronque le texte à max_words mots pour ne pas surcharger Nemo.
    Retry une fois en cas d'échec.
    """
    # Tronquer
    words = text.split()
    if len(words) > max_words:
        truncated = ' '.join(words[:max_words]) + '...'
    else:
        truncated = text
    
    prompt = TAG_PROMPT.format(text=truncated)
    
    for attempt in range(2):
        try:
            result = llm.generate(prompt, temperature=0.0, max_tokens=100)
            tags = parse_tags(result)
            if tags:
                return tags
            if attempt == 0:
                time.sleep(0.3)
        except Exception as e:
            if attempt == 0:
                time.sleep(0.5)
            else:
                logger.warning(f"Tagging échoué après 2 tentatives: {e}")
    
    return []


def main():
    parser = argparse.ArgumentParser(description="Tag RGPD unifié sur tous les chunks ChromaDB")
    parser.add_argument('--dry-run', action='store_true', help="Preview sans modifier ChromaDB")
    parser.add_argument('--force', action='store_true', help="Re-tagger tous les chunks (même ceux déjà taggés)")
    parser.add_argument('--retry-failed', action='store_true', help="Re-tagger uniquement les chunks sans tags (échecs précédents)")
    parser.add_argument('--batch', type=int, default=0, help="Limiter à N chunks (0 = tous)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("TAG RGPD UNIFIÉ — Tagging LLM de tous les chunks")
    print("=" * 70)
    print(f"Mode : vocabulaire guidé ({len(RGPD_CATEGORIES)} catégories RGPD)")
    
    # --- 1. Connect ChromaDB ---
    chroma_path = PROJECT_ROOT / 'data' / 'vectordb' / 'chromadb'
    client = chromadb.PersistentClient(
        path=str(chroma_path),
        settings=Settings(anonymized_telemetry=False, allow_reset=False)
    )
    collection = client.get_collection("rag_dpo_chunks")
    total = collection.count()
    print(f"\nChromaDB : {total} chunks")
    
    # --- 2. Charger tous les chunks ---
    print("Chargement des chunks...")
    all_data = collection.get(include=["metadatas", "documents"])
    
    # Filtrer selon le mode
    checkpoint = load_checkpoint() if not args.force else set()
    
    to_tag = []
    already_tagged = 0
    for chunk_id, meta, doc in zip(all_data["ids"], all_data["metadatas"], all_data["documents"]):
        existing_tags = meta.get('rgpd_topics', '')
        
        if args.retry_failed:
            # Mode retry : uniquement les chunks SANS tags
            if existing_tags:
                already_tagged += 1
                continue
        elif not args.force:
            # Mode normal : skip les chunks déjà taggés ou dans le checkpoint
            if existing_tags or chunk_id in checkpoint:
                already_tagged += 1
                continue
        
        to_tag.append((chunk_id, meta, doc))
    
    if args.batch > 0:
        to_tag = to_tag[:args.batch]
    
    print(f"Déjà taggés : {already_tagged}")
    print(f"À tagger    : {len(to_tag)}")
    
    if not to_tag:
        print("\n✅ Tous les chunks sont déjà taggés !")
        return
    
    # --- 3. Init LLM ---
    if not args.dry_run:
        print("\nInitialisation LLM Nemo...")
        config = RAGConfig()
        llm = config.llm_provider
    else:
        llm = None
    
    # --- 4. Tagging ---
    tag_counter = Counter()
    tagged_count = 0
    failed_count = 0
    batch_ids = []
    batch_metas = []
    SAVE_EVERY = 100
    
    start_time = time.time()
    
    if args.dry_run:
        print(f"\n[DRY RUN] Aperçu des 5 premiers chunks :")
        for i, (chunk_id, meta, doc) in enumerate(to_tag[:5]):
            preview = (doc or '')[:120].replace('\n', ' ')
            print(f"  [{i}] {chunk_id}: {preview}...")
        print(f"  ... et {len(to_tag) - 5} autres" if len(to_tag) > 5 else "")
        return
    
    # Barre de progression tqdm
    pbar = tqdm(
        to_tag,
        desc="Tagging RGPD",
        unit="chunk",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] ✅{postfix}",
        dynamic_ncols=True,
    )
    
    for chunk_id, meta, doc in pbar:
        # Tagger via LLM
        tags = tag_chunk_text(llm, doc or '')
        
        if tags:
            tag_str = ','.join(tags)
            batch_ids.append(chunk_id)
            new_meta = dict(meta)
            new_meta['rgpd_topics'] = tag_str
            batch_metas.append(new_meta)
            
            for t in tags:
                tag_counter[t] += 1
            tagged_count += 1
        else:
            failed_count += 1
        
        checkpoint.add(chunk_id)
        
        # Update postfix avec stats temps réel
        pbar.set_postfix_str(f"{tagged_count} ok, ❌{failed_count} fail")
        
        # Sauvegarder par batch
        if len(batch_ids) >= SAVE_EVERY:
            collection.update(ids=batch_ids, metadatas=batch_metas)
            save_checkpoint(checkpoint)
            batch_ids = []
            batch_metas = []
    
    pbar.close()
    
    # Sauvegarder le reste
    if batch_ids:
        collection.update(ids=batch_ids, metadatas=batch_metas)
        save_checkpoint(checkpoint)
    
    elapsed = time.time() - start_time
    
    # --- 5. Résumé ---
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ")
    print(f"{'='*70}")
    print(f"Chunks traités : {len(to_tag)}")
    print(f"Taggés OK      : {tagged_count}")
    print(f"Échecs         : {failed_count}")
    print(f"Temps total    : {elapsed:.0f}s ({elapsed/60:.1f}min)")
    if tagged_count:
        print(f"Vitesse        : {len(to_tag)/elapsed:.1f} chunks/s")
    
    # Distribution des tags (top 30 uniquement pour lisibilité)
    if tag_counter:
        print(f"\nTop 30 tags (sur {len(tag_counter)} uniques) :")
        for tag, count in tag_counter.most_common(30):
            pct = count / tagged_count * 100 if tagged_count else 0
            print(f"  {tag:40s} : {count:5d} ({pct:5.1f}%)")
        if len(tag_counter) > 30:
            print(f"  ... et {len(tag_counter) - 30} autres tags")
    
    # Cleanup checkpoint
    if failed_count == 0:
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
            print("\n✅ Checkpoint supprimé (tout OK).")
    else:
        print(f"\n⚠️  {failed_count} chunks en échec. Relancez avec --retry-failed pour les re-tagger.")
    
    print(f"\nTerminé !")


if __name__ == '__main__':
    main()
