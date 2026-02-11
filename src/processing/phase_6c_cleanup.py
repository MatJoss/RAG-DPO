"""
Phase 6C : Nettoyage post-résumés.

Actions :
1. Identifier les docs navigation (error=navigation_content dans summaries)
   qui restent des vrais nav avec le filtre corrigé.
2. Supprimer les entrées "error" des docs récupérés → 6B les reprendra.
3. Purger les vrais nav de ChromaDB + JSONL.
4. Archiver les fichiers physiques des vrais nav.
5. Mettre à jour summaries.json.

Usage : python src/processing/phase_6c_cleanup.py [--dry-run]
"""
import json
import re
import shutil
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import chromadb
from chromadb import PersistentClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CHROMADB_PATH = PROJECT_ROOT / "data" / "vectordb" / "chromadb"
SUMMARIES_PATH = PROJECT_ROOT / "data" / "keep" / "cnil" / "document_summaries.json"
CHUNKS_PATH = PROJECT_ROOT / "data" / "raw" / "cnil" / "processed_chunks.jsonl"
KEEP_PATH = PROJECT_ROOT / "data" / "keep" / "cnil"
ARCHIVE_PATH = PROJECT_ROOT / "data" / "archive"

# Seuil : le nouveau filtre considère > 2000 chars = jamais nav
NAV_PATTERNS = [
    r'En savoir plus', r'Lire la suite', r'Voir aussi', r'Consulter',
    r'En détail', r'Accéder', r'Découvrir', r'Toutes les actualités',
    r'Rechercher', r'Affiner la recherche', r'\d+ résultat',
    r'Page \d+ sur \d+', r'Suivant|Précédent',
]


def is_still_navigation(content: str, doc_path: str) -> bool:
    """Le nouveau filtre (identique à generate_document_summaries.py corrigé)."""
    if not doc_path.endswith('.html'):
        return False
    if len(content.strip()) < 100:
        return True
    nav_count = sum(len(re.findall(p, content, re.IGNORECASE)) for p in NAV_PATTERNS)
    if len(content.strip()) > 2000:
        return False
    if nav_count >= 3 and len(content.strip()) < 1000:
        return True
    if len(content.strip()) < 500 and nav_count >= 1:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Phase 6C : Nettoyage post-résumés')
    parser.add_argument('--dry-run', action='store_true', help='Afficher les actions sans les exécuter')
    args = parser.parse_args()
    dry_run = args.dry_run

    logger.info("=" * 70)
    logger.info("  PHASE 6C : NETTOYAGE POST-RÉSUMÉS")
    logger.info("=" * 70)
    if dry_run:
        logger.info("  ⚠️  MODE DRY-RUN — aucune modification")

    # 1. Charger summaries
    summaries = json.loads(SUMMARIES_PATH.read_text(encoding="utf-8"))
    nav_entries = {p: v for p, v in summaries.items() if v.get("error") == "navigation_content"}
    logger.info(f"  Résumés total     : {len(summaries)}")
    logger.info(f"  Entrées nav       : {len(nav_entries)}")

    if not nav_entries:
        logger.info("  ✅ Aucune entrée navigation — rien à faire.")
        return

    # 2. Charger chunks pour reconstituer le contenu
    chunks_by_doc = {}
    all_chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            dp = d.get("document_path", "")
            chunks_by_doc.setdefault(dp, []).append(d)
            all_chunks.append(d)

    # 3. Classifier : vrais nav vs récupérés
    true_nav = []  # à purger
    recovered = []  # à résumer

    for doc_path in sorted(nav_entries.keys()):
        chunks = chunks_by_doc.get(doc_path, [])
        content = "\n\n".join(c.get("text", "") for c in chunks)
        if is_still_navigation(content, doc_path):
            true_nav.append(doc_path)
        else:
            recovered.append(doc_path)

    logger.info(f"\n  === Classification ===")
    logger.info(f"  Vrais nav (purger)  : {len(true_nav)}")
    for p in true_nav:
        name = Path(p).name
        logger.info(f"    🔴 {name}")
    logger.info(f"  Récupérés (résumer) : {len(recovered)}")
    for p in recovered:
        name = Path(p).name
        logger.info(f"    🟢 {name}")

    if dry_run:
        logger.info("\n  DRY-RUN terminé. Relancer sans --dry-run pour appliquer.")
        return

    # === ACTIONS ===

    # 4. Récupérés : supprimer l'entrée error de summaries → 6B resume les reprendra
    for doc_path in recovered:
        del summaries[doc_path]
        logger.info(f"  ♻️  Supprimé de summaries (6B reprendra) : {Path(doc_path).name}")

    # 5. Vrais nav : purger de ChromaDB
    logger.info(f"\n  Connexion ChromaDB...")
    client = PersistentClient(path=str(CHROMADB_PATH))
    collection = client.get_collection("rag_dpo_chunks")
    initial_count = collection.count()
    logger.info(f"  ChromaDB initial : {initial_count} chunks")

    total_deleted_chroma = 0
    for doc_path in true_nav:
        results = collection.get(
            where={"document_path": doc_path},
            include=["metadatas"]
        )
        ids_to_delete = results.get("ids", [])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            total_deleted_chroma += len(ids_to_delete)
            logger.info(f"  🗑️  ChromaDB : {len(ids_to_delete)} chunks supprimés pour {Path(doc_path).name}")

    logger.info(f"  ChromaDB final   : {collection.count()} chunks (supprimés: {total_deleted_chroma})")

    # 6. Vrais nav : purger du JSONL
    true_nav_set = set(true_nav)
    new_chunks = [c for c in all_chunks if c.get("document_path", "") not in true_nav_set]
    removed_jsonl = len(all_chunks) - len(new_chunks)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in new_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info(f"  🗑️  JSONL : {removed_jsonl} chunks supprimés, {len(new_chunks)} restants")

    # 7. Vrais nav : mettre à jour summaries (marquer comme cleaned)
    for doc_path in true_nav:
        summaries[doc_path] = {
            "summary": "Page de navigation nettoyée en Phase 6C",
            "cleaned": True,
            "reason": "navigation_content",
            "source_url": nav_entries[doc_path].get("source_url", ""),
            "document_title": nav_entries[doc_path].get("document_title", ""),
        }

    # 8. Archiver les fichiers physiques des vrais nav
    ARCHIVE_PATH.mkdir(parents=True, exist_ok=True)
    archive_html = ARCHIVE_PATH / "html"
    archive_html.mkdir(parents=True, exist_ok=True)

    archived = 0
    for doc_path in true_nav:
        name = Path(doc_path).name
        src = KEEP_PATH / "html" / name
        dst = archive_html / name
        if src.exists():
            shutil.move(str(src), str(dst))
            archived += 1
            logger.info(f"  📦 Archivé : {name}")
        else:
            logger.warning(f"  ⚠️  Fichier absent : {src}")

    logger.info(f"  Archivés : {archived}/{len(true_nav)}")

    # 9. Sauvegarder summaries
    SUMMARIES_PATH.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    # 10. Bilan
    ok_count = sum(1 for v in summaries.values() if v.get("summary") and "error" not in v and not v.get("cleaned"))
    cleaned_count = sum(1 for v in summaries.values() if v.get("cleaned"))
    pending = sum(1 for v in summaries.values() if "error" in v)

    logger.info(f"\n" + "=" * 70)
    logger.info(f"  ✅ PHASE 6C TERMINÉE")
    logger.info(f"  Résumés OK        : {ok_count}")
    logger.info(f"  Nettoyés (nav)    : {cleaned_count}")
    logger.info(f"  Retirés (→ 6B)    : {len(recovered)}")
    logger.info(f"  En erreur restant : {pending}")
    logger.info(f"  ChromaDB chunks   : {collection.count()}")
    logger.info(f"  JSONL chunks      : {len(new_chunks)}")
    logger.info(f"  Fichiers archivés : {archived}")
    logger.info(f"=" * 70)
    logger.info(f"\n  → Relancer 6B pour générer les {len(recovered)} résumés manquants")


if __name__ == "__main__":
    main()
