"""
🔄 Mise à jour incrémentale de la base CNIL

Script one-shot qui enchaîne :
  1. Scraping incrémental (--update : If-Modified-Since → 304)
  2. Re-classification des nouveaux/modifiés
  3. Re-chunking des documents modifiés
  4. Re-indexation ChromaDB (mode update/append)
  5. Régénération des résumés manquants

Usage :
  python update_cnil.py                  # Update complet
  python update_cnil.py --scrape-only    # Seulement le scraping (vérifier les modifs)
  python update_cnil.py --skip-scrape    # Skip scraping, re-process à partir des fichiers déjà scrapés
  python update_cnil.py --dry-run        # Affiche ce qui serait fait sans rien exécuter
  python update_cnil.py --force-reindex  # Force la réindexation complète (mode reset)

Prévu pour tourner ~1x/mois.
"""
import sys
import time
import json
import subprocess
import argparse
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw" / "cnil"
KEEP_DIR = DATA_ROOT / "keep" / "cnil"
METADATA_DIR = DATA_ROOT / "metadata"
VECTORDB_DIR = DATA_ROOT / "vectordb" / "chromadb"
CHUNKS_FILE = RAW_DIR / "processed_chunks.jsonl"
SUMMARIES_FILE = KEEP_DIR / "document_summaries.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("update_cnil")


def run_cmd(label: str, cmd: list, dry_run: bool = False) -> bool:
    """Exécute une commande subprocess avec affichage."""
    print(f"\n{'─' * 70}")
    print(f"  🔧 {label}")
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    print(f"{'─' * 70}")

    if dry_run:
        print("  [DRY RUN] Commande non exécutée")
        return True

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        logger.error(f"❌ {label} — échec (code {result.returncode}) en {elapsed:.0f}s")
        return False
    else:
        logger.info(f"✅ {label} — OK en {elapsed:.0f}s ({elapsed/60:.1f}min)")
        return True


def get_chromadb_count() -> int:
    """Compte les chunks dans ChromaDB."""
    try:
        import chromadb
        from chromadb.config import Settings
        client = chromadb.PersistentClient(
            path=str(VECTORDB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        collection = client.get_collection("rag_dpo_chunks")
        return collection.count()
    except Exception:
        return 0


def count_modified_files() -> dict:
    """Analyse les metadata pour trouver les fichiers récemment modifiés."""
    stats = {'total_metadata': 0, 'with_hash': 0, 'recent_30d': 0}

    if not METADATA_DIR.exists():
        return stats

    cutoff = datetime.now().timestamp() - (30 * 24 * 3600)  # 30 jours

    for meta_file in METADATA_DIR.glob('*.json'):
        stats['total_metadata'] += 1
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get('content_hash'):
                stats['with_hash'] += 1
            scraped_at = meta.get('scraped_at', '')
            if scraped_at:
                try:
                    dt = datetime.fromisoformat(scraped_at)
                    if dt.timestamp() > cutoff:
                        stats['recent_30d'] += 1
                except Exception:
                    pass
        except Exception:
            pass

    return stats


def count_chunks_jsonl() -> int:
    """Compte les chunks dans le JSONL."""
    if not CHUNKS_FILE.exists():
        return 0
    count = 0
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def count_summaries() -> int:
    """Compte les résumés existants."""
    if not SUMMARIES_FILE.exists():
        return 0
    try:
        with open(SUMMARIES_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return sum(1 for v in data.values() if v.get('summary') and 'error' not in v)
    except Exception:
        return 0


def print_state():
    """Affiche l'état actuel de la base."""
    print("\n" + "=" * 70)
    print("  📊 ÉTAT ACTUEL DE LA BASE CNIL")
    print("=" * 70)

    # Metadata
    meta_stats = count_modified_files()
    print(f"  📄 Metadata           : {meta_stats['total_metadata']} documents")
    print(f"     avec content_hash  : {meta_stats['with_hash']}")
    print(f"     modifiés (<30j)    : {meta_stats['recent_30d']}")

    # Keep
    keep_counts = {}
    for subdir in ['html', 'pdf', 'docs']:
        d = KEEP_DIR / subdir
        keep_counts[subdir] = sum(1 for _ in d.iterdir()) if d.exists() else 0
    print(f"  📁 Keep               : {sum(keep_counts.values())} fichiers")
    for k, v in keep_counts.items():
        print(f"     {k:20s}: {v}")

    # Chunks
    n_chunks = count_chunks_jsonl()
    print(f"  📦 Chunks JSONL       : {n_chunks}")

    # ChromaDB
    n_indexed = get_chromadb_count()
    print(f"  🗄️  ChromaDB indexés   : {n_indexed}")

    # Résumés
    n_summaries = count_summaries()
    print(f"  📝 Résumés            : {n_summaries}")

    print("=" * 70)
    return {
        'metadata': meta_stats['total_metadata'],
        'chunks_jsonl': n_chunks,
        'chromadb': n_indexed,
        'summaries': n_summaries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Mise à jour incrémentale de la base CNIL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python update_cnil.py                  # Update complet (~1x/mois)
  python update_cnil.py --scrape-only    # Vérifier les modifications sans traiter
  python update_cnil.py --skip-scrape    # Re-traiter sans re-scraper
  python update_cnil.py --dry-run        # Voir ce qui serait fait
  python update_cnil.py --force-reindex  # Réindexation complète ChromaDB
  python update_cnil.py --status         # Afficher l'état sans rien faire
        """
    )
    parser.add_argument("--scrape-only", action="store_true",
                        help="Seulement le scraping incrémental")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Sauter le scraping, traiter à partir des fichiers existants")
    parser.add_argument("--dry-run", action="store_true",
                        help="Affiche les commandes sans les exécuter")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Force réindexation ChromaDB complète (mode reset)")
    parser.add_argument("--status", action="store_true",
                        help="Affiche l'état de la base sans rien faire")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Pas de confirmation interactive")

    args = parser.parse_args()

    py = sys.executable

    print("=" * 70)
    print("  🔄 MISE À JOUR INCRÉMENTALE — BASE CNIL")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # État avant
    state_before = print_state()

    if args.status:
        return

    # ── Plan d'exécution ──
    steps = []

    if not args.skip_scrape:
        steps.append(("1. Scraping incrémental CNIL", [
            py, str(PROJECT_ROOT / "src" / "scraping" / "cnil_scraper_final.py"),
            "--update",
            "--project-root", str(PROJECT_ROOT),
        ]))

    if not args.scrape_only:
        # Phase 3 : Re-classification (avec résultats existants = resume auto)
        steps.append(("2. Classification hybride (incrémental)", [
            py, str(PROJECT_ROOT / "src" / "processing" / "hybrid_filter.py"),
        ]))

        # Phase 4 : Organisation keep/archive
        steps.append(("3. Organisation keep/archive", [
            py, str(PROJECT_ROOT / "src" / "processing" / "organize_keep_archive.py"),
            "--execute", "--yes", "--clean",
        ]))

        # Phase 4c : Déduplication
        steps.append(("4. Déduplication corpus", [
            py, str(PROJECT_ROOT / "src" / "processing" / "deduplicate_corpus.py"),
        ]))

        # Phase 5a : Pré-catégorisation (les scripts skipent les docs déjà classifiés)
        steps.append(("5. Pré-catégorisation documents", [
            py, str(PROJECT_ROOT / "src" / "processing" / "classify_documents.py"),
        ]))

        # Phase 5b : Chunking (re-process tout car on ne sait pas quels docs ont changé)
        steps.append(("6. Chunking sémantique", [
            py, str(PROJECT_ROOT / "src" / "processing" / "process_and_chunk.py"),
        ]))

        # Phase 6a : Indexation ChromaDB
        index_mode = "reset" if args.force_reindex else "update"
        steps.append(("7. Indexation ChromaDB", [
            py, str(PROJECT_ROOT / "src" / "processing" / "create_chromadb_index.py"),
            "--mode", index_mode,
        ]))

        # Phase 6b : Résumés (skip les docs déjà résumés)
        steps.append(("8. Génération résumés", [
            py, str(PROJECT_ROOT / "src" / "processing" / "generate_document_summaries.py"),
        ]))

        # Phase 6d : Tagging RGPD des nouveaux chunks (idempotent, skip déjà taggés)
        steps.append(("8b. Tagging RGPD des chunks", [
            py, str(PROJECT_ROOT / "tag_all_chunks.py"),
        ]))

        # Phase 6c : Nettoyage post-résumés
        steps.append(("9. Nettoyage post-résumés", [
            py, str(PROJECT_ROOT / "src" / "processing" / "phase_6c_cleanup.py"),
        ]))

    # ── Affichage plan ──
    print(f"\n📋 Plan d'exécution ({len(steps)} étapes) :")
    for i, (label, cmd) in enumerate(steps):
        print(f"  [{i+1}/{len(steps)}] {label}")

    if not args.yes and not args.dry_run:
        resp = input(f"\n  Lancer la mise à jour ? (O/n) : ").strip().lower()
        if resp == 'n':
            print("  Annulé.")
            return

    # ── Exécution ──
    t_total = time.time()
    results = []

    for label, cmd in steps:
        ok = run_cmd(label, cmd, dry_run=args.dry_run)
        results.append((label, ok))

        if not ok:
            logger.error(f"\n⚠️  Échec à l'étape : {label}")
            if not args.yes:
                resp = input("  Continuer malgré l'erreur ? (o/N) : ").strip().lower()
                if resp != 'o':
                    break

    elapsed_total = time.time() - t_total

    # ── Bilan ──
    print("\n" + "=" * 70)
    print("  📊 BILAN DE LA MISE À JOUR")
    print("=" * 70)

    for label, ok in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {label}")

    if not args.dry_run:
        # État après
        state_after = print_state()

        # Diff
        print("\n📈 Changements :")
        for key in ['metadata', 'chunks_jsonl', 'chromadb', 'summaries']:
            before = state_before.get(key, 0)
            after = state_after.get(key, 0)
            diff = after - before
            if diff != 0:
                sign = "+" if diff > 0 else ""
                print(f"  {key:20s}: {before} → {after} ({sign}{diff})")
            else:
                print(f"  {key:20s}: {after} (inchangé)")

    n_ok = sum(1 for _, ok in results if ok)
    n_total = len(results)
    print(f"\n⏱️  Durée totale : {elapsed_total/60:.1f} min ({elapsed_total/3600:.1f}h)")
    print(f"📋 Résultat : {n_ok}/{n_total} étapes réussies")

    if all(ok for _, ok in results):
        print("\n🎉 Mise à jour terminée avec succès !")
        print(f"   Prochaine update recommandée : ~{(datetime.now().replace(month=datetime.now().month % 12 + 1, day=1)).strftime('%Y-%m-%d')}")
    else:
        print("\n⚠️  Mise à jour terminée avec des erreurs.")
        print("   Consultez les logs ci-dessus pour diagnostiquer.")

    print("=" * 70)


if __name__ == "__main__":
    main()
