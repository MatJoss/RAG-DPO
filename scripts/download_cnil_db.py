"""
download_cnil_db.py — Télécharge la base ChromaDB CNIL pré-construite.

Télécharge le zip depuis GitHub Releases et l'extrait dans data/vectordb/
pour que l'app RAG-DPO soit fonctionnelle immédiatement sans passer par
les phases de scraping et d'indexation.

Usage :
    python scripts/download_cnil_db.py              # Télécharge la dernière version
    python scripts/download_cnil_db.py --force       # Re-télécharge même si la base existe
    python scripts/download_cnil_db.py --check        # Vérifie seulement si la base existe
"""
import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

# ── Configuration ────────────────────────────────────────────────────────────
GITHUB_REPO = "MatJoss/RAG-DPO"
RELEASE_TAG = "latest"  # ou un tag spécifique comme "db-2026-03"
ASSET_NAME = "rag-dpo-cnil-db.zip"

# Chemins
PROJECT_ROOT = Path(__file__).parent.parent
VECTORDB_DIR = PROJECT_ROOT / "data" / "vectordb" / "chromadb"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

# URL de téléchargement (GitHub Releases)
DOWNLOAD_URL = f"https://github.com/{GITHUB_REPO}/releases/{RELEASE_TAG}/download/{ASSET_NAME}"


def check_existing_db() -> bool:
    """Vérifie si une base ChromaDB existe déjà."""
    sqlite_file = VECTORDB_DIR / "chroma.sqlite3"
    return sqlite_file.exists() and sqlite_file.stat().st_size > 1_000_000  # > 1 MB


def get_db_info() -> dict:
    """Retourne des infos sur la base existante."""
    sqlite_file = VECTORDB_DIR / "chroma.sqlite3"
    if not sqlite_file.exists():
        return {"exists": False}
    
    size_mb = sqlite_file.stat().st_size / (1024 * 1024)
    
    # Compter les segments
    n_segments = 0
    if VECTORDB_DIR.exists():
        n_segments = sum(1 for d in VECTORDB_DIR.iterdir() if d.is_dir())
    
    # Compter les métadonnées
    n_metadata = 0
    if METADATA_DIR.exists():
        n_metadata = sum(1 for f in METADATA_DIR.glob("*.json"))
    
    return {
        "exists": True,
        "sqlite_size_mb": round(size_mb, 1),
        "n_segments": n_segments,
        "n_metadata": n_metadata,
    }


def download_with_progress(url: str, dest: Path) -> None:
    """Télécharge un fichier avec une barre de progression."""
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            total_mb = total_size / (1024 * 1024)
            downloaded_mb = downloaded / (1024 * 1024)
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}% ({downloaded_mb:.0f}/{total_mb:.0f} MB)"
            )
            sys.stdout.flush()
    
    urlretrieve(url, str(dest), reporthook=reporthook)
    print()  # Newline après la barre


def download_and_extract(force: bool = False) -> bool:
    """
    Télécharge la base CNIL depuis GitHub Releases et l'extrait.
    
    Returns:
        True si téléchargement réussi, False sinon.
    """
    # Vérifier si la base existe déjà
    if check_existing_db() and not force:
        info = get_db_info()
        print(f"✅ Base ChromaDB déjà présente :")
        print(f"   SQLite : {info['sqlite_size_mb']} MB")
        print(f"   Segments : {info['n_segments']}")
        print(f"   Métadonnées : {info['n_metadata']} fichiers")
        print(f"   Utilisez --force pour re-télécharger.")
        return True
    
    print(f"📥 Téléchargement de la base CNIL pré-construite...")
    print(f"   Source : {DOWNLOAD_URL}")
    print()
    
    # Télécharger dans un fichier temporaire
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / ASSET_NAME
        
        try:
            download_with_progress(DOWNLOAD_URL, zip_path)
        except URLError as e:
            print(f"\n❌ Erreur de téléchargement : {e}")
            print(f"   Vérifiez votre connexion ou l'URL :")
            print(f"   {DOWNLOAD_URL}")
            print(f"\n   Alternative : téléchargez manuellement depuis")
            print(f"   https://github.com/{GITHUB_REPO}/releases")
            print(f"   et extrayez dans data/")
            return False
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            return False
        
        # Vérifier que c'est un zip valide
        if not zipfile.is_zipfile(zip_path):
            print(f"❌ Le fichier téléchargé n'est pas un zip valide.")
            return False
        
        # Extraire
        print(f"📦 Extraction...")
        
        # Backup si existant et --force
        if VECTORDB_DIR.exists() and force:
            backup = VECTORDB_DIR.parent / "chromadb_backup"
            if backup.exists():
                shutil.rmtree(backup)
            VECTORDB_DIR.rename(backup)
            print(f"   Backup de l'ancienne base dans {backup.name}/")
        
        # Créer les dossiers parents
        VECTORDB_DIR.parent.mkdir(parents=True, exist_ok=True)
        METADATA_DIR.parent.mkdir(parents=True, exist_ok=True)
        
        # Extraire le zip dans data/
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Le zip contient vectordb/chromadb/* et metadata/*
            zf.extractall(str(PROJECT_ROOT / "data"))
            n_files = len(zf.namelist())
        
        print(f"   ✅ {n_files} fichiers extraits")
    
    # Vérification post-extraction
    if check_existing_db():
        info = get_db_info()
        print(f"\n✅ Base CNIL installée avec succès :")
        print(f"   SQLite : {info['sqlite_size_mb']} MB")
        print(f"   Segments : {info['n_segments']}")
        print(f"   Métadonnées : {info['n_metadata']} fichiers")
        print(f"\n🚀 L'app est prête : streamlit run app.py")
        return True
    else:
        print(f"\n❌ Extraction échouée — chroma.sqlite3 introuvable")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Télécharge la base ChromaDB CNIL pré-construite depuis GitHub Releases"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-télécharger même si la base existe déjà"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Vérifier seulement si la base existe (pas de téléchargement)"
    )
    args = parser.parse_args()
    
    if args.check:
        info = get_db_info()
        if info["exists"]:
            print(f"✅ Base présente : {info['sqlite_size_mb']} MB, "
                  f"{info['n_segments']} segments, "
                  f"{info['n_metadata']} métadonnées")
            sys.exit(0)
        else:
            print(f"❌ Base absente. Lancez : python scripts/download_cnil_db.py")
            sys.exit(1)
    
    success = download_and_extract(force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
