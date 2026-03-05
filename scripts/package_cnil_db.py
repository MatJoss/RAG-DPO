"""
package_cnil_db.py — Prépare le zip de la base CNIL pour GitHub Releases.

Script mainteneur : crée le zip contenant la base ChromaDB + métadonnées
pour publication sur GitHub Releases. Les utilisateurs le téléchargeront
via scripts/download_cnil_db.py.

Usage (depuis la racine du projet) :
    python scripts/package_cnil_db.py
    # → Crée rag-dpo-cnil-db.zip (~500-700 MB compressé)

Puis uploader sur GitHub :
    1. Aller sur https://github.com/MatJoss/RAG-DPO/releases
    2. Créer une release (tag: db-YYYY-MM, ex: db-2026-03)
    3. Attacher le fichier rag-dpo-cnil-db.zip
"""
import sys
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
VECTORDB_DIR = PROJECT_ROOT / "data" / "vectordb" / "chromadb"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"
OUTPUT_NAME = "rag-dpo-cnil-db.zip"
OUTPUT_PATH = PROJECT_ROOT / OUTPUT_NAME


def package():
    """Crée le zip de la base CNIL."""
    
    # Vérifications
    sqlite = VECTORDB_DIR / "chroma.sqlite3"
    if not sqlite.exists():
        print(f"❌ Base ChromaDB introuvable : {VECTORDB_DIR}")
        sys.exit(1)
    
    if not METADATA_DIR.exists():
        print(f"❌ Métadonnées introuvables : {METADATA_DIR}")
        sys.exit(1)
    
    print(f"📦 Création du package base CNIL...")
    print(f"   ChromaDB : {VECTORDB_DIR}")
    print(f"   Métadonnées : {METADATA_DIR}")
    print()
    
    n_files = 0
    total_size = 0
    
    with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        
        # ── ChromaDB (vectordb/chromadb/*) ────────────────────────────────
        print(f"   Ajout de ChromaDB...")
        for f in sorted(VECTORDB_DIR.rglob("*")):
            if f.is_file():
                # Chemin relatif dans le zip : vectordb/chromadb/...
                arcname = f"vectordb/chromadb/{f.relative_to(VECTORDB_DIR)}"
                zf.write(f, arcname)
                n_files += 1
                total_size += f.stat().st_size
                if n_files % 10 == 0:
                    sys.stdout.write(f"\r   ... {n_files} fichiers")
                    sys.stdout.flush()
        print(f"\r   ✅ ChromaDB : {n_files} fichiers")
        
        # ── Métadonnées (metadata/*) ──────────────────────────────────────
        print(f"   Ajout des métadonnées...")
        n_meta = 0
        for f in sorted(METADATA_DIR.rglob("*")):
            if f.is_file():
                arcname = f"metadata/{f.relative_to(METADATA_DIR)}"
                zf.write(f, arcname)
                n_files += 1
                n_meta += 1
                total_size += f.stat().st_size
                if n_meta % 1000 == 0:
                    sys.stdout.write(f"\r   ... {n_meta} fichiers metadata")
                    sys.stdout.flush()
        print(f"\r   ✅ Métadonnées : {n_meta} fichiers")
    
    # Stats finales
    zip_size = OUTPUT_PATH.stat().st_size
    ratio = (1 - zip_size / total_size) * 100 if total_size > 0 else 0
    
    print()
    print(f"✅ Package créé : {OUTPUT_NAME}")
    print(f"   Fichiers : {n_files}")
    size_orig = total_size / (1024 * 1024)
    size_zip = zip_size / (1024 * 1024)
    print(f"   Taille originale : {size_orig:.0f} MB")
    print(f"   Taille compressée : {size_zip:.0f} MB ({ratio:.0f}% compression)")
    print(f"   Date : {datetime.now().strftime('%Y-%m-%d')}")
    print()
    print(f"📤 Pour publier :")
    print(f"   1. https://github.com/MatJoss/RAG-DPO/releases/new")
    print(f"   2. Tag : db-{datetime.now().strftime('%Y-%m')}")
    print(f"   3. Titre : Base CNIL {datetime.now().strftime('%B %Y')}")
    print(f"   4. Attacher : {OUTPUT_PATH}")


if __name__ == "__main__":
    package()
