"""
🔄 Phase 4C : Déduplication du corpus
Élimine les documents dont le contenu est identique à un autre.

Stratégie :
  - HTML  : hash MD5 du texte region-content (contenu utile)
  - PDF   : hash MD5 du fichier binaire
  - Docs  : hash MD5 du fichier binaire
  - Images: hash MD5 du fichier binaire

Pour chaque groupe de doublons, on garde 1 "canonical" et on retire les autres.
Critère canonical : URL la plus courte en https, sinon la première.

Le manifest est mis à jour (doublons retirés).
Les fichiers doublons sont déplacés dans keep/dedup_archive/ (pas supprimés).
Un rapport JSON est sauvegardé pour traçabilité.

Usage :
  python src/processing/deduplicate_corpus.py              # Exécuter
  python src/processing/deduplicate_corpus.py --dry-run    # Simuler
  python src/processing/deduplicate_corpus.py --fresh      # Ignorer cache
"""

import json
import hashlib
import shutil
import logging
import argparse
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

PROJECT_ROOT = Path(__file__).parent.parent.parent
CNIL_PATH = PROJECT_ROOT / "data" / "raw" / "cnil"
KEEP_PATH = PROJECT_ROOT / "data" / "keep" / "cnil"


class CorpusDeduplicator:
    """Déduplique le corpus en éliminant les contenus identiques."""

    def __init__(self):
        self.manifest_file = CNIL_PATH / "keep_manifest.json"
        self.report_file = CNIL_PATH / "dedup_report.json"
        self.archive_dir = KEEP_PATH / "dedup_archive"

        self.stats = {
            'html_before': 0, 'html_after': 0, 'html_removed': 0,
            'pdf_before': 0, 'pdf_after': 0, 'pdf_removed': 0,
            'docs_before': 0, 'docs_after': 0, 'docs_removed': 0,
            'images_before': 0, 'images_after': 0, 'images_removed': 0,
            'relationships_cleaned': 0,
        }
        self.report = {
            'timestamp': '',
            'groups': [],   # groupes de doublons détectés
            'removed': [],  # fichiers retirés
            'stats': {},
        }

    # ──────────────────────── Hashing ────────────────────────

    def _hash_html_content(self, file_path: Path) -> Optional[str]:
        """Hash le texte region-content d'un HTML (contenu utile uniquement)."""
        try:
            html = file_path.read_text(encoding='utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'html.parser')
            region = soup.find(class_='region-content')
            if region:
                text = region.get_text(separator=' ', strip=True)
                if len(text) >= 20:
                    return hashlib.md5(text.encode('utf-8')).hexdigest()
            # Fallback : hash du body entier si pas de region-content
            body = soup.find('body')
            if body:
                text = body.get_text(separator=' ', strip=True)
                if len(text) >= 20:
                    return hashlib.md5(text.encode('utf-8')).hexdigest()
            return None
        except Exception as e:
            logger.warning(f"Erreur hash HTML {file_path.name}: {e}")
            return None

    def _hash_binary(self, file_path: Path) -> Optional[str]:
        """Hash MD5 d'un fichier binaire."""
        try:
            return hashlib.md5(file_path.read_bytes()).hexdigest()
        except Exception as e:
            logger.warning(f"Erreur hash binaire {file_path.name}: {e}")
            return None

    # ──────────────────────── Canonical selection ────────────────────────

    def _pick_canonical(self, items: List[Dict]) -> Tuple[Dict, List[Dict]]:
        """Choisit le document canonical dans un groupe de doublons.
        
        Critères (par ordre) :
        1. URL https > http
        2. URL la plus courte (souvent la plus propre)
        3. Premier dans la liste (stable)
        """
        def score(item):
            url = item.get('url', '')
            is_https = 1 if url.startswith('https://') else 0
            # Pénaliser les URLs avec ?page=, les tags en doublon, etc.
            has_params = 1 if '?' in url else 0
            return (-is_https, has_params, len(url))

        sorted_items = sorted(items, key=score)
        canonical = sorted_items[0]
        duplicates = sorted_items[1:]
        return canonical, duplicates

    # ──────────────────────── Dédup par type ────────────────────────

    def _dedup_list(self, items: List[Dict], item_type: str,
                    hash_fn) -> Tuple[List[Dict], List[Dict]]:
        """Déduplique une liste de documents du manifest.
        
        Returns:
            (kept, removed) — listes de documents gardés et retirés
        """
        # Grouper par hash contenu
        groups = defaultdict(list)
        no_hash = []

        for item in items:
            fp = PROJECT_ROOT / item.get('metadata', {}).get('file_path', '')
            if fp.exists():
                h = hash_fn(fp)
                if h:
                    groups[h].append(item)
                else:
                    no_hash.append(item)
            else:
                no_hash.append(item)

        kept = []
        removed = []

        for h, group_items in groups.items():
            canonical, duplicates = self._pick_canonical(group_items)
            kept.append(canonical)

            if duplicates:
                # Enregistrer le groupe pour le rapport
                self.report['groups'].append({
                    'type': item_type,
                    'hash': h,
                    'count': len(group_items),
                    'canonical': {
                        'file': canonical.get('file', ''),
                        'url': canonical.get('url', ''),
                    },
                    'duplicates': [
                        {'file': d.get('file', ''), 'url': d.get('url', '')}
                        for d in duplicates
                    ],
                })

                for d in duplicates:
                    removed.append(d)
                    self.report['removed'].append({
                        'type': item_type,
                        'file': d.get('file', ''),
                        'url': d.get('url', ''),
                        'canonical_url': canonical.get('url', ''),
                    })

        # Les docs sans hash sont gardés (prudence)
        kept.extend(no_hash)

        return kept, removed

    # ──────────────────────── Nettoyage relationships ────────────────────────

    def _clean_relationships(self, relationships: Dict,
                             kept_html_urls: set) -> Dict:
        """Nettoie les relationships pour ne garder que les HTML canonical."""
        cleaned = {}
        removed_count = 0
        for url, resources in relationships.items():
            if url in kept_html_urls:
                cleaned[url] = resources
            else:
                removed_count += 1
        self.stats['relationships_cleaned'] = removed_count
        return cleaned

    # ──────────────────────── Archivage fichiers ────────────────────────

    def _archive_removed_files(self, removed: List[Dict], dry_run: bool):
        """Déplace les fichiers doublons dans dedup_archive/."""
        if dry_run:
            return

        self.archive_dir.mkdir(parents=True, exist_ok=True)

        for item in removed:
            rel_file = item.get('file', '')
            if not rel_file:
                continue

            src = KEEP_PATH / rel_file
            if src.exists():
                # Garder la structure de sous-dossiers
                dest_dir = self.archive_dir / Path(rel_file).parent
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / src.name

                try:
                    shutil.move(str(src), str(dest))
                except Exception as e:
                    logger.warning(f"Impossible de déplacer {src.name}: {e}")

            # Aussi déplacer le metadata associé
            stem = Path(rel_file).stem
            meta_src = KEEP_PATH / 'metadata' / f"{stem}.json"
            if meta_src.exists():
                meta_dest_dir = self.archive_dir / 'metadata'
                meta_dest_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.move(str(meta_src), str(meta_dest_dir / meta_src.name))
                except Exception:
                    pass

    # ──────────────────────── Run principal ────────────────────────

    def run(self, dry_run: bool = False, fresh: bool = False):
        """Exécute la déduplication complète."""
        start = time.time()

        print("=" * 70)
        print("🧹 PHASE 4C : DÉDUPLICATION DU CORPUS")
        print("=" * 70)

        if dry_run:
            print("   MODE SIMULATION — aucune modification")

        # Charger manifest
        if not self.manifest_file.exists():
            print("\n❌ keep_manifest.json introuvable")
            return

        with open(self.manifest_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)

        # Rapport existant ?
        if not fresh and self.report_file.exists():
            print("\n⚠️  dedup_report.json existe déjà.")
            print("   Utilisez --fresh pour re-dédupliquer (depuis le backup manifest).")
            # Afficher stats du rapport existant
            with open(self.report_file, 'r', encoding='utf-8') as f:
                old_report = json.load(f)
            old_stats = old_report.get('stats', {})
            for k, v in old_stats.items():
                print(f"   {k}: {v}")
            return

        # En mode --fresh, restaurer le backup du manifest si disponible
        backup = self.manifest_file.with_suffix('.json.pre_dedup')
        if fresh and backup.exists():
            shutil.copy2(str(backup), str(self.manifest_file))
            print(f"\n♻️  Manifest restauré depuis backup ({backup.name})")
            with open(self.manifest_file, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

        # ── Stats avant ──
        self.stats['html_before'] = len(manifest.get('html', []))
        self.stats['pdf_before'] = len(manifest.get('pdfs', []))
        self.stats['docs_before'] = len(manifest.get('docs', []))
        self.stats['images_before'] = len(manifest.get('images', []))
        total_before = sum(self.stats[f'{t}_before'] for t in ['html', 'pdf', 'docs', 'images'])

        print(f"\n📊 Corpus avant déduplication :")
        print(f"   HTML   : {self.stats['html_before']}")
        print(f"   PDFs   : {self.stats['pdf_before']}")
        print(f"   Docs   : {self.stats['docs_before']}")
        print(f"   Images : {self.stats['images_before']}")
        print(f"   TOTAL  : {total_before}")

        # ── Dédup HTML (par region-content) ──
        print("\n📄 Déduplication HTML (hash region-content)...")
        html_kept, html_removed = self._dedup_list(
            manifest.get('html', []), 'html', self._hash_html_content
        )
        self.stats['html_after'] = len(html_kept)
        self.stats['html_removed'] = len(html_removed)
        print(f"   {self.stats['html_before']} → {self.stats['html_after']} "
              f"(-{self.stats['html_removed']})")

        # ── Dédup PDFs (hash binaire) ──
        print("\n📑 Déduplication PDFs (hash binaire)...")
        pdf_kept, pdf_removed = self._dedup_list(
            manifest.get('pdfs', []), 'pdf', self._hash_binary
        )
        self.stats['pdf_after'] = len(pdf_kept)
        self.stats['pdf_removed'] = len(pdf_removed)
        print(f"   {self.stats['pdf_before']} → {self.stats['pdf_after']} "
              f"(-{self.stats['pdf_removed']})")

        # ── Dédup Docs (hash binaire) ──
        print("\n📝 Déduplication Docs (hash binaire)...")
        docs_kept, docs_removed = self._dedup_list(
            manifest.get('docs', []), 'doc', self._hash_binary
        )
        self.stats['docs_after'] = len(docs_kept)
        self.stats['docs_removed'] = len(docs_removed)
        print(f"   {self.stats['docs_before']} → {self.stats['docs_after']} "
              f"(-{self.stats['docs_removed']})")

        # ── Dédup Images (hash binaire) ──
        print("\n🖼️  Déduplication Images (hash binaire)...")
        images_kept, images_removed = self._dedup_list(
            manifest.get('images', []), 'image', self._hash_binary
        )
        self.stats['images_after'] = len(images_kept)
        self.stats['images_removed'] = len(images_removed)
        print(f"   {self.stats['images_before']} → {self.stats['images_after']} "
              f"(-{self.stats['images_removed']})")

        # ── Nettoyer relationships ──
        kept_html_urls = {item['url'] for item in html_kept}
        relationships = self._clean_relationships(
            manifest.get('relationships', {}), kept_html_urls
        )

        # ── Bilan ──
        total_after = (self.stats['html_after'] + self.stats['pdf_after'] +
                       self.stats['docs_after'] + self.stats['images_after'])
        total_removed = total_before - total_after

        print("\n" + "=" * 70)
        print("📊 BILAN DÉDUPLICATION")
        print("=" * 70)
        print(f"""
           AVANT      APRÈS      RETIRÉS
  HTML   : {self.stats['html_before']:>5}      {self.stats['html_after']:>5}      -{self.stats['html_removed']}
  PDF    : {self.stats['pdf_before']:>5}      {self.stats['pdf_after']:>5}      -{self.stats['pdf_removed']}
  Docs   : {self.stats['docs_before']:>5}      {self.stats['docs_after']:>5}      -{self.stats['docs_removed']}
  Images : {self.stats['images_before']:>5}      {self.stats['images_after']:>5}      -{self.stats['images_removed']}
  ─────────────────────────────────────────
  TOTAL  : {total_before:>5}      {total_after:>5}      -{total_removed} ({total_removed/max(1,total_before)*100:.0f}%)
  Relations nettoyées : {self.stats['relationships_cleaned']}
""")

        if dry_run:
            print("   MODE SIMULATION — rien n'a été modifié")
            return

        # ── Sauvegarder ──

        # 1. Archiver fichiers doublons
        all_removed = html_removed + pdf_removed + docs_removed + images_removed
        print(f"📦 Archivage de {len(all_removed)} fichiers dans keep/dedup_archive/...")
        self._archive_removed_files(all_removed, dry_run)

        # 2. Mettre à jour manifest
        new_manifest = {
            'html': html_kept,
            'pdfs': pdf_kept,
            'docs': docs_kept,
            'images': images_kept,
            'relationships': relationships,
        }

        # Backup du manifest original (seulement si pas déjà existant)
        backup = self.manifest_file.with_suffix('.json.pre_dedup')
        if not backup.exists():
            shutil.copy2(str(self.manifest_file), str(backup))
            print(f"💾 Backup manifest : {backup.name}")
        else:
            print(f"💾 Backup manifest existant conservé : {backup.name}")

        with open(self.manifest_file, 'w', encoding='utf-8') as f:
            json.dump(new_manifest, f, indent=2, ensure_ascii=False)
        print(f"💾 Manifest mis à jour : {self.manifest_file.name}")

        # 3. Sauvegarder rapport
        self.report['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.report['stats'] = self.stats
        with open(self.report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        print(f"💾 Rapport sauvé : {self.report_file.name}")

        elapsed = time.time() - start
        print(f"\n✅ Déduplication terminée en {elapsed:.1f}s")
        print(f"   Le corpus est prêt pour Phase 5A → 6B")


def main():
    parser = argparse.ArgumentParser(description='Phase 4C : Déduplication corpus')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simuler sans modifier les fichiers')
    parser.add_argument('--fresh', action='store_true',
                        help='Ignorer le rapport existant, re-dédupliquer')
    args = parser.parse_args()

    dedup = CorpusDeduplicator()
    dedup.run(dry_run=args.dry_run, fresh=args.fresh)


if __name__ == "__main__":
    main()
