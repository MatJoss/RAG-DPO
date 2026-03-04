"""
🔄 Script de reconstruction du pipeline RAG-DPO
Relance les étapes de traitement avec Mistral Nemo 12B + extraction region-content.

Étapes disponibles :
  3  : Classification hybride Keywords + LLM (hybrid_filter.py)
  4  : Organisation keep/archive (organize_keep_archive.py)
  5a : Pré-catégorisation documents (classify_documents.py)
  5b : Chunking + classification chunk-level (process_and_chunk.py)
  6a : Indexation ChromaDB (create_chromadb_index.py)
  6b : Génération résumés structurés (generate_document_summaries.py)

Usage :
  python rebuild_pipeline.py              # Tout relancer (3 → 6b)
  python rebuild_pipeline.py --from 5b    # Reprendre depuis 5b
  python rebuild_pipeline.py --only 6b    # Seulement 6b
  python rebuild_pipeline.py --steps 5a 5b 6a  # Seulement ces étapes
  python rebuild_pipeline.py --check      # Vérifier l'état du pipeline
  python rebuild_pipeline.py --test 10    # Mode test : N documents max
  python rebuild_pipeline.py --fresh      # Ignorer résultats existants
"""

import subprocess
import sys
import time
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# Définition ordonnée de toutes les étapes
STEPS = {
    '3':  {
        'name': 'Phase 3 : Classification hybride Keywords + LLM (Nemo)',
        'script': 'src/processing/hybrid_filter.py',
        'args': [],
        'description': 'Classe les ~8000 HTML bruts en keep/exclude via LLM',
    },
    '4':  {
        'name': 'Phase 4 : Organisation keep/archive',
        'script': 'src/processing/organize_keep_archive.py',
        'args': ['--execute', '--yes', '--clean'],
        'description': 'Copie les fichiers pertinents dans data/keep/',
    },
    '4b': {
        'name': 'Phase 4B : Classification images (OCR + LLaVA)',
        'script': 'src/processing/classify_images.py',
        'args': [],
        'description': 'Tesseract OCR + LLaVA vision → trie schémas DPO vs photos déco',
    },
    '4c': {
        'name': 'Phase 4C : Déduplication corpus',
        'script': 'src/processing/deduplicate_corpus.py',
        'args': [],
        'description': 'Élimine 51% de doublons (mêmes contenus sous URLs différentes)',
    },
    '5a': {
        'name': 'Phase 5A : Pré-catégorisation documents (Nemo)',
        'script': 'src/processing/classify_documents.py',
        'args': [],
        'description': 'Classifie nature juridique: DOCTRINE/GUIDE/SANCTION/TECHNIQUE',
    },
    '5b': {
        'name': 'Phase 5B : Chunking + Classification chunk-level',
        'script': 'src/processing/process_and_chunk.py',
        'args': [],
        'description': 'Découpe en chunks structurels + classification heuristique/LLM',
    },
    '6a': {
        'name': 'Phase 6A : Indexation ChromaDB (reset)',
        'script': 'src/processing/create_chromadb_index.py',
        'args': ['--mode', 'reset'],
        'description': 'Génère embeddings BGE-M3 et indexe dans ChromaDB',
    },
    '6b': {
        'name': 'Phase 6B : Génération résumés structurés (Nemo)',
        'script': 'src/processing/generate_document_summaries.py',
        'args': [],
        'description': 'Génère fiches synthétiques par document pour recherche hiérarchique',
    },
    '6c': {
        'name': 'Phase 6C : Nettoyage post-résumés',
        'script': 'src/processing/phase_6c_cleanup.py',
        'args': [],
        'description': 'Purge pages navigation, archive fichiers, nettoie ChromaDB/JSONL',
    },
    '6d': {
        'name': 'Phase 6D : Tagging RGPD des chunks (Nemo)',
        'script': 'tag_all_chunks.py',
        'args': [],
        'description': 'Ajoute des tags RGPD libres à tous les chunks via LLM (idempotent)',
    },
}

STEP_ORDER = ['3', '4', '4b', '4c', '5a', '5b', '6a', '6b', '6c', '6d']


def run_step(step_id: str, test_mode: int = None, fresh: bool = False):
    """Exécute une étape du pipeline."""
    step = STEPS[step_id]
    
    print("\n" + "=" * 70)
    print(f"  {step['name']}")
    print(f"  {step['description']}")
    print("=" * 70)
    
    cmd = [sys.executable, str(PROJECT_ROOT / step['script'])]
    cmd.extend(step['args'])
    
    # Mode test : ajouter --test N si supporté
    if test_mode:
        cmd.extend(['--test', str(test_mode)])
    
    # Mode fresh : passer --fresh aux scripts qui le supportent
    if fresh and step_id in ('3', '4b', '4c', '5a', '5b'):
        cmd.append('--fresh')
    
    start = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"\n  ECHEC: {step['name']} (code {result.returncode})")
        print(f"  Temps: {elapsed:.0f}s")
        
        resp = input("\n  Continuer malgré l'erreur ? (o/N) : ").strip().lower()
        if resp != 'o':
            sys.exit(1)
    else:
        print(f"\n  OK: {step['name']} en {elapsed:.0f}s ({elapsed/60:.1f} min)")
    
    return elapsed


def sanity_check(step_id: str, test_mode: int = None) -> bool:
    """Vérifie la cohérence des données après une phase.
    
    Retourne True si tout est OK, False si des problèmes détectés.
    """
    print(f"\n🔍 Sanity check post-phase {step_id}...")
    
    issues = []
    warnings = []
    
    if step_id == '3':
        # Phase 3 : hybrid_classification.json doit exister et être cohérent
        hc_file = PROJECT_ROOT / "data" / "raw" / "cnil" / "hybrid_classification.json"
        if not hc_file.exists():
            issues.append("hybrid_classification.json MANQUANT")
        else:
            try:
                with open(hc_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                llm_count = len(data.get('llm_classified', {}))
                obvious_count = len(data.get('excluded_obvious', []))
                keyword_count = len(data.get('excluded_keywords', []))
                total_processed = llm_count + obvious_count + keyword_count
                
                # Compter les erreurs LLM
                error_count = sum(1 for v in data.get('llm_classified', {}).values() 
                                 if v.get('error') or 'Erreur' in v.get('raison', ''))
                
                # Compter les pertinents
                kept_count = sum(1 for v in data.get('llm_classified', {}).values() 
                                if v.get('pertinent', False))
                
                print(f"   📊 Total traités    : {total_processed}")
                print(f"   📊 Classifiés LLM   : {llm_count}")
                print(f"   📊 Exclus URL       : {obvious_count}")
                print(f"   📊 Exclus keywords  : {keyword_count}")
                print(f"   📊 Gardés (pertinents): {kept_count}")
                
                if error_count > 0:
                    error_pct = error_count / max(1, llm_count) * 100
                    if error_pct > 10:
                        issues.append(f"{error_count} erreurs LLM ({error_pct:.0f}%) — taux trop élevé")
                    else:
                        warnings.append(f"{error_count} erreurs LLM ({error_pct:.1f}%)")
                
                if not test_mode:
                    raw_html_count = sum(1 for _ in (PROJECT_ROOT / "data" / "raw" / "html").glob('*.html'))
                    if total_processed < raw_html_count * 0.9:
                        warnings.append(f"Seulement {total_processed}/{raw_html_count} HTML traités — possible interruption")
                
                if kept_count == 0:
                    issues.append("Aucun document gardé ! Problème de classification ?")
                
            except json.JSONDecodeError as e:
                issues.append(f"JSON invalide: {e}")
    
    elif step_id == '4':
        # Phase 4 : les fichiers keep doivent exister
        keep_html = PROJECT_ROOT / "data" / "keep" / "cnil" / "html"
        keep_pdf = PROJECT_ROOT / "data" / "keep" / "cnil" / "pdf"
        keep_docs = PROJECT_ROOT / "data" / "keep" / "cnil" / "docs"
        
        html_count = sum(1 for _ in keep_html.glob('*.html')) if keep_html.exists() else 0
        pdf_count = sum(1 for _ in keep_pdf.glob('*.pdf')) if keep_pdf.exists() else 0
        doc_count = sum(1 for _ in keep_docs.iterdir()) if keep_docs.exists() else 0
        
        print(f"   📊 HTML keep : {html_count}")
        print(f"   📊 PDF keep  : {pdf_count}")
        print(f"   📊 Docs keep : {doc_count}")
        
        if html_count == 0:
            issues.append("Aucun HTML dans keep/ — Phase 3 ou 4 a échoué ?")
        
        # Vérifier que keep_manifest existe
        manifest = PROJECT_ROOT / "data" / "raw" / "cnil" / "keep_manifest.json"
        if not manifest.exists():
            issues.append("keep_manifest.json MANQUANT")
    
    elif step_id == '4b':
        # Phase 4B : image_classification.json doit exister
        ic_file = PROJECT_ROOT / "data" / "raw" / "cnil" / "image_classification.json"
        keep_images = PROJECT_ROOT / "data" / "keep" / "cnil" / "images"
        
        if ic_file.exists():
            with open(ic_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            stats = data.get('stats', {})
            total = stats.get('total', 0)
            schema = stats.get('schema_dpo', 0)
            infog = stats.get('infographie', 0)
            deco = stats.get('photo_deco', 0)
            removed = stats.get('removed_from_keep', 0)
            
            print(f"   📊 Images totales     : {total}")
            print(f"   📊 Schémas DPO        : {schema}")
            print(f"   📊 Infographies       : {infog}")
            print(f"   📊 Photos/Déco        : {deco} (retirées)")
            print(f"   📊 Retirées de keep/  : {removed}")
            
            # Vérification cohérence
            img_count = sum(1 for _ in keep_images.glob('*')) if keep_images.exists() else 0
            print(f"   📊 Images restantes   : {img_count}")
            
            if schema + infog == 0:
                warnings.append("Aucun schéma/infographie gardé — seuils trop stricts ?")
        else:
            issues.append("image_classification.json MANQUANT")
    
    elif step_id == '4c':
        # Phase 4C : déduplication — rapport doit exister, manifest mis à jour
        report_file = PROJECT_ROOT / "data" / "raw" / "cnil" / "dedup_report.json"
        manifest_file = PROJECT_ROOT / "data" / "raw" / "cnil" / "keep_manifest.json"
        
        if report_file.exists():
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            stats = report.get('stats', {})
            
            for typ in ['html', 'pdf', 'docs', 'images']:
                before = stats.get(f'{typ}_before', 0)
                after = stats.get(f'{typ}_after', 0)
                removed = stats.get(f'{typ}_removed', 0)
                print(f"   📊 {typ.upper():6s} : {before} → {after} (-{removed})")
            
            total_before = sum(stats.get(f'{t}_before', 0) for t in ['html', 'pdf', 'docs', 'images'])
            total_after = sum(stats.get(f'{t}_after', 0) for t in ['html', 'pdf', 'docs', 'images'])
            print(f"   📊 TOTAL  : {total_before} → {total_after} (-{total_before - total_after})")
            
            # Vérifier le manifest
            if manifest_file.exists():
                with open(manifest_file, 'r', encoding='utf-8') as f:
                    mf = json.load(f)
                manifest_total = len(mf.get('html',[])) + len(mf.get('pdfs',[])) + len(mf.get('docs',[])) + len(mf.get('images',[]))
                if manifest_total != total_after:
                    issues.append(f"Manifest ({manifest_total}) != rapport ({total_after})")
                else:
                    print(f"   📊 Manifest cohérent : {manifest_total} documents")
            
            if total_after == 0:
                issues.append("Corpus vide après déduplication !")
        else:
            issues.append("dedup_report.json MANQUANT")
    
    elif step_id == '5a':
        # Phase 5A : document_metadata.json doit exister avec classifications
        dm_file = PROJECT_ROOT / "data" / "raw" / "cnil" / "document_metadata.json"
        if not dm_file.exists():
            issues.append("document_metadata.json MANQUANT")
        else:
            try:
                with open(dm_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                classified = sum(1 for v in metadata.values() if v.get('nature'))
                error_count = sum(1 for v in metadata.values() if v.get('error'))
                
                print(f"   📊 Documents classifiés : {classified}/{len(metadata)}")
                
                if error_count > 0:
                    error_pct = error_count / max(1, len(metadata)) * 100
                    if error_pct > 10:
                        issues.append(f"{error_count} erreurs classification ({error_pct:.0f}%)")
                    else:
                        warnings.append(f"{error_count} erreurs classification ({error_pct:.1f}%)")
                
                # Distribution natures
                natures = {}
                for v in metadata.values():
                    n = v.get('nature', 'UNKNOWN')
                    natures[n] = natures.get(n, 0) + 1
                for n, c in sorted(natures.items()):
                    print(f"   📊   {n:12s} : {c}")
                
            except json.JSONDecodeError as e:
                issues.append(f"JSON invalide: {e}")
    
    elif step_id == '5b':
        # Phase 5B : processed_chunks.jsonl doit exister et être non vide
        chunks_file = PROJECT_ROOT / "data" / "raw" / "cnil" / "processed_chunks.jsonl"
        if not chunks_file.exists():
            issues.append("processed_chunks.jsonl MANQUANT")
        else:
            chunk_count = 0
            doc_ids = set()
            errors = 0
            
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        chunk_count += 1
                        doc_ids.add(chunk.get('document_id', ''))
                    except json.JSONDecodeError:
                        errors += 1
            
            print(f"   📊 Chunks totaux    : {chunk_count}")
            print(f"   📊 Documents uniques: {len(doc_ids)}")
            print(f"   📊 Moyenne          : {chunk_count / max(1, len(doc_ids)):.1f} chunks/doc")
            
            if errors > 0:
                warnings.append(f"{errors} lignes JSON invalides dans JSONL")
            
            if chunk_count == 0:
                issues.append("JSONL vide — aucun chunk produit")
    
    elif step_id == '6a':
        # Phase 6A : ChromaDB doit exister
        chromadb_dir = PROJECT_ROOT / "data" / "vectordb" / "chromadb"
        if not chromadb_dir.exists():
            issues.append("Dossier ChromaDB MANQUANT")
        else:
            # Essayer de compter les documents indexés
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(chromadb_dir))
                collection = client.get_collection("rag_dpo_chunks")
                count = collection.count()
                print(f"   📊 Documents indexés : {count}")
                
                if count == 0:
                    issues.append("ChromaDB vide — aucun document indexé")
            except Exception as e:
                warnings.append(f"Impossible de vérifier ChromaDB: {e}")
    
    elif step_id == '6b':
        # Phase 6B : document_summaries.json doit exister
        summaries_file = PROJECT_ROOT / "data" / "keep" / "cnil" / "document_summaries.json"
        if not summaries_file.exists():
            issues.append("document_summaries.json MANQUANT")
        else:
            try:
                with open(summaries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Format: dict {doc_path: {summary, ...}} 
                total = len(data)
                nav_count = sum(1 for s in data.values() if s.get('error') == 'navigation_content')
                empty_count = sum(1 for s in data.values() if s.get('error') == 'empty_content')
                error_count = sum(1 for s in data.values() if s.get('error') and s.get('error') not in ('navigation_content', 'empty_content'))
                ok_count = total - nav_count - empty_count - error_count
                
                print(f"   📊 Résumés générés  : {ok_count}")
                print(f"   📊 Navigation (skip): {nav_count}")
                if empty_count:
                    print(f"   📊 Vides (skip)     : {empty_count}")
                
                if error_count > 0:
                    error_pct = error_count / max(1, total) * 100
                    if error_pct > 10:
                        issues.append(f"{error_count} erreurs résumés ({error_pct:.0f}%)")
                    else:
                        warnings.append(f"{error_count} erreurs résumés ({error_pct:.1f}%)")
                
                if ok_count == 0:
                    issues.append("Aucun résumé généré")
                    
            except json.JSONDecodeError as e:
                issues.append(f"JSON invalide: {e}")
    
    elif step_id == '6c':
        # Phase 6C : vérifier nettoyage post-résumés
        summaries_file = PROJECT_ROOT / "data" / "keep" / "cnil" / "document_summaries.json"
        if not summaries_file.exists():
            issues.append("document_summaries.json MANQUANT")
        else:
            try:
                with open(summaries_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                cleaned = sum(1 for s in data.values() if s.get('cleaned'))
                nav_remaining = sum(1 for s in data.values() if s.get('error') == 'navigation_content')
                ok_count = sum(1 for s in data.values() if s.get('summary') and 'error' not in s and not s.get('cleaned'))
                
                print(f"   📊 Résumés OK       : {ok_count}")
                print(f"   📊 Nav nettoyés     : {cleaned}")
                print(f"   📊 Nav restants     : {nav_remaining}")
                
                if nav_remaining > 0:
                    warnings.append(f"{nav_remaining} entrées navigation non nettoyées")
                
                # Vérifier cohérence ChromaDB
                chromadb_dir = PROJECT_ROOT / "data" / "vectordb" / "chromadb"
                if chromadb_dir.exists():
                    try:
                        import chromadb
                        client = chromadb.PersistentClient(path=str(chromadb_dir))
                        collection = client.get_collection("rag_dpo_chunks")
                        print(f"   📊 ChromaDB chunks  : {collection.count()}")
                    except Exception:
                        pass
                
            except json.JSONDecodeError as e:
                issues.append(f"JSON invalide: {e}")
    
    # Bilan
    if warnings:
        for w in warnings:
            print(f"   ⚠️  {w}")
    
    if issues:
        for issue in issues:
            print(f"   ❌ {issue}")
        print(f"\n   ❌ Sanity check Phase {step_id} : {len(issues)} PROBLÈME(S)")
        return False
    else:
        print(f"   ✅ Sanity check Phase {step_id} : OK")
        return True


def check_state():
    """Vérifie l'état actuel du pipeline."""
    print("=" * 70)
    print("  ÉTAT DU PIPELINE RAG-DPO")
    print("=" * 70)
    
    checks = {
        "HTML bruts (raw)": ("dir", PROJECT_ROOT / "data" / "raw" / "cnil" / "html"),
        "PDF bruts (raw)": ("dir", PROJECT_ROOT / "data" / "raw" / "cnil" / "pdf"),
        "Docs bruts (raw)": ("dir", PROJECT_ROOT / "data" / "raw" / "cnil" / "docs"),
        "Classification hybride": ("file", PROJECT_ROOT / "data" / "raw" / "cnil" / "hybrid_classification.json"),
        "Keep manifest": ("file", PROJECT_ROOT / "data" / "raw" / "cnil" / "keep_manifest.json"),
        "HTML gardés (keep)": ("dir", PROJECT_ROOT / "data" / "keep" / "cnil" / "html"),
        "PDF gardés (keep)": ("dir", PROJECT_ROOT / "data" / "keep" / "cnil" / "pdf"),
        "Docs gardés (keep)": ("dir", PROJECT_ROOT / "data" / "keep" / "cnil" / "docs"),
        "Images gardées (keep)": ("dir", PROJECT_ROOT / "data" / "keep" / "cnil" / "images"),
        "Metadata (keep)": ("dir", PROJECT_ROOT / "data" / "keep" / "cnil" / "metadata"),
        "Classification images": ("file", PROJECT_ROOT / "data" / "raw" / "cnil" / "image_classification.json"),
        "Rapport déduplication": ("file", PROJECT_ROOT / "data" / "raw" / "cnil" / "dedup_report.json"),
        "Classification docs": ("file", PROJECT_ROOT / "data" / "raw" / "cnil" / "document_metadata.json"),
        "Chunks (processed)": ("file", PROJECT_ROOT / "data" / "raw" / "cnil" / "processed_chunks.jsonl"),
        "VectorDB ChromaDB": ("dir", PROJECT_ROOT / "data" / "vectordb" / "chromadb"),
        "Résumés documents": ("file", PROJECT_ROOT / "data" / "keep" / "cnil" / "document_summaries.json"),
    }
    
    for label, (kind, path) in checks.items():
        if path.exists():
            if kind == "file":
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"  OK  {label:30s} ({size_mb:.1f} MB)")
            else:
                count = sum(1 for _ in path.iterdir()) if path.is_dir() else 0
                print(f"  OK  {label:30s} ({count} fichiers)")
        else:
            print(f"  --  {label:30s} MANQUANT")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Reconstruction pipeline RAG-DPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python rebuild_pipeline.py                    # Pipeline complet (3→6b)
  python rebuild_pipeline.py --from 5b          # Reprendre depuis 5b
  python rebuild_pipeline.py --only 6a          # Seulement l'indexation
  python rebuild_pipeline.py --steps 5a 5b 6a   # Étapes spécifiques
  python rebuild_pipeline.py --check            # État du pipeline
  python rebuild_pipeline.py --only 3 --test 20 # Test Phase 3 sur 20 docs
  python rebuild_pipeline.py --fresh            # Ignorer résultats existants
  python rebuild_pipeline.py --no-sanity        # Pas de vérification post-phase
        """
    )
    parser.add_argument("--from", dest="from_step", choices=STEP_ORDER,
                       help="Reprendre depuis une étape spécifique")
    parser.add_argument("--only", choices=STEP_ORDER,
                       help="Exécuter seulement une étape")
    parser.add_argument("--steps", nargs='+', choices=STEP_ORDER,
                       help="Exécuter ces étapes spécifiques (dans l'ordre)")
    parser.add_argument("--check", action="store_true",
                       help="Vérifier l'état du pipeline")
    parser.add_argument("--test", type=int, default=None,
                       help="Mode test : limiter à N documents")
    parser.add_argument("--fresh", action="store_true",
                       help="Ignorer les résultats existants (recommencer à zéro)")
    parser.add_argument("--no-sanity", action="store_true",
                       help="Désactiver les sanity checks post-phase")
    
    args = parser.parse_args()
    
    if args.check:
        check_state()
        # Lancer aussi les sanity checks détaillés
        print("\n" + "=" * 70)
        print("  SANITY CHECKS DÉTAILLÉS")
        print("=" * 70)
        for step_id in STEP_ORDER:
            sanity_check(step_id, test_mode=args.test)
        return
    
    # Déterminer quelles étapes exécuter
    if args.only:
        steps_to_run = [args.only]
    elif args.steps:
        # Respecter l'ordre canonique
        steps_to_run = [s for s in STEP_ORDER if s in args.steps]
    elif args.from_step:
        start_idx = STEP_ORDER.index(args.from_step)
        steps_to_run = STEP_ORDER[start_idx:]
    else:
        steps_to_run = STEP_ORDER[:]
    
    # Afficher plan
    print("=" * 70)
    print("  RECONSTRUCTION PIPELINE RAG-DPO")
    print("  Modèle: mistral-nemo (12B, 128K ctx)")
    print("  Embeddings: BAAI/bge-m3 (1024 dim)")
    if args.test:
        print(f"  MODE TEST: {args.test} documents max")
    if args.fresh:
        print(f"  MODE FRESH: résultats existants ignorés")
    print(f"  Resume: {'désactivé (fresh)' if args.fresh else 'activé (reprise auto)'}")
    print(f"  Sanity checks: {'désactivés' if args.no_sanity else 'activés (post-phase)'}")
    print("=" * 70)
    
    print(f"\n  Étapes planifiées ({len(steps_to_run)}) :")
    for s in steps_to_run:
        print(f"    [{s:>2s}] {STEPS[s]['name']}")
    
    if not args.test:
        resp = input("\n  Lancer ? (O/n) : ").strip().lower()
        if resp == 'n':
            print("  Annulé.")
            return
    
    # Exécution
    total_start = time.time()
    results = []
    sanity_results = []
    
    for step_id in steps_to_run:
        elapsed = run_step(step_id, test_mode=args.test, fresh=args.fresh)
        results.append((step_id, STEPS[step_id]['name'], elapsed))
        
        # Sanity check post-phase
        if not args.no_sanity:
            ok = sanity_check(step_id, test_mode=args.test)
            sanity_results.append((step_id, ok))
            
            if not ok:
                print(f"\n⚠️  Problèmes détectés en Phase {step_id}.")
                resp = input("  [C]ontinuer / [R]elancer la phase / [A]rrêter ? (c/r/a) : ").strip().lower()
                if resp == 'r':
                    print(f"  🔄 Relance Phase {step_id}...")
                    elapsed2 = run_step(step_id, test_mode=args.test, fresh=args.fresh)
                    results[-1] = (step_id, STEPS[step_id]['name'], elapsed + elapsed2)
                    ok2 = sanity_check(step_id, test_mode=args.test)
                    sanity_results[-1] = (step_id, ok2)
                    if not ok2:
                        resp2 = input("  Toujours KO. Continuer quand même ? (o/N) : ").strip().lower()
                        if resp2 != 'o':
                            break
                elif resp == 'a':
                    print("  🛑 Arrêt du pipeline.")
                    break
        else:
            # Pause entre étapes pour confirmer
            if step_id != steps_to_run[-1]:
                resp = input(f"\n  Phase {step_id} terminée. Continuer ? (O/n) : ").strip().lower()
                if resp == 'n':
                    print("  🛑 Pipeline interrompu.")
                    break
    
    # Bilan
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print("  RECONSTRUCTION TERMINÉE")
    print("=" * 70)
    for step_id, name, elapsed in results:
        # Chercher le status sanity
        sanity_status = ""
        for sid, ok in sanity_results:
            if sid == step_id:
                sanity_status = " ✅" if ok else " ❌"
                break
        print(f"  [{step_id:>2s}] {name:50s} : {elapsed/60:.1f} min{sanity_status}")
    print(f"  {'TOTAL':55s} : {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f}h)")
    print("=" * 70)
    
    # État final
    check_state()


if __name__ == "__main__":
    main()
