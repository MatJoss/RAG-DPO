"""
Diagnostic Rapide du Scraping
Analyse l'état actuel et identifie les problèmes
Détermine si le scraping est complet ou non
"""

import json
from pathlib import Path
from collections import Counter
from datetime import datetime
import sys


def analyze_completion(project_path: Path, state: dict, actual_html: int, visited: int) -> bool:
    """Détermine si le scraping est complet"""
    
    print("\n" + "=" * 70)
    print("🔍 ANALYSE DE COMPLÉTION")
    print("=" * 70)
    
    # 1. Vérifier la date de dernière mise à jour
    last_update = state.get('last_update')
    if last_update:
        try:
            last_dt = datetime.fromisoformat(last_update)
            now = datetime.now()
            hours_ago = (now - last_dt).total_seconds() / 3600
            
            print(f"\n⏰ Dernière mise à jour : il y a {hours_ago:.1f}h")
            
            if hours_ago < 0.5:
                print(f"   ⚠️  Scraping très récent ou en cours")
                is_recent = True
            elif hours_ago < 2:
                print(f"   ⚠️  Scraping récent (moins de 2h)")
                is_recent = True
            else:
                print(f"   ✅ Scraping ancien (probablement terminé)")
                is_recent = False
        except:
            print(f"\n⏰ Pas de date de dernière mise à jour")
            is_recent = False
    else:
        is_recent = False
    
    # 2. Analyser les HTML vs URLs visitées (avec vrais chiffres)
    if visited > 0:
        save_ratio = actual_html / visited
        print(f"\n📊 Ratio sauvegarde : {save_ratio*100:.1f}%")
        print(f"   ({actual_html} HTML réels / {visited} URLs)")
        
        if save_ratio > 0.95:
            print(f"   ✅ Excellent ratio (>95%)")
            good_ratio = True
        elif save_ratio > 0.90:
            print(f"   ✅ Très bon ratio (>90%)")
            good_ratio = True
        elif save_ratio > 0.85:
            print(f"   ⚠️  Ratio correct (>85%)")
            good_ratio = True
        else:
            print(f"   ❌ Ratio faible (<85%) - beaucoup d'échecs ou incomplet")
            good_ratio = False
    else:
        good_ratio = False
    
    # 3. Vérifier les échecs
    failed_urls = state.get('failed_urls', [])
    failed_count = len(failed_urls)
    
    if failed_count > 0:
        print(f"\n❌ Échecs enregistrés : {failed_count}")
        fail_ratio = failed_count / visited if visited > 0 else 0
        if fail_ratio < 0.03:
            print(f"   ✅ Très peu d'échecs (<3%)")
        elif fail_ratio < 0.05:
            print(f"   ✅ Peu d'échecs (<5%)")
        else:
            print(f"   ⚠️  Beaucoup d'échecs (≥5%)")
    
    # 4. Conclusion
    print(f"\n" + "=" * 70)
    print("🎯 CONCLUSION")
    print("=" * 70)
    
    # Critères de complétion
    criteria = {
        'good_ratio': save_ratio > 0.95,
        'not_too_recent': not is_recent,
        'enough_html': actual_html > 5000,
    }
    
    all_good = all(criteria.values())
    
    if all_good:
        print(f"\n✅ Le scraping semble COMPLET !")
        print(f"\n   Critères validés :")
        print(f"   ✓ Excellent ratio ({save_ratio*100:.1f}%)")
        print(f"   ✓ {actual_html} HTML collectés")
        if not is_recent:
            print(f"   ✓ Scraping terminé il y a {hours_ago:.1f}h")
        
        print(f"\n💡 Actions recommandées :")
        print(f"   1. ✅ Scraping HTML TERMINÉ")
        print(f"   2. ⚠️  0 PDF/Docs détectés (problème de détection)")
        print(f"   3. → Lancer patch pour récupérer PDF/docs")
        print(f"      python src/scraping/patch_missing_files.py")
        print(f"   4. → Retry des échecs temporaires si besoin")
        print(f"      python src/scraping/retry_failed_urls.py")
        
        return True
    
    else:
        print(f"\n⚠️  Le scraping semble INCOMPLET ou EN COURS")
        
        print(f"\n   Problèmes détectés :")
        if not criteria['good_ratio']:
            print(f"   ❌ Ratio sauvegarde faible ({save_ratio*100:.1f}%)")
            print(f"      → {visited - actual_html} URLs visitées non sauvegardées")
        if is_recent:
            print(f"   ⚠️  Dernière activité il y a {hours_ago:.1f}h (récent)")
        if not criteria['enough_html']:
            print(f"   ❌ Peu de HTML ({actual_html} < 5000)")
        
        print(f"\n💡 Actions recommandées :")
        print(f"   1. Test : relancer pour voir si de nouvelles URLs")
        print(f"      python src/scraping/cnil_scraper.py --depth 5")
        print(f"   2. Observer si ça scrape ou s'arrête immédiatement")
        print(f"   3. Si 0 nouvelles pages → scraping fini")
        print(f"   4. Si nouvelles pages → laisser finir")
        
        return False
    
    print("=" * 70)


def analyze_scraping_state(project_root: str = '.'):
    """Analyse l'état du scraping"""
    
    project_path = Path(project_root)
    state_file = project_path / 'data' / 'metadata' / 'scraping_state.json'
    
    if not state_file.exists():
        print("❌ Fichier scraping_state.json introuvable")
        return
    
    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)
    
    print("=" * 70)
    print("📊 DIAGNOSTIC SCRAPING")
    print("=" * 70)
    
    # Stats du state
    stats = state.get('stats', {})
    visited = len(state.get('visited_urls', []))
    
    html_state = stats.get('html', 0)
    pdf_state = stats.get('pdf', 0)
    docs_state = stats.get('docs', 0)
    errors = stats.get('errors', 0)
    
    print(f"\n📄 URLs visitées (state) : {visited}")
    
    # Compter les fichiers RÉELS sur disque
    html_dir = project_path / 'data' / 'raw' / 'html'
    pdf_dir = project_path / 'data' / 'raw' / 'pdf'
    docs_dir = project_path / 'data' / 'raw' / 'docs'
    
    actual_html = len(list(html_dir.glob('*.html'))) if html_dir.exists() else 0
    actual_pdf = len(list(pdf_dir.glob('*.pdf'))) if pdf_dir.exists() else 0
    actual_odt = len(list(docs_dir.glob('*.odt'))) if docs_dir.exists() else 0
    actual_xlsx = len(list(docs_dir.glob('*.xlsx'))) if docs_dir.exists() else 0
    actual_docx = len(list(docs_dir.glob('*.docx'))) if docs_dir.exists() else 0
    actual_docs = actual_odt + actual_xlsx + actual_docx
    
    print(f"\n💾 Fichiers RÉELS sur disque :")
    print(f"   HTML       : {actual_html}")
    print(f"   PDF        : {actual_pdf}")
    print(f"   Documents  : {actual_docs} (ODT: {actual_odt}, XLSX: {actual_xlsx}, DOCX: {actual_docx})")
    print(f"   Total      : {actual_html + actual_pdf + actual_docs}")
    
    print(f"\n📊 Stats dans scraping_state.json :")
    print(f"   HTML       : {html_state}")
    print(f"   PDF        : {pdf_state}")
    print(f"   Documents  : {docs_state}")
    
    # Vérifier cohérence
    if actual_html != html_state:
        print(f"\n⚠️  INCOHÉRENCE DÉTECTÉE !")
        print(f"   State dit {html_state} HTML, mais il y en a {actual_html} sur disque")
        print(f"   → Les stats du state ne sont pas à jour")
    
    # URLs manquantes
    missing = visited - actual_html
    
    if missing > 0:
        print(f"\n⚠️  URLs MANQUANTES : {missing}")
        print(f"   ({missing / visited * 100:.1f}% des URLs visitées)")
        print(f"   Causes possibles :")
        print(f"   - Redirections (URL comptée 2x)")
        print(f"   - Erreurs non loggées")
        print(f"   - Fichiers binaires mal détectés")
    
    print(f"\n❌ Erreurs dans state : {errors}")
    
    # Analyser complétion avec les VRAIS chiffres
    is_complete = analyze_completion(project_path, state, actual_html, visited)
    
    # Analyser l'état "fini ou pas"
    is_complete = analyze_completion(project_path, state)
    
    print("=" * 70)
    print("📊 DIAGNOSTIC SCRAPING")
    print("=" * 70)
    
    # Stats générales
    stats = state.get('stats', {})
    visited = len(state.get('visited_urls', []))
    
    html = stats.get('html', 0)
    pdf = stats.get('pdf', 0)
    docs = stats.get('docs', 0)
    errors = stats.get('errors', 0)
    
    print(f"\n📄 URLs visitées : {visited}")
    print(f"\n💾 Fichiers sauvegardés :")
    print(f"   HTML       : {html}")
    print(f"   PDF        : {pdf}")
    print(f"   Documents  : {docs}")
    print(f"   Total      : {html + pdf + docs}")
    
    # URLs manquantes
    missing = visited - (html + pdf + docs)
    
    if missing > 0:
        print(f"\n⚠️  URLs MANQUANTES : {missing}")
        print(f"   ({missing / visited * 100:.1f}% des URLs visitées)")
    
    print(f"\n❌ Erreurs détectées : {errors}")
    
    # Analyser les échecs
    failed_urls = state.get('failed_urls', [])
    
    if failed_urls:
        print(f"\n📋 URLs EN ÉCHEC : {len(failed_urls)}")
        
        # Classifier les erreurs
        error_types = Counter()
        error_examples = {}
        
        for failed in failed_urls:
            error = failed.get('error', 'Unknown')
            error_lower = error.lower()
            
            # Classifier
            if any(x in error_lower for x in ['timeout', '503', '502', '504']):
                error_type = 'Temporaire (timeout/503)'
            elif '429' in error_lower or 'rate limit' in error_lower:
                error_type = 'Rate Limit (429)'
            elif any(x in error_lower for x in ['404', 'not found']):
                error_type = 'Not Found (404)'
            elif any(x in error_lower for x in ['403', 'forbidden']):
                error_type = 'Forbidden (403)'
            elif any(x in error_lower for x in ['connection', 'network', 'ssl']):
                error_type = 'Erreur Réseau'
            else:
                error_type = 'Autre'
            
            error_types[error_type] += 1
            
            if error_type not in error_examples:
                error_examples[error_type] = {
                    'url': failed.get('url', '')[:60] + '...',
                    'error': error[:80]
                }
        
        print("\n   Répartition par type :")
        for error_type, count in error_types.most_common():
            pct = count / len(failed_urls) * 100
            print(f"   {error_type:30s} : {count:4d} ({pct:5.1f}%)")
            
            # Exemple
            example = error_examples.get(error_type)
            if example:
                print(f"      Ex: {example['url']}")
                print(f"          {example['error']}")
        
        # Recommandations
        print("\n💡 RECOMMANDATIONS :")
        
        temp_count = sum(count for error_type, count in error_types.items() 
                        if 'Temporaire' in error_type or 'Rate Limit' in error_type 
                        or 'Réseau' in error_type)
        
        if temp_count > 0:
            print(f"\n   ✅ {temp_count} erreurs temporaires détectées")
            print(f"      → RÉCUPÉRABLES avec retry intelligent")
            print(f"      → Commande : python src/scraping/retry_failed_urls.py")
        
        perm_count = sum(count for error_type, count in error_types.items() 
                        if '404' in error_type or '403' in error_type)
        
        if perm_count > 0:
            print(f"\n   ⚠️  {perm_count} erreurs permanentes (404/403)")
            print(f"      → Non récupérables (pages n'existent plus)")
    
    else:
        print("\n✅ Aucune URL en échec enregistrée")
    
    # Analyse des 130 manquantes
    if missing > len(failed_urls):
        diff = missing - len(failed_urls)
        print(f"\n🤔 MYSTÈRE : {diff} URLs manquantes non dans failed_urls")
        print(f"   Possible causes :")
        print(f"   - Erreurs non catchées (bugs)")
        print(f"   - Redirections non suivies")
        print(f"   - URLs dupliquées comptées 2 fois")
        print(f"   - Scraping interrompu puis repris")
    
    # Taille des données
    size_mb = stats.get('total_size_mb', 0)
    print(f"\n💾 Taille totale : {size_mb:.2f} MB ({size_mb/1024:.2f} GB)")
    
    print("\n" + "=" * 70)


def main():
    if len(sys.argv) > 1:
        project_root = sys.argv[1]
    else:
        project_root = '.'
    
    analyze_scraping_state(project_root)


if __name__ == "__main__":
    main()