"""
Analyse des Natures Juridiques (Post Run 7)
Analyse la distribution des natures LLM et vérifie le mapping
"""

import json
from pathlib import Path
from collections import Counter

def analyze_natures(metadata_file: str = 'data/document_metadata.json'):
    """Analyse la distribution des natures juridiques"""
    
    metadata_path = Path(metadata_file)
    
    if not metadata_path.exists():
        print(f"❌ Fichier {metadata_file} introuvable")
        return
    
    print("=" * 70)
    print("📊 ANALYSE DES NATURES JURIDIQUES")
    print("=" * 70)
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    total_docs = len(metadata)
    
    # Compter les natures
    natures = []
    nature_to_docs = {}
    
    for file_path, result in metadata.items():
        nature = result.get('nature', 'UNKNOWN')
        natures.append(nature)
        
        # Ajouter la nature si pas encore vue
        if nature not in nature_to_docs:
            nature_to_docs[nature] = []
        
        nature_to_docs[nature].append(file_path)
    
    nature_counts = Counter(natures)
    
    print(f"\n📄 Documents analysés : {total_docs}")
    
    print(f"\n🔬 Distribution des NATURES :")
    
    # Trier par count décroissant
    for nature, count in nature_counts.most_common():
        pct = count / total_docs * 100 if total_docs > 0 else 0
        bar = "█" * int(pct / 2)
        
        # Marquer les natures inattendues
        if nature not in ['DOCTRINE', 'GUIDE', 'SANCTION', 'TECHNIQUE', 'MIXTE']:
            marker = "⚠️  "
        else:
            marker = "   "
        
        print(f"{marker}{nature:12s} : {count:4d} ({pct:5.1f}%) {bar}")
    
    # Vérifier le mapping nature → index
    print(f"\n🔄 Vérification du mapping Nature → Index :")
    
    mapping_check = {
        'DOCTRINE': {'expected': 'FONDAMENTAUX', 'actual': []},
        'GUIDE': {'expected': 'OPERATIONNEL', 'actual': []},
        'SANCTION': {'expected': 'SANCTIONS', 'actual': []},
        'TECHNIQUE': {'expected': 'TECHNIQUE', 'actual': []},
        'MIXTE': {'expected': 'FONDAMENTAUX', 'actual': []},
    }
    
    for file_path, result in metadata.items():
        nature = result.get('nature', 'UNKNOWN')
        primary_index = result.get('primary_index', 'UNKNOWN')
        
        if nature in mapping_check:
            mapping_check[nature]['actual'].append(primary_index)
    
    for nature, check in mapping_check.items():
        expected = check['expected']
        actual = Counter(check['actual'])
        
        if actual:
            most_common = actual.most_common(1)[0]
            actual_index, actual_count = most_common
            total_nature = len(check['actual'])
            pct = actual_count / total_nature * 100
            
            status = "✅" if actual_index == expected else "❌"
            print(f"   {status} {nature:12s} → {actual_index:15s} ({actual_count}/{total_nature} = {pct:.1f}%)")
            
            # Si pas 100%, montrer les autres
            if pct < 100:
                for idx, count in actual.most_common()[1:]:
                    pct_other = count / total_nature * 100
                    print(f"      ⚠️  {idx:15s} ({count}/{total_nature} = {pct_other:.1f}%)")
    
    # Distribution finale des index (pour référence)
    print(f"\n📂 Distribution finale des INDEX (résultat du mapping) :")
    
    index_counts = Counter([r.get('primary_index', 'UNKNOWN') for r in metadata.values()])
    
    for index in ['FONDAMENTAUX', 'OPERATIONNEL', 'SANCTIONS', 'SECTORIELS', 'TECHNIQUE']:
        count = index_counts.get(index, 0)
        pct = count / total_docs * 100 if total_docs > 0 else 0
        print(f"   {index:15s} : {count:4d} ({pct:5.1f}%)")
    
    # Analyse MIXTE en détail
    if nature_counts.get('MIXTE', 0) > 0:
        print(f"\n🔀 Analyse détaillée des documents MIXTE :")
        
        mixte_docs = nature_to_docs['MIXTE']
        mixte_secondary = []
        
        for file_path in mixte_docs:
            result = metadata[file_path]
            secondary = result.get('secondary_indexes', [])
            mixte_secondary.extend(secondary)
        
        secondary_counts = Counter(mixte_secondary)
        
        print(f"   Total MIXTE : {len(mixte_docs)}")
        print(f"   Index secondaires :")
        for idx, count in secondary_counts.most_common():
            pct = count / len(mixte_docs) * 100
            print(f"      {idx:15s} : {count:4d} ({pct:.1f}%)")
    
    # Exemples par nature
    print(f"\n📋 Exemples par nature (premiers 3 de chaque) :")
    
    # Prendre toutes les natures présentes
    for nature in sorted(nature_to_docs.keys()):
        docs = nature_to_docs[nature][:3]
        if docs:
            print(f"\n   {nature} ({len(nature_to_docs[nature])} docs) :")
            for file_path in docs:
                result = metadata[file_path]
                title = result.get('title', Path(file_path).stem)[:60]
                raison = result.get('raison', '')[:80]
                print(f"      - {title}")
                print(f"        → {raison}")
    
    print("\n" + "=" * 70)
    print("✅ Analyse terminée")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    
    metadata_file = sys.argv[1] if len(sys.argv) > 1 else 'data/document_metadata.json'
    analyze_natures(metadata_file)