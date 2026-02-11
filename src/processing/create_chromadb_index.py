"""
Phase 6A : Embeddings et Indexation ChromaDB
Génère les embeddings des chunks et les indexe dans ChromaDB
"""

import json
from pathlib import Path
import sys
import logging
from typing import Dict, List
from tqdm import tqdm
import chromadb
from chromadb.config import Settings

# Ajouter chemins
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src' / 'utils'))

from llm_provider import RAGConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Désactiver les logs verbeux HTTP d'Ollama
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)


class ChromaDBIndexer:
    """Indexation des chunks dans ChromaDB"""
    
    def __init__(self, project_root: str = '.'):
        self.project_root = Path(project_root)
        self.data_path = self.project_root / 'data'
        self.raw_cnil_path = self.data_path / 'raw' / 'cnil'
        
        # Fichiers
        self.chunks_file = self.raw_cnil_path / 'processed_chunks.jsonl'
        self.chroma_path = self.data_path / 'vectordb' / 'chromadb'
        
        # ChromaDB
        self.chroma_client = None
        self.collection = None
        
        # Embeddings
        try:
            config = RAGConfig()
            self.llm_provider = config.llm_provider
            logger.info(f"✅ Provider LLM initialisé pour embeddings")
        except Exception as e:
            logger.error(f"❌ Erreur init provider: {e}")
            raise
        
        # Cache URLs (document_path -> url)
        self.url_cache = {}
        self._load_url_cache()
        
        # Stats
        self.stats = {
            'chunks_loaded': 0,
            'chunks_indexed': 0,
            'errors': 0,
        }
    
    def init_chromadb(self, mode: str = 'reset'):
        """Initialise ChromaDB
        
        Args:
            mode: 'reset' (supprime et recrée), 'append' (ajoute), 'update' (mise à jour)
        """
        
        logger.info(f"📂 Initialisation ChromaDB : {self.chroma_path}")
        logger.info(f"⚡ Mode : {mode}")
        
        # Créer dossier si nécessaire
        self.chroma_path.mkdir(parents=True, exist_ok=True)
        
        # Client ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Gestion de la collection selon le mode
        if mode == 'reset':
            try:
                self.chroma_client.delete_collection(name="rag_dpo_chunks")
                logger.info("  🗑️  Collection existante supprimée")
            except:
                pass
            
            self.collection = self.chroma_client.create_collection(
                name="rag_dpo_chunks",
                metadata={
                    "description": "Chunks RGPD/CNIL avec classification hybride",
                    "hnsw:space": "cosine"
                }
            )
            logger.info(f"✅ Collection créée : rag_dpo_chunks")
        
        elif mode in ['append', 'update']:
            try:
                self.collection = self.chroma_client.get_collection(name="rag_dpo_chunks")
                existing_count = self.collection.count()
                logger.info(f"✅ Collection existante chargée : {existing_count} chunks")
                
                if mode == 'update':
                    # En mode update, on va charger les IDs existants pour détecter les modifications
                    logger.info("📋 Chargement des IDs existants pour update...")
                    self.existing_ids = set(self.collection.get(include=[])['ids'])
                    logger.info(f"   {len(self.existing_ids)} chunks existants")
            except:
                logger.warning("⚠️  Collection inexistante, création...")
                self.collection = self.chroma_client.create_collection(
                    name="rag_dpo_chunks",
                    metadata={
                        "description": "Chunks RGPD/CNIL avec classification hybride",
                        "hnsw:space": "cosine"
                    }
                )
                logger.info(f"✅ Collection créée : rag_dpo_chunks")
                self.existing_ids = set()
    
    def load_chunks(self) -> List[Dict]:
        """Charge les chunks depuis JSONL"""
        
        if not self.chunks_file.exists():
            logger.error(f"❌ {self.chunks_file} introuvable")
            return []
        
        chunks = []
        
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    chunks.append(chunk)
                except Exception as e:
                    logger.warning(f"Erreur parsing ligne: {e}")
                    self.stats['errors'] += 1
        
        self.stats['chunks_loaded'] = len(chunks)
        logger.info(f"📄 Chunks chargés : {len(chunks)}")
        
        return chunks
    
    def _load_url_cache(self):
        """Charge le cache URL depuis data/metadata/*.json
        
        Optimisation: Charge seulement les URLs nécessaires en lisant
        les document_path depuis keep_manifest.json
        
        Structure: {document_path: url}
        """
        
        metadata_dir = self.data_path / "metadata"
        if not metadata_dir.exists():
            logger.warning(f"⚠️  Répertoire metadata introuvable: {metadata_dir}")
            return
        
        logger.info("📥 Chargement cache URLs depuis keep_manifest...")
        
        # Charger keep_manifest pour savoir quels documents sont utilisés
        keep_manifest_path = self.data_path / "raw" / "cnil" / "keep_manifest.json"
        if not keep_manifest_path.exists():
            logger.warning(f"⚠️  keep_manifest introuvable, chargement URLs incomplet")
            return
        
        try:
            with open(keep_manifest_path, 'r', encoding='utf-8') as f:
                keep_data = json.load(f)
            
            # Extraire tous les file_path ET URLs des documents pertinents
            # L'URL est directement dans keep_manifest.json !
            for list_key in ['html', 'pdfs', 'docs']:  # NB: 'pdfs' pas 'pdf' !
                for item in keep_data.get(list_key, []):
                    metadata = item.get('metadata', {})
                    file_path = metadata.get('file_path', '')
                    url = metadata.get('url', '') or item.get('url', '')  # URL dans metadata OU top-level
                    parent_url = item.get('parent_url', '') or metadata.get('source_url', '')
                    
                    if not file_path:
                        continue
                    
                    # Normaliser path
                    normalized_path = file_path.replace('\\', '/')
                    
                    # Utiliser l'URL du keep_manifest (c'est la vraie URL CNIL)
                    if url:
                        self.url_cache[normalized_path] = url
                    # Pour PDFs/docs: stocker aussi parent_url comme fallback
                    if not url and parent_url:
                        self.url_cache[normalized_path] = parent_url
            
            logger.info(f"✅ Cache URLs chargé: {len(self.url_cache)} documents")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement keep_manifest: {e}")
    
    def _get_url(self, doc_path: str) -> str:
        """Récupère l'URL du document depuis le cache
        
        Args:
            doc_path: Chemin du document (data/raw/cnil/html/xxx.html)
        
        Returns:
            URL CNIL ou chemin si non trouvé
        """
        
        # Normaliser path
        normalized = doc_path.replace('\\', '/')
        
        return self.url_cache.get(normalized, doc_path)
    
    def _detect_source(self, doc_path: str) -> str:
        """Détecte la source du document
        
        Returns:
            'CNIL' ou 'ENTREPRISE'
        """
        
        path_lower = doc_path.lower()
        
        # Documents entreprise (custom)
        if 'entreprise' in path_lower or 'custom' in path_lower or 'internal' in path_lower:
            return 'ENTREPRISE'
        
        # Par défaut : CNIL
        return 'CNIL'
    
    def _detect_source_type(self, doc_path: str) -> str:
        """Détecte le type de fichier source
        
        Returns:
            'html', 'pdf', 'docx', 'xlsx', etc.
        """
        
        path = Path(doc_path)
        ext = path.suffix.lower()
        
        if ext in ['.html', '.htm']:
            return 'html'
        elif ext == '.pdf':
            return 'pdf'
        elif ext in ['.docx', '.doc']:
            return 'docx'
        elif ext in ['.xlsx', '.xls']:
            return 'xlsx'
        elif ext in ['.odt', '.ods']:
            return 'odt'
        else:
            return 'unknown'
    
    def _is_priority_source(self, doc_path: str) -> bool:
        """Détermine si le document est prioritaire
        
        Prioritaire si :
        - Source ENTREPRISE
        - Template/modèle entreprise
        - Politique interne
        
        Returns:
            True si prioritaire
        """
        
        path_lower = doc_path.lower()
        
        # Documents entreprise = toujours prioritaires
        if 'entreprise' in path_lower or 'custom' in path_lower or 'internal' in path_lower:
            return True
        
        # Templates/modèles = prioritaires
        if 'template' in path_lower or 'modele' in path_lower or 'modèle' in path_lower:
            return True
        
        # Politiques internes
        if 'politique' in path_lower or 'policy' in path_lower:
            return True
        
        return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Génère les embeddings pour une liste de textes"""
        
        try:
            embeddings = self.llm_provider.embed(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Erreur génération embeddings: {e}")
            return []
    
    def index_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """Indexe les chunks dans ChromaDB par batch"""
        
        logger.info(f"🔄 Indexation en cours (batch={batch_size})...")
        
        total_chunks = len(chunks)
        
        for i in tqdm(range(0, total_chunks, batch_size), desc="Indexation"):
            batch = chunks[i:i + batch_size]
            
            # Préparer données batch
            ids = []
            documents = []
            metadatas = []
            
            for chunk in batch:
                # ID unique
                chunk_id = chunk.get('chunk_id', f"chunk_{i}")
                ids.append(chunk_id)
                
                # Texte pour embedding
                text = chunk.get('text', '')
                heading = chunk.get('heading', '')
                
                # Combiner heading + text pour meilleur contexte
                if heading:
                    full_text = f"{heading}\n\n{text}"
                else:
                    full_text = text
                
                # Sécurité : tronquer si dépasse (ne devrait pas arriver avec chunks < 450 mots)
                documents.append(full_text[:2500])
                
                # Metadata pour filtrage
                doc_path = chunk.get('document_path', '')
                
                # URL source : priorité chunk.source_url (from process_and_chunk) > url_cache
                source_url = chunk.get('source_url', '') or self._get_url(doc_path)
                parent_url = chunk.get('parent_url', '')
                
                metadata = {
                    'document_id': chunk.get('document_id', ''),
                    'document_path': doc_path,
                    'document_nature': chunk.get('document_nature', 'GUIDE'),
                    'chunk_nature': chunk.get('chunk_nature', 'GUIDE'),
                    'chunk_index': chunk.get('chunk_index', 'OPERATIONNEL'),
                    'heading': heading[:200] if heading else '',
                    'page_info': chunk.get('page_info', ''),  # "Pages 3-5", "Sheet: Registre"
                    'confidence': chunk.get('confidence', 0.5),
                    'method': chunk.get('method', 'unknown'),
                    'word_count': len(text.split()),
                    'sectors': ','.join(chunk.get('sectors', [])),  # Liste → string
                    'file_type': chunk.get('file_type', self._detect_source_type(doc_path)),
                    'title': chunk.get('title', '')[:300],  # Description du document
                    
                    # SOURCE (CNIL vs Entreprise)
                    'source': self._detect_source(doc_path),
                    'source_type': self._detect_source_type(doc_path),
                    'is_priority': self._is_priority_source(doc_path),
                    'source_url': source_url,  # URL directe (page CNIL ou download PDF)
                    'parent_url': parent_url,  # Page CNIL qui référence ce PDF/doc
                }
                
                metadatas.append(metadata)
            
            # Générer embeddings
            embeddings = self.generate_embeddings(documents)
            
            if not embeddings or len(embeddings) != len(documents):
                logger.warning(f"⚠️  Batch {i//batch_size}: embeddings incomplets")
                self.stats['errors'] += len(batch)
                continue
            
            # Ajouter à ChromaDB
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                
                self.stats['chunks_indexed'] += len(batch)
            
            except Exception as e:
                logger.error(f"Erreur indexation batch {i//batch_size}: {e}")
                self.stats['errors'] += len(batch)
        
        logger.info(f"✅ Indexation terminée : {self.stats['chunks_indexed']} chunks")
    
    def verify_index(self):
        """Vérifie l'index créé avec tests de filtrage"""
        
        count = self.collection.count()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"🔍 VÉRIFICATION INDEX")
        logger.info(f"{'='*70}")
        logger.info(f"Chunks indexés : {count}")
        
        # Test 1 : Query simple
        logger.info(f"\n📋 Test 1 : Query simple - 'Comment faire une AIPD ?'")
        
        # Générer embedding avec Ollama (DOIT utiliser le même modèle que l'indexation)
        query_embedding = self.llm_provider.embed(["Comment faire une AIPD ?"])[0]
        
        results1 = self.collection.query(
            query_embeddings=[query_embedding],  # Utiliser embedding Ollama, pas query_texts
            n_results=10  # Récupérer plus pour voir la diversité
        )
        
        # Déduplication par document
        seen_docs = set()
        unique_results = []
        for doc, meta in zip(results1['documents'][0], results1['metadatas'][0]):
            doc_path = meta.get('document_path', '')
            if doc_path not in seen_docs:
                seen_docs.add(doc_path)
                unique_results.append((doc, meta))
                if len(unique_results) >= 3:  # Garder top 3 documents uniques
                    break
        
        logger.info(f"\n  📄 Top 3 documents uniques (sur {len(results1['documents'][0])} chunks):")
        for i, (doc, meta) in enumerate(unique_results):
            logger.info(f"\n  Résultat {i+1}:")
            logger.info(f"    Document  : {meta.get('document_path', '')[-60:]}")
            logger.info(f"    Nature    : {meta.get('chunk_nature')}")
            logger.info(f"    Index     : {meta.get('chunk_index')}")
            logger.info(f"    Source    : {meta.get('source')} ({meta.get('source_type')})")
            logger.info(f"    Priorité  : {'✅ OUI' if meta.get('is_priority') else 'Non'}")
            logger.info(f"    Confidence: {meta.get('confidence', 0):.2f}")
            logger.info(f"    Heading   : {meta.get('heading', '')[:60]}")
        
        # Test 2 : Query avec filtre GUIDE
        logger.info(f"\n📋 Test 2 : Query avec filtre (chunk_nature=GUIDE)")
        
        results2 = self.collection.query(
            query_embeddings=[query_embedding],  # Réutiliser même embedding
            n_results=3,
            where={"chunk_nature": "GUIDE"}
        )
        
        for i, (doc, meta) in enumerate(zip(results2['documents'][0], results2['metadatas'][0])):
            logger.info(f"\n  Résultat {i+1}:")
            logger.info(f"    Nature    : {meta.get('chunk_nature')} ✅ (filtré)")
            logger.info(f"    Method    : {meta.get('method')}")
            logger.info(f"    Preview   : {doc[:120]}...")
        
        # Test 3 : Stats par source (via get + len)
        logger.info(f"\n📊 Statistiques par source :")
        
        try:
            # Utiliser get() avec where et compter (ChromaDB 0.5+)
            cnil_results = self.collection.get(where={"source": "CNIL"}, limit=100000)
            cnil_count = len(cnil_results['ids'])
            
            entreprise_results = self.collection.get(where={"source": "ENTREPRISE"}, limit=100000)
            entreprise_count = len(entreprise_results['ids'])
            
            # Stats par nature
            doctrine_results = self.collection.get(where={"chunk_nature": "DOCTRINE"}, limit=100000)
            doctrine_count = len(doctrine_results['ids'])
            
            guide_results = self.collection.get(where={"chunk_nature": "GUIDE"}, limit=100000)
            guide_count = len(guide_results['ids'])
            
            sanction_results = self.collection.get(where={"chunk_nature": "SANCTION"}, limit=100000)
            sanction_count = len(sanction_results['ids'])
            
            technique_results = self.collection.get(where={"chunk_nature": "TECHNIQUE"}, limit=100000)
            technique_count = len(technique_results['ids'])
            
            logger.info(f"    Total chunks : {count:,}")
            logger.info(f"\n  Par source:")
            logger.info(f"    CNIL       : {cnil_count:,} chunks ({cnil_count/count*100:.1f}%)")
            logger.info(f"    ENTREPRISE : {entreprise_count:,} chunks ({entreprise_count/count*100:.1f}%)")
            
            logger.info(f"\n  Par nature:")
            logger.info(f"    DOCTRINE   : {doctrine_count:,} chunks ({doctrine_count/count*100:.1f}%)")
            logger.info(f"    GUIDE      : {guide_count:,} chunks ({guide_count/count*100:.1f}%)")
            logger.info(f"    SANCTION   : {sanction_count:,} chunks ({sanction_count/count*100:.1f}%)")
            logger.info(f"    TECHNIQUE  : {technique_count:,} chunks ({technique_count/count*100:.1f}%)")
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur stats with where: {e}")
            logger.info(f"    Total chunks : {count:,}")
        
        logger.info(f"\n{'='*70}")
    
    def run(self):
        """Exécute l'indexation complète"""
        
        print("=" * 70)
        print("🗄️  PHASE 6A : INDEXATION CHROMADB")
        print("=" * 70)
        
        # Mode d'indexation
        mode = getattr(self, 'mode', 'reset')
        
        # Init ChromaDB
        self.init_chromadb(mode=mode)
        
        # Load chunks
        chunks = self.load_chunks()
        
        if not chunks:
            print("\n❌ Pas de chunks à indexer")
            return
        
        print(f"\n📊 Chunks à indexer : {len(chunks)}")
        print(f"📍 Base ChromaDB : {self.chroma_path}")
        
        # Estimation
        est_min = len(chunks) / 100 * 0.5  # ~0.5 min par batch de 100
        print(f"⏱️  Durée estimée : ~{est_min:.0f} minutes")
        
        input("\n   Appuyez sur Entrée pour continuer...")
        
        # Indexation
        self.index_chunks(chunks, batch_size=100)
        
        # Vérification
        self.verify_index()
        
        # Résumé
        self._print_summary()
    
    def _print_summary(self):
        """Affiche résumé"""
        
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ INDEXATION")
        print("=" * 70)
        
        print(f"\n📄 Chunks chargés  : {self.stats['chunks_loaded']}")
        print(f"✅ Chunks indexés  : {self.stats['chunks_indexed']}")
        
        if self.stats['errors'] > 0:
            print(f"⚠️  Erreurs        : {self.stats['errors']}")
        
        success_rate = (self.stats['chunks_indexed'] / max(1, self.stats['chunks_loaded'])) * 100
        print(f"📈 Taux de succès  : {success_rate:.1f}%")
        
        print("\n" + "=" * 70)
        print(f"💾 ChromaDB : {self.chroma_path}")
        print(f"📦 Collection : rag_dpo_chunks")
        print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 6A : Indexation ChromaDB')
    parser.add_argument('--project-root', type=str, default='.', help='Racine du projet')
    parser.add_argument('--batch-size', type=int, default=100, help='Taille des batchs')
    parser.add_argument('--mode', type=str, default='reset', 
                       choices=['reset', 'append', 'update'],
                       help='Mode: reset (supprime et recrée), append (ajoute nouveaux), update (met à jour existants)')
    parser.add_argument('--verify-only', action='store_true',
                       help='Vérifie l\'index existant sans réindexer')
    
    args = parser.parse_args()
    
    indexer = ChromaDBIndexer(args.project_root)
    
    if args.verify_only:
        # Mode vérification uniquement
        print("=" * 70)
        print("🔍 MODE VÉRIFICATION UNIQUEMENT")
        print("=" * 70)
        
        indexer.chroma_path.mkdir(parents=True, exist_ok=True)
        indexer.chroma_client = chromadb.PersistentClient(
            path=str(indexer.chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        try:
            indexer.collection = indexer.chroma_client.get_collection(name="rag_dpo_chunks")
            print(f"\n✅ Collection chargée : {indexer.collection.count()} chunks\n")
            indexer.verify_index()
        except Exception as e:
            print(f"\n❌ Erreur : {e}")
            print("   Collection introuvable. Lancez d'abord une indexation avec --mode reset")
    else:
        # Mode indexation normal
        indexer.mode = args.mode
        indexer.run()


if __name__ == "__main__":
    main()
