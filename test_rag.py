#!/usr/bin/env python
"""
Test du système RAG complet en CLI
"""
import os
import sys
import logging
import argparse
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import chromadb
from src.utils.llm_provider import OllamaProvider
from src.utils.embedding_provider import EmbeddingProvider
from src.rag.pipeline import create_pipeline


# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Calmer les logs HTTP
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Questions de test DPO
TEST_QUESTIONS = [
    "Comment faire une analyse d'impact (AIPD) ?",
    "Quelles sont les obligations d'un DPO ?",
    "Comment gérer une violation de données personnelles ?",
    "Qu'est-ce qu'une donnée à caractère personnel ?",
    "Quelles sanctions en cas de non-conformité RGPD ?",
]


def main():
    parser = argparse.ArgumentParser(description="Test du système RAG")
    parser.add_argument(
        "--question",
        "-q",
        type=str,
        help="Question à poser (sinon mode interactif ou test)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Lance les questions de test prédéfinies"
    )
    parser.add_argument(
        "--filter-nature",
        type=str,
        choices=["DOCTRINE", "GUIDE", "SANCTION", "TECHNIQUE"],
        help="Filtre par nature de document"
    )
    parser.add_argument(
        "--n-documents",
        type=int,
        default=3,
        help="Nombre de documents à récupérer (défaut: 3)"
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=4,
        help="Chunks par document (défaut: 4)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mode debug avec affichage du contexte"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistral",
        help="Modèle Ollama à utiliser (défaut: mistral)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Température LLM (défaut: 0.0 = factuel strict)"
    )
    
    args = parser.parse_args()
    
    # Configuration paths
    vectordb_path = project_root / "data" / "vectordb" / "chromadb"
    
    if not vectordb_path.exists():
        logger.error(f"❌ VectorDB non trouvée: {vectordb_path}")
        logger.error("💡 Lancer d'abord: python src/processing/create_chromadb_index.py")
        return 1
    
    logger.info("🚀 Initialisation du système RAG...\n")
    
    # 1. Init ChromaDB
    logger.info(f"📂 Chargement ChromaDB: {vectordb_path}")
    client = chromadb.PersistentClient(path=str(vectordb_path))
    collection = client.get_collection("rag_dpo_chunks")
    logger.info(f"✅ Collection chargée: {collection.count()} chunks\n")
    
    # 2. Init LLM Provider
    logger.info(f"🤖 Initialisation Ollama ({args.model})...")
    llm_provider = OllamaProvider(
        base_url="http://localhost:11434",
        model=args.model
    )
    logger.info("✅ Ollama prêt\n")
    
    # 2b. Init Embedding Provider (BGE-M3)
    embedding_provider = EmbeddingProvider(
        cache_dir=str(project_root / "models" / "huggingface" / "hub"),
    )
    
    # 3. Init Pipeline
    logger.info(f"🔧 Configuration pipeline (docs={args.n_documents}, chunks={args.n_chunks})...")
    pipeline = create_pipeline(
        collection=collection,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        n_documents=args.n_documents,
        n_chunks_per_doc=args.n_chunks,
        model=args.model,
        temperature=args.temperature,
        debug_mode=args.debug
    )
    logger.info("✅ Pipeline configuré\n")
    
    # Préparation filtre
    where_filter = None
    if args.filter_nature:
        where_filter = {"chunk_nature": args.filter_nature}
        logger.info(f"🔍 Filtre actif: nature={args.filter_nature}\n")
    
    # Mode de fonctionnement
    if args.test:
        # Mode test avec questions prédéfinies
        logger.info("🧪 MODE TEST - Questions prédéfinies\n")
        run_test_mode(pipeline, where_filter)
    
    elif args.question:
        # Mode single question
        logger.info("💬 MODE SINGLE QUESTION\n")
        run_single_question(pipeline, args.question, where_filter)
    
    else:
        # Mode interactif
        logger.info("💬 MODE INTERACTIF - Tapez 'quit' pour quitter\n")
        run_interactive_mode(pipeline, where_filter)
    
    return 0


def run_single_question(pipeline, question: str, where_filter=None):
    """Mode question unique"""
    response = pipeline.query(
        question=question,
        where_filter=where_filter
    )
    
    if response.error:
        logger.error(f"❌ Erreur: {response.error}")
        return
    
    print(pipeline.format_response(response, show_sources=True))


def run_test_mode(pipeline, where_filter=None):
    """Mode test avec questions prédéfinies"""
    results = []
    
    for i, question in enumerate(TEST_QUESTIONS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST {i}/{len(TEST_QUESTIONS)}")
        logger.info(f"{'='*80}\n")
        
        response = pipeline.query(
            question=question,
            where_filter=where_filter
        )
        
        results.append({
            "question": question,
            "success": not response.error,
            "time": response.total_time,
            "cited_sources": len(response.cited_sources)
        })
        
        print(pipeline.format_response(response, show_sources=True))
        print("\n")
    
    # Résumé
    print("\n" + "="*80)
    print("RÉSUMÉ DES TESTS")
    print("="*80)
    
    success_count = sum(1 for r in results if r["success"])
    avg_time = sum(r["time"] for r in results) / len(results)
    avg_sources = sum(r["cited_sources"] for r in results) / len(results)
    
    print(f"Succès: {success_count}/{len(results)}")
    print(f"Temps moyen: {avg_time:.2f}s")
    print(f"Sources citées (moyenne): {avg_sources:.1f}")
    print("="*80)


def run_interactive_mode(pipeline, where_filter=None):
    """Mode interactif"""
    conversation_history = []
    
    while True:
        try:
            question = input("\n🤔 Votre question (ou 'quit'): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("👋 Au revoir!")
                break
            
            if not question:
                continue
            
            print()  # Ligne vide
            
            # Query avec historique
            response = pipeline.query(
                question=question,
                where_filter=where_filter,
                conversation_history=conversation_history
            )
            
            if response.error:
                logger.error(f"❌ Erreur: {response.error}")
                continue
            
            print(pipeline.format_response(response, show_sources=True))
            
            # Mise à jour historique
            conversation_history.append({"role": "user", "content": question})
            conversation_history.append({"role": "assistant", "content": response.answer})
            
            # Garde seulement les 5 derniers échanges
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
        
        except KeyboardInterrupt:
            logger.info("\n👋 Interrupted, au revoir!")
            break
        except Exception as e:
            logger.error(f"❌ Erreur: {e}", exc_info=True)


if __name__ == "__main__":
    sys.exit(main())
