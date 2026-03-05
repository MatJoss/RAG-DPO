"""
Paths — Chemins centralisés pour le projet RAG-DPO.

Tous les chemins sont calculés par défaut à partir de la racine du projet,
mais peuvent être overridés via des variables d'environnement.
Cela permet au même code de tourner en local ET dans Docker sans modification.

Variables d'environnement supportées :
    PROJECT_ROOT     : racine du projet (défaut: détecté automatiquement)
    OLLAMA_BASE_URL  : URL du serveur Ollama (défaut: http://localhost:11434)
    OLLAMA_MODEL     : modèle LLM Ollama (défaut: mistral-nemo)

Les dossiers sont créés automatiquement si nécessaire (logs, data, etc.).

Usage :
    from src.utils.paths import PROJECT_ROOT, LOGS_DIR, OLLAMA_BASE_URL
"""
import os
from pathlib import Path

# ── Racine du projet ─────────────────────────────────────────────────────────
# En local   : détecté via la position de ce fichier (src/utils/paths.py → 2x parent)
# En Docker  : /app (défini par WORKDIR dans le Dockerfile)
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent.parent))

# ── Dossiers de données ──────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
VECTORDB_DIR = DATA_DIR / "vectordb" / "chromadb"
RAW_DIR = DATA_DIR / "raw"
KEEP_DIR = DATA_DIR / "keep"
METADATA_DIR = DATA_DIR / "metadata"

# ── Dossiers de modèles ─────────────────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
HF_CACHE_DIR = MODELS_DIR / "huggingface" / "hub"

# ── Logs ─────────────────────────────────────────────────────────────────────
LOGS_DIR = PROJECT_ROOT / "logs"

# ── Configuration ────────────────────────────────────────────────────────────
CONFIGS_DIR = PROJECT_ROOT / "configs"
CONFIG_PATH = CONFIGS_DIR / "config.yaml"
ENTERPRISE_TAGS_PATH = CONFIGS_DIR / "enterprise_tags.json"

# ── Services externes ────────────────────────────────────────────────────────
# En local   : Ollama tourne sur localhost
# En Docker  : Ollama tourne dans un conteneur frère, accessible via son nom de service
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral-nemo")

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMADB_COLLECTION = "rag_dpo_chunks"
