# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — RAG-DPO Streamlit App
# ─────────────────────────────────────────────────────────────────────────────
# Image légère Python 3.11 + dépendances PyTorch CPU-only pour l'app.
# Le GPU est utilisé par Ollama (conteneur séparé), pas par ce conteneur.
# Les embeddings BGE-M3 tournent sur CPU ici (plus lent mais fonctionnel).
# Pour GPU embeddings aussi, voir la variante cuda dans les commentaires.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Empêcher les prompts interactifs pendant l'install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── Dossier de travail ───────────────────────────────────────────────────────
WORKDIR /app

# ── Dépendances Python (couche cachée si requirements.txt ne change pas) ────
COPY requirements.txt .

# Installer PyTorch CPU-only d'abord (plus léger, ~200 MB au lieu de ~2 GB)
# Note: si tu veux le GPU pour les embeddings aussi, remplace par:
#   pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu

# Installer le reste des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# ── Code source ──────────────────────────────────────────────────────────────
# En dev, on monte le code en volume (docker-compose). En prod, on le copie.
COPY . .

# Rendre l'entrypoint exécutable
RUN chmod +x /app/scripts/entrypoint.sh

# ── Variables d'environnement par défaut ─────────────────────────────────────
# Surchargées par docker-compose.yml ou .env
ENV OLLAMA_BASE_URL=http://ollama:11434
ENV OLLAMA_MODEL=mistral-nemo
ENV PROJECT_ROOT=/app
ENV AUTO_DOWNLOAD_DB=true

# ── Port Streamlit ───────────────────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── Commande de lancement ────────────────────────────────────────────────────
ENTRYPOINT ["/app/scripts/entrypoint.sh"]
