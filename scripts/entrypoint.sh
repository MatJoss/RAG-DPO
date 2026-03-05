#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# entrypoint.sh — Script de démarrage du conteneur RAG-DPO
# ─────────────────────────────────────────────────────────────────────────────
# Vérifie que la base ChromaDB est présente avant de lancer Streamlit.
# Si absente, propose de la télécharger automatiquement.

set -e

VECTORDB_PATH="/app/data/vectordb/chromadb/chroma.sqlite3"

echo "🔒 RAG-DPO — Démarrage..."

# ── Vérifier la base ChromaDB ────────────────────────────────────────────────
if [ ! -f "$VECTORDB_PATH" ]; then
    echo ""
    echo "⚠️  Base ChromaDB CNIL introuvable."
    echo ""
    
    if [ "${AUTO_DOWNLOAD_DB:-true}" = "true" ]; then
        echo "📥 Téléchargement automatique de la base CNIL pré-construite..."
        python /app/scripts/download_cnil_db.py || {
            echo ""
            echo "❌ Téléchargement échoué."
            echo "   Téléchargez manuellement depuis :"
            echo "   https://github.com/MatJoss/RAG-DPO/releases"
            echo "   et placez les fichiers dans data/vectordb/chromadb/"
            echo ""
            echo "   Ou lancez le pipeline de construction :"
            echo "   python rebuild_pipeline.py"
            echo ""
            exit 1
        }
    else
        echo "   Pour télécharger la base pré-construite :"
        echo "   python scripts/download_cnil_db.py"
        echo ""
        echo "   Ou montez votre propre base dans data/vectordb/chromadb/"
        echo ""
        exit 1
    fi
fi

echo "✅ Base ChromaDB trouvée"

# ── Vérifier la connexion Ollama ─────────────────────────────────────────────
echo "🔗 Vérification connexion Ollama (${OLLAMA_BASE_URL})..."
MAX_RETRIES=30
RETRY=0
until curl -sf "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; do
    RETRY=$((RETRY + 1))
    if [ $RETRY -ge $MAX_RETRIES ]; then
        echo "❌ Impossible de joindre Ollama après ${MAX_RETRIES} tentatives"
        echo "   URL : ${OLLAMA_BASE_URL}"
        exit 1
    fi
    echo "   Attente Ollama... (${RETRY}/${MAX_RETRIES})"
    sleep 2
done
echo "✅ Ollama connecté"

# ── Vérifier le modèle LLM ──────────────────────────────────────────────────
echo "🤖 Vérification modèle ${OLLAMA_MODEL}..."
if ! curl -sf "${OLLAMA_BASE_URL}/api/tags" | python -c "
import sys, json
data = json.load(sys.stdin)
models = [m.get('name','').split(':')[0] for m in data.get('models', [])]
if '${OLLAMA_MODEL}' not in models:
    print('   Modèle ${OLLAMA_MODEL} non trouvé, modèles disponibles :', models)
    sys.exit(1)
" 2>/dev/null; then
    echo "⚠️  Modèle ${OLLAMA_MODEL} non trouvé dans Ollama."
    echo "   Téléchargement automatique..."
    curl -sf "${OLLAMA_BASE_URL}/api/pull" -d "{\"name\": \"${OLLAMA_MODEL}\"}" | \
        python -c "
import sys, json
for line in sys.stdin:
    try:
        d = json.loads(line)
        status = d.get('status', '')
        if 'pulling' in status:
            total = d.get('total', 0)
            completed = d.get('completed', 0)
            if total > 0:
                pct = completed * 100 / total
                print(f'   {status}: {pct:.0f}%', end='\r')
        elif status == 'success':
            print(f'\n✅ Modèle ${OLLAMA_MODEL} téléchargé')
    except json.JSONDecodeError:
        pass
" || echo "⚠️  Pull échoué — lancez manuellement : docker exec rag-dpo-ollama ollama pull ${OLLAMA_MODEL}"
else
    echo "✅ Modèle ${OLLAMA_MODEL} disponible"
fi

# ── Lancer Streamlit ─────────────────────────────────────────────────────────
echo ""
echo "🚀 Lancement Streamlit sur le port 8501..."
echo "   → http://localhost:8501"
echo ""

exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
