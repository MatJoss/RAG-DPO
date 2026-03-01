"""
RAG-DPO Assistant — Point d'entrée Streamlit multipage.

Initialise les ressources partagées (pipeline, logger, alerter)
et affiche la page d'accueil avec navigation vers :
- 💬 Chat : interface de questions-réponses RAG
- 📊 Dashboard : observabilité, métriques, alertes
"""
import streamlit as st
import sys
import logging
from pathlib import Path

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from src.utils.structured_logger import setup_structured_logging

# Structured logging — une seule fois
if "logging_initialized" not in st.session_state:
    setup_structured_logging(console=True)
    st.session_state.logging_initialized = True

logger = logging.getLogger(__name__)

# Configuration page
st.set_page_config(
    page_title="RAG-DPO Assistant",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Initialisation partagée (cache cross-page) ──

@st.cache_resource
def init_rag_system():
    """Initialise le pipeline RAG — appelé une seule fois, partagé entre pages."""
    import chromadb
    from src.utils.llm_provider import OllamaProvider
    from src.utils.embedding_provider import EmbeddingProvider
    from src.rag.pipeline import create_pipeline
    from src.utils.query_logger import QueryLogger
    from src.utils.alerter import Alerter, load_alert_config

    try:
        vectordb_path = project_root / "data" / "vectordb" / "chromadb"
        if not vectordb_path.exists():
            st.error(f"❌ VectorDB introuvable : {vectordb_path}")
            st.stop()

        client = chromadb.PersistentClient(path=str(vectordb_path))
        collection = client.get_collection("rag_dpo_chunks")

        llm_provider = OllamaProvider(
            base_url="http://localhost:11434",
            model="mistral-nemo"
        )
        embedding_provider = EmbeddingProvider(
            cache_dir=str(project_root / "models" / "huggingface" / "hub"),
        )

        pipeline = create_pipeline(
            collection=collection,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            n_documents=5,
            n_chunks_per_doc=3,
            enable_hybrid=True,
            enable_reranker=True,
            enable_summary_prefilter=True,
            enable_validation=True,
            model="mistral-nemo",
            temperature=0.0,
            debug_mode=False,
        )

        query_logger = QueryLogger()
        alert_config = load_alert_config()
        alerter = Alerter(alert_config)

        logger.info(
            "RAG system initialized",
            extra={"event": "rag_init", "component": "app", "detail": f"{collection.count()} chunks"}
        )

        return {
            "pipeline": pipeline,
            "chunk_count": collection.count(),
            "query_logger": query_logger,
            "alerter": alerter,
        }

    except Exception as e:
        logger.error(f"Init error: {e}", exc_info=True)
        st.error(f"❌ Erreur initialisation RAG : {e}")
        st.stop()


def get_rag_system():
    """Get or initialize the shared RAG system."""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = init_rag_system()
    return st.session_state.rag_system


# ── Page d'accueil ──

def main():
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f77b4; font-size: 3rem;">🔒 RAG-DPO Assistant</h1>
        <p style="font-size: 1.2rem; opacity: 0.8;">
            Assistant intelligent pour les Délégués à la Protection des Données
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Init système en arrière-plan
    with st.spinner("🚀 Initialisation du système RAG..."):
        system = get_rag_system()

    # Stats rapides
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📚 Chunks indexés", f"{system['chunk_count']:,}")
    with col2:
        stats = system["query_logger"].get_stats(hours=24)
        st.metric("📝 Queries (all-time)", stats["total_queries"])
    with col3:
        if stats["satisfaction_rate"] is not None:
            st.metric("😊 Satisfaction", f"{stats['satisfaction_rate']:.0f}%")
        else:
            st.metric("😊 Satisfaction", "—")

    st.markdown("---")

    # Navigation
    col_chat, col_dash = st.columns(2)

    with col_chat:
        st.markdown("""
        ### 💬 Chat RAG
        Posez vos questions sur le **RGPD**, la protection des données,
        et vos documents internes.
        
        - Recherche hybride (BM25 + sémantique)
        - Sources CNIL + documents internes
        - Reranking cross-encoder
        - 100% local
        """)
        st.page_link("pages/1_💬_Chat.py", label="**Ouvrir le Chat →**", icon="💬")

    with col_dash:
        st.markdown("""
        ### 📊 Dashboard
        Suivez les performances du système en temps réel.
        
        - Métriques de performance (temps, citations)
        - Feedback utilisateurs (👍/👎)
        - Historique des queries
        - Alertes automatiques
        """)
        st.page_link("pages/2_📊_Dashboard.py", label="**Ouvrir le Dashboard →**", icon="📊")

    # Footer
    st.markdown("---")
    st.caption(
        "🔒 RAG-DPO Assistant — Sources CNIL + documents internes — "
        "mistral-nemo 12B — 100% local"
    )


if __name__ == "__main__":
    main()
