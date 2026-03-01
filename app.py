"""
Interface Streamlit - Chat RAG pour DPO

Architecture clé en main :
- Hybrid Search (BM25 + Semantic + RRF)
- Cross-Encoder Reranking
- Reverse Repacking
- Grounding Validation
"""
import streamlit as st
import json
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import chromadb
from src.utils.llm_provider import OllamaProvider
from src.utils.embedding_provider import EmbeddingProvider
from src.rag.pipeline import create_pipeline


# Configuration page
st.set_page_config(
    page_title="RAG-DPO Assistant",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging : on ne veut que les messages utiles
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
for noisy in ("httpx", "ollama", "chromadb", "sentence_transformers", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ── CSS ──
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .source-card {
        padding: 0.6rem 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: rgba(128, 128, 128, 0.1);
        margin: 0.3rem 0;
        color: inherit;
        font-size: 0.9rem;
    }
    .cited { border-left-color: #28a745; }
    .uncited { border-left-color: #6c757d; opacity: 0.7; }
    .metadata { font-size: 0.8rem; opacity: 0.8; }
    .source-card a { color: #4da6ff; }
    .msg-time {
        font-size: 0.75rem;
        opacity: 0.5;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialisation RAG (cachée) ──

@st.cache_resource
def init_rag_system():
    """Initialise le pipeline RAG complet — appelé une seule fois."""
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

        # Embedding provider BGE-M3 (FP16, GPU, 1024 dims)
        embedding_provider = EmbeddingProvider(
            cache_dir=str(project_root / "models" / "huggingface" / "hub"),
        )

        pipeline = create_pipeline(
            collection=collection,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            # Retrieval
            n_documents=5,
            n_chunks_per_doc=3,
            # Architecture complète — pas de toggle
            enable_hybrid=True,
            enable_reranker=True,
            enable_summary_prefilter=True,
            enable_validation=True,
            # Génération
            model="mistral-nemo",
            temperature=0.0,
            debug_mode=False,
        )

        return pipeline, collection.count()

    except Exception as e:
        st.error(f"❌ Erreur initialisation RAG : {e}")
        logger.error(f"Init error: {e}", exc_info=True)
        st.stop()


# ── Carte source (inline dans le chat) ──

def render_source_card(source):
    """Retourne le HTML d'une carte source."""
    cited_class = "cited" if source['cited'] else "uncited"
    cited_icon = "✅" if source['cited'] else "📄"

    url = source.get('url', source.get('path', ''))
    file_type = source.get('file_type', '')
    title = source.get('title', '')

    if url.startswith('http'):
        source_line = f'🔗 <a href="{url}" target="_blank">{url[:80]}</a>'
    else:
        source_line = f'📁 {url}'

    # Badge origine : CNIL ou Interne
    origin = source.get('origin', 'CNIL')
    if origin == 'ENTREPRISE':
        origin_badge = (
            ' <span style="background:rgba(255,165,0,0.3);padding:1px 6px;'
            'border-radius:3px;font-size:0.8em">📋 Interne</span>'
        )
    else:
        origin_badge = (
            ' <span style="background:rgba(0,128,255,0.2);padding:1px 6px;'
            'border-radius:3px;font-size:0.8em">🏛️ CNIL</span>'
        )

    type_badge = ''
    if file_type:
        type_badge = (
            f' <span style="background:rgba(128,128,128,0.2);padding:1px 6px;'
            f'border-radius:3px;font-size:0.8em">{file_type.upper()}</span>'
        )

    title_line = f'<br>📝 {title[:120]}' if title else ''

    locations = source.get('locations', [])
    location_line = ''
    if locations:
        location_line = '<br>📍 ' + ' | '.join(loc[:80] for loc in locations[:3])

    return f"""<div class="source-card {cited_class}">
        <strong>{cited_icon} Source {source['id']} - {source['nature']}{origin_badge}{type_badge}</strong>
        <div class="metadata">
            {source_line}{title_line}{location_line}<br>
            📊 Score : {source['score']:.3f}
        </div>
    </div>"""


def render_sources_block(sources):
    """Affiche les sources dans un expander sous la réponse."""
    if not sources:
        return
    
    cited = [s for s in sources if s['cited']]
    uncited = [s for s in sources if not s['cited']]
    n_cited = len(cited)
    n_total = len(sources)
    
    with st.expander(f"📚 {n_cited}/{n_total} sources citées", expanded=False):
        # Sources citées d'abord
        for source in sorted(cited, key=lambda s: s['id']):
            st.markdown(render_source_card(source), unsafe_allow_html=True)
        # Sources non citées ensuite
        for source in sorted(uncited, key=lambda s: s['id']):
            st.markdown(render_source_card(source), unsafe_allow_html=True)


# ── Application ──

def main():
    st.markdown('<div class="main-header">🔒 RAG-DPO Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Filtres optionnels
        st.subheader("🔍 Filtres")
        filter_nature = st.multiselect(
            "Nature des documents",
            ["DOCTRINE", "GUIDE", "SANCTION", "TECHNIQUE"],
            default=[]
        )
        
        # Filtre tags entreprise
        tags_registry_path = project_root / "configs" / "enterprise_tags.json"
        enterprise_tag_options = []
        tag_labels = {}
        default_tags = []
        if tags_registry_path.exists():
            with open(tags_registry_path, 'r', encoding='utf-8') as f:
                tags_registry = json.load(f)
            enterprise_tag_options = tags_registry.get('active_tags', [])
            tag_labels = tags_registry.get('labels', {})
            default_tags = [
                t for t in tags_registry.get('default_tags', [])
                if t in enterprise_tag_options
            ]
        
        selected_enterprise_tags = []
        if enterprise_tag_options:
            st.subheader("🏷️ Documents internes")
            # Afficher les labels lisibles dans le multiselect
            display_options = [
                tag_labels.get(t, t) for t in enterprise_tag_options
            ]
            display_defaults = [
                tag_labels.get(t, t) for t in default_tags
            ]
            selected_display = st.multiselect(
                "Catégories à inclure",
                display_options,
                default=display_defaults,
                help="Sélectionnez les types de documents internes à consulter (CNIL est toujours inclus)"
            )
            # Reconvertir labels → tags techniques
            label_to_tag = {tag_labels.get(t, t): t for t in enterprise_tag_options}
            selected_enterprise_tags = [label_to_tag[d] for d in selected_display]

        # Profondeur de recherche
        st.subheader("📥 Profondeur")
        depth = st.select_slider(
            "Ressources à consulter",
            options=["Normal", "Approfondi", "Exhaustif"],
            value="Normal",
            help="Normal = 5 docs / Approfondi = 8 docs / Exhaustif = 12 docs"
        )

        st.markdown("---")

        # Nouvelle conversation
        if st.button("🔄 Nouvelle conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()

        # Stats
        st.markdown("---")
        if 'chunk_count' in st.session_state:
            st.metric("📚 Chunks indexés", f"{st.session_state.chunk_count:,}")
        if 'messages' in st.session_state:
            n_questions = sum(1 for m in st.session_state.messages if m['role'] == 'user')
            if n_questions:
                st.metric("💬 Questions posées", n_questions)

    # ── Paramètres dérivés du slider ──
    depth_config = {
        "Normal":     {"n_documents": 5,  "n_chunks_per_doc": 3},
        "Approfondi": {"n_documents": 8,  "n_chunks_per_doc": 4},
        "Exhaustif":  {"n_documents": 12, "n_chunks_per_doc": 5},
    }
    params = depth_config[depth]

    # ── Init pipeline ──
    if 'pipeline' not in st.session_state:
        with st.spinner("🚀 Initialisation du système RAG..."):
            pipeline, chunk_count = init_rag_system()
            st.session_state.pipeline = pipeline
            st.session_state.chunk_count = chunk_count

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # ── Filtre ChromaDB ──
    where_filter = None
    if filter_nature:
        where_filter = {"chunk_nature": {"$in": filter_nature}}

    # ── Affichage de l'historique ──
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Horodatage
            if "timestamp" in message:
                st.markdown(f'<div class="msg-time">{message["timestamp"]}</div>', unsafe_allow_html=True)
            # Contenu
            st.markdown(message["content"])
            # Sources (sous les réponses assistant uniquement)
            if message["role"] == "assistant" and "sources" in message:
                render_sources_block(message["sources"])

    # ── Saisie (pleine largeur, toujours en bas) ──
    if prompt := st.chat_input("Posez votre question sur le RGPD / la protection des données..."):
        now = datetime.now().strftime("%H:%M")

        # Message utilisateur
        with st.chat_message("user"):
            st.markdown(f'<div class="msg-time">{now}</div>', unsafe_allow_html=True)
            st.markdown(prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": now,
        })

        # Réponse assistant
        with st.chat_message("assistant"):
            with st.spinner("🤔 Recherche et analyse en cours..."):
                try:
                    response = st.session_state.pipeline.query(
                        question=prompt,
                        where_filter=where_filter,
                        enterprise_tags=selected_enterprise_tags or None,
                        conversation_history=st.session_state.conversation_history,
                        n_documents=params["n_documents"],
                        n_chunks_per_doc=params["n_chunks_per_doc"],
                    )

                    if response.error:
                        st.error(f"❌ {response.error}")
                    else:
                        resp_time = datetime.now().strftime("%H:%M")
                        st.markdown(f'<div class="msg-time">{resp_time}</div>', unsafe_allow_html=True)
                        st.markdown(response.answer)

                        st.caption(
                            f"⏱️ {response.total_time:.1f}s "
                            f"| 📚 {len(response.sources)} sources "
                            f"| ✅ {len(response.cited_sources)} citées"
                        )

                        # Sources inline sous la réponse
                        render_sources_block(response.sources)

                        # Stocker message + sources pour le replay
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "timestamp": resp_time,
                            "sources": response.sources,
                        })
                        st.session_state.conversation_history.append({
                            "role": "user", "content": prompt
                        })
                        st.session_state.conversation_history.append({
                            "role": "assistant", "content": response.answer
                        })

                except Exception as e:
                    st.error(f"❌ Erreur inattendue : {e}")
                    logger.error(f"Query error: {e}", exc_info=True)

    # Footer
    st.markdown("---")
    st.caption("🔒 RAG-DPO Assistant — Sources CNIL + documents internes — mistral-nemo 12B — 100% local")


if __name__ == "__main__":
    main()
