"""
Page Chat — Interface de questions-réponses RAG pour DPO.

Hérite du pipeline initialisé dans app.py (st.cache_resource).
"""
import streamlit as st
import json
import sys
import hashlib
import logging
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

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


# ── Helpers ──

def get_system():
    """Récupère le système RAG partagé, initialisé dans app.py."""
    if "rag_system" not in st.session_state:
        # Fallback : si on arrive directement sur cette page
        from app import init_rag_system
        st.session_state.rag_system = init_rag_system()
    return st.session_state.rag_system


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
        for source in sorted(cited, key=lambda s: s['id']):
            st.markdown(render_source_card(source), unsafe_allow_html=True)
        for source in sorted(uncited, key=lambda s: s['id']):
            st.markdown(render_source_card(source), unsafe_allow_html=True)


# ── Page principale ──

def main():
    st.markdown('<div class="main-header">🔒 RAG-DPO Assistant</div>', unsafe_allow_html=True)
    st.markdown("---")

    system = get_system()
    pipeline = system["pipeline"]
    agent_pipeline = system.get("agent_pipeline")
    query_logger = system["query_logger"]

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Toggle mode pipeline
        if agent_pipeline is not None:
            st.subheader("🤖 Mode Pipeline")
            use_agent = st.toggle(
                "Agent LangGraph",
                value=True,
                help="Active le pipeline agent (LangGraph) avec boucle de validation"
            )
            if use_agent:
                st.caption("🤖 Agent: classify→retrieve→generate→validate→respond")
            else:
                st.caption("⚡ Natif: intent-aware single-gen")
        else:
            use_agent = False

        # Filtres nature
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
            display_options = [tag_labels.get(t, t) for t in enterprise_tag_options]
            display_defaults = [tag_labels.get(t, t) for t in default_tags]
            selected_display = st.multiselect(
                "Catégories à inclure",
                display_options,
                default=display_defaults,
                help="Sélectionnez les types de documents internes à consulter (CNIL est toujours inclus)"
            )
            label_to_tag = {tag_labels.get(t, t): t for t in enterprise_tag_options}
            selected_enterprise_tags = [label_to_tag[d] for d in selected_display]

        # Profondeur
        st.subheader("📥 Profondeur")
        depth = st.select_slider(
            "Ressources à consulter",
            options=["Normal", "Approfondi", "Exhaustif"],
            value="Normal",
            help="Normal = 5 docs / Approfondi = 8 docs / Exhaustif = 12 docs"
        )

        st.markdown("---")

        if st.button("🔄 Nouvelle conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.rerun()

        # Stats compactes sidebar
        st.markdown("---")
        st.metric("📚 Chunks indexés", f"{system['chunk_count']:,}")
        if 'messages' in st.session_state:
            n_questions = sum(1 for m in st.session_state.messages if m['role'] == 'user')
            if n_questions:
                st.metric("💬 Questions posées", n_questions)

        # Mini observabilité sidebar
        stats = query_logger.get_stats(hours=24)
        if stats['total_queries'] > 0:
            st.markdown("---")
            st.caption("📊 Observabilité")
            c1, c2 = st.columns(2)
            c1.metric("📝 Queries", stats['total_queries'])
            c2.metric("⏱️ Moy.", f"{stats['avg_total_time']:.1f}s")
            if stats['recent_errors'] > 0:
                st.warning(f"⚠️ {stats['recent_errors']} erreur(s) / 24h")

    # ── Params ──
    depth_config = {
        "Normal":     {"n_documents": 5,  "n_chunks_per_doc": 3},
        "Approfondi": {"n_documents": 8,  "n_chunks_per_doc": 4},
        "Exhaustif":  {"n_documents": 12, "n_chunks_per_doc": 5},
    }
    params = depth_config[depth]

    # ── Init session ──
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # ── Filtre ChromaDB ──
    where_filter = None
    if filter_nature:
        where_filter = {"chunk_nature": {"$in": filter_nature}}

    # ── Affichage historique ──
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if "timestamp" in message:
                st.markdown(f'<div class="msg-time">{message["timestamp"]}</div>', unsafe_allow_html=True)
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                render_sources_block(message["sources"])
                if "answer_hash" in message:
                    msg_idx = st.session_state.messages.index(message)
                    feedback_key = f"feedback_{msg_idx}"
                    if feedback_key not in st.session_state:
                        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 10])
                        with col_fb1:
                            if st.button("👍", key=f"up_{msg_idx}", help="Réponse utile"):
                                st.session_state[feedback_key] = 1
                                query_logger.log_feedback(
                                    question=message.get("question", ""),
                                    answer_hash=message["answer_hash"],
                                    rating=1,
                                )
                                st.rerun()
                        with col_fb2:
                            if st.button("👎", key=f"down_{msg_idx}", help="Réponse à améliorer"):
                                st.session_state[feedback_key] = -1
                                query_logger.log_feedback(
                                    question=message.get("question", ""),
                                    answer_hash=message["answer_hash"],
                                    rating=-1,
                                )
                                st.rerun()
                    else:
                        fb_val = st.session_state[feedback_key]
                        fb_icon = "👍" if fb_val > 0 else "👎"
                        st.caption(f"{fb_icon} Merci pour votre retour")

    # ── Saisie ──
    if prompt := st.chat_input("Posez votre question sur le RGPD / la protection des données..."):
        now = datetime.now().strftime("%H:%M")

        with st.chat_message("user"):
            st.markdown(f'<div class="msg-time">{now}</div>', unsafe_allow_html=True)
            st.markdown(prompt)
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": now,
        })

        with st.chat_message("assistant"):
            with st.spinner("🤔 Recherche et analyse en cours..."):
                try:
                    active_pipeline = agent_pipeline if use_agent else pipeline
                    response = active_pipeline.query(
                        question=prompt,
                        where_filter=where_filter,
                        enterprise_tags=selected_enterprise_tags or None,
                        conversation_history=st.session_state.conversation_history,
                        n_documents=params["n_documents"],
                        n_chunks_per_doc=params["n_chunks_per_doc"],
                    )

                    if response.error:
                        st.error(f"❌ {response.error}")
                        logger.warning(
                            f"Query error: {response.error}",
                            extra={"event": "query_error", "question": prompt, "error": response.error}
                        )
                    else:
                        resp_time = datetime.now().strftime("%H:%M")
                        st.markdown(f'<div class="msg-time">{resp_time}</div>', unsafe_allow_html=True)
                        st.markdown(response.answer)

                        st.caption(
                            f"⏱️ {response.total_time:.1f}s "
                            f"| 📚 {len(response.sources)} sources "
                            f"| ✅ {len(response.cited_sources)} citées"
                            f"{'  | 🤖 Agent' if use_agent else ''}"
                        )

                        render_sources_block(response.sources)

                        # Log
                        answer_hash = hashlib.md5(response.answer.encode()).hexdigest()[:8]
                        query_logger.log_query(
                            question=prompt,
                            response=response,
                            enterprise_tags=selected_enterprise_tags or None,
                            depth=depth,
                            filter_nature=filter_nature or None,
                        )

                        logger.info(
                            f"Query OK: {prompt[:80]}",
                            extra={
                                "event": "query_success",
                                "question": prompt,
                                "total_time": response.total_time,
                                "n_sources": len(response.sources),
                                "n_cited": len(response.cited_sources),
                                "model": response.model,
                            }
                        )

                        # Stocker pour replay
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "timestamp": resp_time,
                            "sources": response.sources,
                            "answer_hash": answer_hash,
                            "question": prompt,
                        })
                        st.session_state.conversation_history.append({
                            "role": "user", "content": prompt
                        })
                        st.session_state.conversation_history.append({
                            "role": "assistant", "content": response.answer
                        })

                except Exception as e:
                    st.error(f"❌ Erreur inattendue : {e}")
                    logger.error(f"Query exception: {e}", exc_info=True)

    # Footer
    st.markdown("---")
    st.caption("🔒 RAG-DPO Assistant — Sources CNIL + documents internes — mistral-nemo 12B — 100% local")


main()
