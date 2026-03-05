"""
Page À propos — Crédits, architecture et liens du projet RAG-DPO.
"""
import streamlit as st

st.set_page_config(page_title="À propos — RAG-DPO", page_icon="ℹ️", layout="wide")

st.markdown("""
<div style="text-align: center; padding: 1.5rem 0 0.5rem;">
    <h1 style="color: #1f77b4;">ℹ️ À propos de RAG-DPO</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Présentation ──
st.markdown("""
### 🎯 Le projet

**RAG-DPO** est un assistant intelligent dédié aux **Délégués à la Protection des Données** (DPO).  
Il s'appuie sur un pipeline **RAG** (Retrieval-Augmented Generation) 100 % local pour répondre
aux questions liées au **RGPD**, à la **protection des données personnelles** et aux **recommandations de la CNIL**.

> *Aucune donnée ne quitte votre machine.  
> Aucun appel API cloud. Tout tourne en local.*
""")

st.markdown("---")

# ── Architecture ──
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🏗️ Architecture technique
    
    | Composant | Technologie |
    |-----------|-------------|
    | **LLM** | Ollama — Mistral-Nemo 12B |
    | **Embeddings** | BAAI/bge-m3 (1024 dims) |
    | **Reranker** | Jina Reranker v2 (cross-encoder) |
    | **VectorDB** | ChromaDB (PersistentClient) |
    | **Recherche** | Hybride BM25 + sémantique |
    | **Agent** | LangGraph (classification d'intent) |
    | **Interface** | Streamlit multipage |
    | **Déploiement** | Docker Compose (GPU Ollama + CPU App) |
    """)

with col2:
    st.markdown("""
    ### ✨ Fonctionnalités
    
    - 🔍 **Recherche hybride** — BM25 lexical + similarité cosinus
    - 🧠 **Reranking cross-encoder** — pertinence fine des résultats
    - 🏷️ **Classification d'intent** — adaptation du pipeline à la question
    - 📊 **Dashboard d'observabilité** — métriques, feedback, alertes
    - 📂 **Documents entreprise** — upload et indexation de vos documents
    - 🐳 **Docker-ready** — déploiement one-click avec GPU support
    - 🔒 **100 % local** — aucune fuite de données vers le cloud
    """)

st.markdown("---")

# ── Stack complète ──
with st.expander("🐍 Stack Python complète", expanded=False):
    st.markdown("""
    ```
    Python 3.11 · PyTorch (CPU/CUDA) · Transformers · Sentence-Transformers
    ChromaDB · LangGraph · LangChain-Core · Ollama SDK
    Streamlit · Pandas · BeautifulSoup4 · Scrapy · Playwright
    PyMuPDF · pdfplumber · python-docx · openpyxl
    ```
    """)

st.markdown("---")

# ── Crédits ──
st.markdown("""
### 👨‍💻 Auteur

**MatJoss**  

Projet développé dans le cadre de l'exploration des architectures RAG appliquées
à la conformité RGPD et à l'assistance aux DPO.

<a href="https://github.com/MatJoss/RAG-DPO" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-RAG--DPO-181717?style=for-the-badge&logo=github" alt="GitHub"/>
</a>

""", unsafe_allow_html=True)

st.markdown("---")

# ── Licence ──
st.caption("""
📄 Les données CNIL sont issues de [cnil.fr](https://www.cnil.fr) et restent la propriété de la CNIL.  
Ce projet est un outil d'aide à la décision et ne constitue pas un avis juridique.
""")
