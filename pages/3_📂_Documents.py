"""
Page Documents Entreprise — Upload, indexation et gestion des documents internes.

Permet au DPO de :
- Uploader des fichiers (PDF, DOCX, XLSX, HTML, TXT)
- Choisir un tag métier (ou en créer un nouveau)
- Lancer l'indexation dans ChromaDB
- Voir les documents indexés
- Purger par tag ou tout l'index entreprise
"""
import streamlit as st
import json
import sys
import time
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)

TAGS_REGISTRY_PATH = project_root / "configs" / "enterprise_tags.json"
CHROMA_PATH = project_root / "data" / "vectordb" / "chromadb"
COLLECTION_NAME = "rag_dpo_chunks"
SUPPORTED_EXTENSIONS = {'.pdf', '.html', '.htm', '.docx', '.doc', '.xlsx', '.xls', '.odt', '.txt'}

# ── CSS ──
st.markdown("""
<style>
    .doc-card {
        padding: 0.6rem 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ff9800;
        background-color: rgba(255, 152, 0, 0.08);
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }
    .tag-badge {
        background: rgba(255, 165, 0, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-right: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──

def load_tags_registry() -> dict:
    """Charge le registre des tags."""
    if TAGS_REGISTRY_PATH.exists():
        with open(TAGS_REGISTRY_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"default_tags": [], "active_tags": [], "labels": {}}


def save_tags_registry(registry: dict):
    """Sauvegarde le registre des tags."""
    with open(TAGS_REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def get_collection():
    """Récupère la collection ChromaDB."""
    import chromadb
    from chromadb.config import Settings
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_collection(COLLECTION_NAME)


def get_enterprise_docs(collection) -> list:
    """Liste les docs entreprise indexés."""
    try:
        existing = collection.get(
            where={"source": "ENTREPRISE"},
            include=["metadatas"],
        )
        docs = {}
        for mid, meta in zip(existing.get('ids', []), existing.get('metadatas', [])):
            doc_id = meta.get('document_id', '')
            if doc_id not in docs:
                doc_tags = [k[4:] for k, v in meta.items() if k.startswith('tag_') and v is True]
                docs[doc_id] = {
                    'document_id': doc_id,
                    'document_path': meta.get('document_path', ''),
                    'file_type': meta.get('file_type', ''),
                    'title': meta.get('title', ''),
                    'tags': sorted(doc_tags),
                    'ingested_at': meta.get('ingested_at', ''),
                    'n_chunks': 0,
                }
            docs[doc_id]['n_chunks'] += 1
        return sorted(docs.values(), key=lambda x: x.get('ingested_at', ''), reverse=True)
    except Exception as e:
        logger.error(f"Erreur listing enterprise docs: {e}")
        return []


def get_collection_stats(collection) -> dict:
    """Stats de la collection."""
    total = collection.count()
    try:
        ent = collection.get(where={"source": "ENTREPRISE"}, include=[])
        n_ent = len(ent.get('ids', []))
    except Exception:
        n_ent = 0
    return {
        'total': total,
        'cnil': total - n_ent,
        'enterprise': n_ent,
    }


def refresh_active_tags(collection):
    """Recalcule les tags actifs depuis ChromaDB."""
    registry = load_tags_registry()
    try:
        existing = collection.get(where={"source": "ENTREPRISE"}, include=["metadatas"])
        active = set()
        for meta in existing.get('metadatas', []):
            for key, val in meta.items():
                if key.startswith('tag_') and val is True:
                    active.add(key[4:])
        registry['active_tags'] = sorted(active)
        save_tags_registry(registry)
    except Exception as e:
        logger.error(f"Erreur refresh tags: {e}")


# ── Page principale ──

def main():
    st.markdown("# 📂 Documents Entreprise")
    st.markdown(
        "Ajoutez vos **documents internes** (politiques, registres, contrats, PIA…) "
        "pour enrichir les réponses du RAG. Les sources **CNIL prévalent toujours**."
    )
    st.markdown("---")

    # Onglets
    tab_upload, tab_docs, tab_manage = st.tabs([
        "📤 Importer", "📋 Documents indexés", "⚙️ Gestion"
    ])

    # ── Tab 1 : Upload + Indexation ──
    with tab_upload:
        st.subheader("📤 Importer des documents")

        # Upload
        uploaded_files = st.file_uploader(
            "Glissez-déposez vos fichiers",
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'html', 'htm', 'odt', 'txt'],
            accept_multiple_files=True,
            help="Formats supportés : PDF, DOCX, XLSX, HTML, TXT, ODT"
        )

        if uploaded_files:
            st.info(f"📎 {len(uploaded_files)} fichier(s) sélectionné(s)")
            for f in uploaded_files:
                size_kb = len(f.getvalue()) / 1024
                st.caption(f"  • {f.name} ({size_kb:.0f} KB)")

        st.markdown("---")

        # Choix du tag
        st.subheader("🏷️ Tag métier")
        st.caption(
            "Le tag permet de catégoriser vos documents et de filtrer "
            "dans le Chat quelles sources internes consulter."
        )

        registry = load_tags_registry()
        existing_labels = registry.get('labels', {})
        # Tous les tags connus (actifs + ceux qui ont un label)
        all_known_tags = sorted(set(
            list(existing_labels.keys()) +
            registry.get('active_tags', [])
        ))

        col_tag, col_new = st.columns([2, 1])

        with col_tag:
            tag_options = ["— Choisir un tag existant —"] + [
                f"{existing_labels.get(t, t)}  ({t})" for t in all_known_tags
            ]
            selected_tag_display = st.selectbox("Tag existant", tag_options)

        with col_new:
            new_tag = st.text_input(
                "Ou créer un nouveau tag",
                placeholder="ex: service_rh",
                help="Identifiant court, sans espaces ni accents (snake_case)"
            )
            new_tag_label = st.text_input(
                "Label affiché dans l'UI",
                placeholder="ex: 📋 Service RH",
                help="Nom lisible affiché dans le sélecteur du Chat"
            )

        # Déterminer le tag final
        final_tag = None
        if new_tag.strip():
            final_tag = new_tag.strip().lower().replace(' ', '_').replace('-', '_')
        elif selected_tag_display != "— Choisir un tag existant —":
            # Extraire le tag_id depuis "Label  (tag_id)"
            final_tag = selected_tag_display.split('(')[-1].rstrip(')')

        if final_tag:
            label_display = existing_labels.get(final_tag, new_tag_label or final_tag)
            st.success(f"🏷️ Tag sélectionné : **{label_display}** (`{final_tag}`)")

        st.markdown("---")

        # Bouton d'indexation
        can_index = uploaded_files and final_tag
        if st.button(
            "🚀 Indexer les documents",
            type="primary",
            disabled=not can_index,
            use_container_width=True,
        ):
            _run_ingestion(uploaded_files, final_tag, new_tag_label)

    # ── Tab 2 : Documents indexés ──
    with tab_docs:
        st.subheader("📋 Documents internes indexés")

        try:
            collection = get_collection()
            docs = get_enterprise_docs(collection)
            stats = get_collection_stats(collection)
        except Exception as e:
            st.error(f"❌ Erreur accès VectorDB : {e}")
            return

        # Stats
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Total chunks", f"{stats['total']:,}")
        c2.metric("🏛️ CNIL", f"{stats['cnil']:,}")
        c3.metric("📋 Entreprise", f"{stats['enterprise']:,}")

        if not docs:
            st.info(
                "ℹ️ Aucun document entreprise indexé. "
                "Utilisez l'onglet **📤 Importer** pour ajouter vos documents."
            )
        else:
            # Filtre par tag
            all_tags = sorted(set(t for d in docs for t in d['tags']))
            if all_tags:
                filter_tag = st.multiselect("Filtrer par tag", all_tags)
            else:
                filter_tag = []

            for doc in docs:
                if filter_tag and not any(t in filter_tag for t in doc['tags']):
                    continue

                tags_html = ' '.join(
                    f'<span class="tag-badge">🏷️ {t}</span>'
                    for t in doc['tags']
                ) if doc['tags'] else '<span style="opacity:0.5">pas de tag</span>'

                ingested_str = ''
                if doc['ingested_at']:
                    try:
                        dt = datetime.fromisoformat(doc['ingested_at'])
                        ingested_str = dt.strftime('%d/%m/%Y %H:%M')
                    except Exception:
                        ingested_str = doc['ingested_at'][:16]

                st.markdown(f"""<div class="doc-card">
                    <strong>📄 {doc['title']}</strong>
                    <span style="float:right;font-size:0.8em;opacity:0.7">{doc['file_type'].upper()} • {doc['n_chunks']} chunks</span>
                    <br><span style="font-size:0.85em">{tags_html}</span>
                    <span style="float:right;font-size:0.8em;opacity:0.5">{ingested_str}</span>
                </div>""", unsafe_allow_html=True)

    # ── Tab 3 : Gestion (purge, tags) ──
    with tab_manage:
        st.subheader("⚙️ Gestion des tags et purge")

        try:
            collection = get_collection()
        except Exception as e:
            st.error(f"❌ Erreur accès VectorDB : {e}")
            return

        # Gestion des labels de tags
        st.markdown("#### 🏷️ Labels des tags")
        st.caption(
            "Personnalisez les noms affichés dans le sélecteur du Chat. "
            "Les tags sont les identifiants techniques, les labels sont les noms lisibles."
        )

        registry = load_tags_registry()
        labels = registry.get('labels', {})
        active_tags = registry.get('active_tags', [])
        default_tags = registry.get('default_tags', [])

        # Afficher les labels éditables pour tous les tags connus
        all_tags_for_labels = sorted(set(list(labels.keys()) + active_tags))

        if all_tags_for_labels:
            updated_labels = {}
            for tag in all_tags_for_labels:
                current_label = labels.get(tag, tag)
                col_id, col_label, col_default = st.columns([2, 3, 1])
                with col_id:
                    st.code(tag, language=None)
                with col_label:
                    new_label = st.text_input(
                        f"Label pour {tag}",
                        value=current_label,
                        key=f"label_{tag}",
                        label_visibility="collapsed",
                    )
                    updated_labels[tag] = new_label
                with col_default:
                    is_default = st.checkbox(
                        "Défaut",
                        value=tag in default_tags,
                        key=f"default_{tag}",
                        help="Pré-sélectionné dans le Chat"
                    )
                    if is_default and tag not in default_tags:
                        default_tags.append(tag)
                    elif not is_default and tag in default_tags:
                        default_tags.remove(tag)

            if st.button("💾 Sauvegarder les labels", use_container_width=True):
                registry['labels'] = updated_labels
                registry['default_tags'] = default_tags
                save_tags_registry(registry)
                st.success("✅ Labels et défauts sauvegardés")
                st.rerun()
        else:
            st.info("Aucun tag connu. Importez des documents pour créer des tags.")

        st.markdown("---")

        # Purge par tag
        st.markdown("#### 🗑️ Purge")

        if active_tags:
            col_purge_tag, col_purge_btn = st.columns([3, 1])
            with col_purge_tag:
                purge_tag = st.selectbox(
                    "Purger les documents d'un tag",
                    ["— Choisir —"] + active_tags
                )
            with col_purge_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Purger ce tag", type="secondary"):
                    if purge_tag != "— Choisir —":
                        _purge_tag(collection, purge_tag)

        st.markdown("")

        # Purge totale
        st.warning(
            "⚠️ **Purge totale** — Supprime TOUS les documents entreprise "
            "(sans affecter la base CNIL)."
        )
        col_confirm, col_purge_all = st.columns([3, 1])
        with col_confirm:
            confirm_purge = st.text_input(
                "Tapez PURGER pour confirmer",
                key="confirm_purge",
                label_visibility="collapsed",
                placeholder="Tapez PURGER pour confirmer"
            )
        with col_purge_all:
            if st.button("🗑️ Purge totale", type="secondary"):
                if confirm_purge == "PURGER":
                    _purge_all(collection)
                else:
                    st.error("Tapez PURGER pour confirmer")


def _run_ingestion(uploaded_files, tag: str, tag_label: str = ""):
    """Lance l'indexation des fichiers uploadés."""
    from src.processing.ingest_enterprise import (
        extract_and_chunk, make_document_id, detect_file_type,
        update_tags_registry,
    )
    from src.utils.embedding_provider import EmbeddingProvider

    progress = st.progress(0, text="Préparation...")

    try:
        collection = get_collection()
        embedding_provider = EmbeddingProvider(
            cache_dir=str(project_root / "models" / "huggingface" / "hub"),
        )
    except Exception as e:
        st.error(f"❌ Erreur init : {e}")
        return

    # Récupérer les docs déjà indexés
    existing_docs = set()
    try:
        existing = collection.get(where={"source": "ENTREPRISE"}, include=[])
        for mid in (existing.get('ids') or []):
            doc_id = mid.rsplit('_', 1)[0]
            existing_docs.add(doc_id)
    except Exception:
        pass

    # Sauvegarder les fichiers uploadés dans un dossier temporaire
    tmp_dir = Path(tempfile.mkdtemp(prefix="ragdpo_"))
    saved_files = []
    for uf in uploaded_files:
        tmp_path = tmp_dir / uf.name
        tmp_path.write_bytes(uf.getvalue())
        saved_files.append(tmp_path)

    stats = {'processed': 0, 'skipped': 0, 'errors': 0, 'chunks': 0}
    all_chunks = []
    n_files = len(saved_files)

    for i, file_path in enumerate(saved_files):
        progress.progress((i + 1) / (n_files + 2), text=f"Extraction {file_path.name}...")

        doc_id = make_document_id(file_path)
        if doc_id in existing_docs:
            st.caption(f"♻️ Déjà indexé : {file_path.name}")
            stats['skipped'] += 1
            continue

        chunks = extract_and_chunk(file_path)
        if not chunks:
            st.caption(f"⚠️ Pas de contenu : {file_path.name}")
            stats['errors'] += 1
            continue

        file_type = detect_file_type(file_path)

        for ci, chunk in enumerate(chunks):
            text = chunk.get('text', '').strip()
            heading = chunk.get('heading', '').strip()

            if not text or len(text.split()) < 10:
                continue

            full_text = f"[{heading}] {text}" if heading and not text.startswith(f"[{heading}]") else text

            metadata = {
                'document_id': doc_id,
                'document_path': str(file_path),
                'document_nature': 'INTERNE',
                'chunk_nature': 'INTERNE',
                'chunk_index': 'OPERATIONNEL',
                'heading': heading[:200] if heading else '',
                'page_info': chunk.get('page_info', ''),
                'confidence': 0.8,
                'method': 'structural',
                'word_count': len(text.split()),
                'sectors': '',
                'file_type': file_type,
                'title': file_path.stem[:300],
                'source': 'ENTREPRISE',
                'source_type': file_type,
                'is_priority': False,
                'source_url': str(file_path),
                'parent_url': '',
                'ingested_at': datetime.now().isoformat(),
                f'tag_{tag}': True,
            }

            all_chunks.append({
                'id': f"{doc_id}_{ci}",
                'text': full_text,
                'metadata': metadata,
            })

        stats['processed'] += 1

    if not all_chunks:
        progress.empty()
        if stats['skipped'] > 0:
            st.info(f"♻️ Tous les fichiers étaient déjà indexés ({stats['skipped']})")
        else:
            st.warning("⚠️ Aucun chunk extractible des fichiers fournis.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # Embedding + indexation
    progress.progress(0.9, text=f"Embedding de {len(all_chunks)} chunks...")

    batch_size = 50
    indexed = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        ids = [c['id'] for c in batch]
        texts = [c['text'] for c in batch]
        metadatas = [c['metadata'] for c in batch]

        try:
            embeddings = embedding_provider.embed(texts)
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            indexed += len(batch)
        except Exception as e:
            st.error(f"❌ Erreur indexation : {e}")
            logger.error(f"Ingestion batch error: {e}", exc_info=True)

    stats['chunks'] = indexed

    # MAJ registre des tags
    update_tags_registry([tag])
    if tag_label.strip():
        registry = load_tags_registry()
        registry['labels'][tag] = tag_label.strip()
        save_tags_registry(registry)

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)
    progress.progress(1.0, text="Terminé !")

    # Résumé
    st.success(
        f"✅ **Indexation terminée** — "
        f"{stats['processed']} fichier(s), {stats['chunks']} chunks indexés, "
        f"tag `{tag}`"
    )
    if stats['skipped']:
        st.info(f"♻️ {stats['skipped']} fichier(s) déjà indexé(s) (ignorés)")
    if stats['errors']:
        st.warning(f"⚠️ {stats['errors']} fichier(s) sans contenu extractible")

    # Invalider le cache RAG pour refresh chunk_count
    if hasattr(st, 'cache_resource'):
        try:
            from app import init_rag_system
            init_rag_system.clear()
        except Exception:
            pass

    logger.info(
        f"Enterprise ingestion via UI: {stats['processed']} files, {stats['chunks']} chunks, tag={tag}",
        extra={"event": "enterprise_ingestion_ui", "tag": tag, "chunks": stats['chunks']}
    )


def _purge_tag(collection, tag: str):
    """Purge les docs d'un tag spécifique."""
    tag_field = f'tag_{tag}'
    try:
        existing = collection.get(
            where={"$and": [{"source": "ENTREPRISE"}, {tag_field: True}]},
            include=[],
        )
        ids = existing.get('ids', [])
        if not ids:
            st.info(f"Aucun chunk avec le tag `{tag}`.")
            return

        for i in range(0, len(ids), 5000):
            collection.delete(ids=ids[i:i + 5000])

        refresh_active_tags(collection)
        st.success(f"✅ {len(ids)} chunks supprimés (tag: `{tag}`)")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Erreur purge : {e}")


def _purge_all(collection):
    """Purge tous les docs entreprise."""
    try:
        existing = collection.get(where={"source": "ENTREPRISE"}, include=[])
        ids = existing.get('ids', [])
        if not ids:
            st.info("Aucun document entreprise à supprimer.")
            return

        for i in range(0, len(ids), 5000):
            collection.delete(ids=ids[i:i + 5000])

        refresh_active_tags(collection)
        st.success(f"✅ {len(ids)} chunks entreprise supprimés.")
        st.rerun()
    except Exception as e:
        st.error(f"❌ Erreur purge : {e}")


main()
