"""
Page Dashboard — Observabilité et métriques du système RAG-DPO.

Affiche les stats détaillées, l'historique des queries, le feedback,
et les alertes. Permet l'export des logs.
"""
import streamlit as st
import json
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

logger = logging.getLogger(__name__)


def get_system():
    """Récupère le système RAG partagé."""
    if "rag_system" not in st.session_state:
        from app import init_rag_system
        st.session_state.rag_system = init_rag_system()
    return st.session_state.rag_system


def main():
    st.markdown("# 📊 Dashboard — Observabilité RAG-DPO")
    st.markdown("---")

    system = get_system()
    query_logger = system["query_logger"]
    alerter = system["alerter"]

    # ── Contrôles sidebar ──
    with st.sidebar:
        st.header("📊 Dashboard")
        hours = st.selectbox(
            "Période d'analyse",
            [1, 6, 12, 24, 48, 168, 720],
            index=3,
            format_func=lambda h: {
                1: "Dernière heure",
                6: "6 heures",
                12: "12 heures",
                24: "24 heures",
                48: "2 jours",
                168: "7 jours",
                720: "30 jours",
            }.get(h, f"{h}h"),
        )

        if st.button("🔄 Rafraîchir", use_container_width=True):
            st.rerun()

        st.markdown("---")

        # Vérification alertes
        stats = query_logger.get_stats(hours=hours)
        alerts = alerter.check_and_alert(stats)
        if alerts:
            st.error(f"🚨 {len(alerts)} alerte(s) active(s)")
        else:
            st.success("✅ Aucune alerte")

    # ── Métriques principales ──
    stats = query_logger.get_stats(hours=hours)

    st.subheader("📈 Métriques principales")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📝 Total queries", stats["total_queries"])
    col2.metric("⏱️ Temps moyen", f"{stats['avg_total_time']:.1f}s")
    col3.metric("❌ Erreurs", stats["error_count"],
                delta=f"{stats['error_rate']:.1f}%" if stats["error_rate"] > 0 else None,
                delta_color="inverse")
    col4.metric("✅ Taux citation", f"{stats['citation_rate']:.0f}%")
    if stats["satisfaction_rate"] is not None:
        col5.metric("😊 Satisfaction", f"{stats['satisfaction_rate']:.0f}%")
    else:
        col5.metric("😊 Satisfaction", "—")

    st.markdown("---")

    # ── Détail timing ──
    col_t1, col_t2, col_t3 = st.columns(3)
    col_t1.metric("🔍 Retrieval moyen", f"{stats['avg_retrieval_time']:.2f}s")
    col_t2.metric("🤖 Génération moyenne", f"{stats['avg_generation_time']:.2f}s")
    col_t3.metric("📊 Sources moy./query", f"{stats['avg_sources_total']:.1f}")

    st.markdown("---")

    # ── Tabs pour les sections détaillées ──
    tab_queries, tab_feedback, tab_alerts, tab_export = st.tabs([
        "📝 Queries récentes",
        "💬 Feedback",
        "🚨 Alertes",
        "📥 Export",
    ])

    # ── Tab Queries ──
    with tab_queries:
        recent = query_logger.get_recent_queries(n=50)
        if not recent:
            st.info("Aucune query enregistrée.")
        else:
            st.subheader(f"📝 {len(recent)} dernières queries")

            # Distribution temporelle
            all_entries = query_logger._read_jsonl(query_logger.queries_path)
            if all_entries:
                # Histogramme par heure (dernières 24h)
                now = datetime.now()
                hour_counts = Counter()
                for e in all_entries:
                    try:
                        ts = datetime.fromisoformat(e["timestamp"])
                        hours_ago = (now - ts).total_seconds() / 3600
                        if hours_ago <= 24:
                            hour_label = ts.strftime("%H:00")
                            hour_counts[hour_label] += 1
                    except (KeyError, ValueError):
                        pass

                if hour_counts:
                    # Construire les 24h en ordre
                    chart_data = {}
                    for h in range(24):
                        label = f"{h:02d}:00"
                        chart_data[label] = hour_counts.get(label, 0)
                    st.bar_chart(chart_data, height=200)

            # Table détaillée
            for q in recent[:20]:
                ts = q.get("timestamp", "")[:16].replace("T", " ")
                question = q.get("question", "")[:80]
                total_time = q.get("total_time", 0)
                n_cited = q.get("n_cited", 0)
                n_sources = q.get("n_sources", 0)
                error = q.get("error")
                depth_val = q.get("depth", "Normal")
                tags = q.get("enterprise_tags", [])

                # Indicateurs visuels
                time_color = "🟢" if total_time < 15 else ("🟡" if total_time < 30 else "🔴")
                cite_icon = "✅" if n_cited > 0 else "⚠️"
                err_icon = "❌ " if error else ""

                with st.expander(
                    f"{time_color} {ts} — {err_icon}{question}",
                    expanded=False
                ):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("⏱️ Temps", f"{total_time:.1f}s")
                    c2.metric(f"{cite_icon} Citées", f"{n_cited}/{n_sources}")
                    c3.caption(f"📥 {depth_val}")
                    if tags:
                        c4.caption(f"🏷️ {', '.join(tags)}")
                    if error:
                        st.error(f"Erreur: {error}")
                    preview = q.get("answer_preview", "")
                    if preview:
                        st.caption(f"📄 {preview}")

    # ── Tab Feedback ──
    with tab_feedback:
        feedback = query_logger.get_recent_feedback(n=50)
        if not feedback:
            st.info("Aucun feedback reçu.")
        else:
            st.subheader(f"💬 {len(feedback)} feedbacks récents")

            # Stats feedback
            fb_positive = sum(1 for f in feedback if f.get("rating", 0) > 0)
            fb_negative = sum(1 for f in feedback if f.get("rating", 0) < 0)
            fb_total = fb_positive + fb_negative

            col_fb1, col_fb2, col_fb3 = st.columns(3)
            col_fb1.metric("👍 Positifs", fb_positive)
            col_fb2.metric("👎 Négatifs", fb_negative)
            if fb_total > 0:
                col_fb3.metric("📊 Satisfaction", f"{fb_positive / fb_total * 100:.0f}%")

            st.markdown("---")

            for fb in feedback[:20]:
                ts = fb.get("timestamp", "")[:16].replace("T", " ")
                rating = fb.get("rating", 0)
                icon = "👍" if rating > 0 else "👎"
                question = fb.get("question", "")[:80]
                comment = fb.get("comment", "")

                line = f"{icon} **{ts}** — {question}"
                if comment:
                    line += f" — _\"{comment}\"_"
                st.markdown(line)

    # ── Tab Alertes ──
    with tab_alerts:
        st.subheader("🚨 Alertes")

        # Alertes actives
        current_alerts = alerter.check_and_alert(stats)
        if current_alerts:
            for alert in current_alerts:
                severity = alert.get("severity", "warning")
                if severity == "critical":
                    st.error(f"🔴 **{alert['type'].upper()}** — {alert['message']}")
                else:
                    st.warning(f"🟡 **{alert['type'].upper()}** — {alert['message']}")
        else:
            st.success("✅ Tous les indicateurs sont dans les seuils normaux.")

        st.markdown("---")

        # Seuils configurés
        thresholds = alerter.config.get("thresholds", {})
        st.caption("**Seuils configurés** (modifiables dans `configs/config.yaml`)")
        col_th1, col_th2 = st.columns(2)
        with col_th1:
            st.markdown(f"- Taux erreur max : **{thresholds.get('error_rate_pct', 20)}%**")
            st.markdown(f"- Temps réponse max : **{thresholds.get('avg_response_time_s', 60)}s**")
        with col_th2:
            st.markdown(f"- Satisfaction min : **{thresholds.get('satisfaction_below_pct', 50)}%**")
            st.markdown(f"- Taux zéro-citation max : **{thresholds.get('zero_citation_rate_pct', 30)}%**")

        # SMTP status
        smtp_config = alerter.config.get("smtp", {})
        if smtp_config.get("enabled"):
            st.info(f"📧 Alertes email activées → {', '.join(smtp_config.get('to_addrs', []))}")
        else:
            st.caption("📧 Alertes email désactivées — configurez SMTP dans `configs/config.yaml`")

        st.markdown("---")

        # Historique alertes
        recent_alerts = alerter.get_recent_alerts(n=20)
        if recent_alerts:
            st.subheader("📋 Historique des alertes")
            for a in recent_alerts[:10]:
                ts = a.get("timestamp", "")[:16].replace("T", " ")
                severity = a.get("severity", "warning")
                icon = "🔴" if severity == "critical" else "🟡"
                st.markdown(f"{icon} **{ts}** — [{a.get('type', '')}] {a.get('message', '')}")

    # ── Tab Export ──
    with tab_export:
        st.subheader("📥 Export des données")

        col_exp1, col_exp2, col_exp3 = st.columns(3)

        with col_exp1:
            # Export queries
            all_queries = query_logger._read_jsonl(query_logger.queries_path)
            if all_queries:
                queries_json = json.dumps(all_queries, ensure_ascii=False, indent=2)
                st.download_button(
                    "📝 Télécharger queries.json",
                    data=queries_json,
                    file_name=f"queries_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.caption(f"{len(all_queries)} queries")
            else:
                st.caption("Aucune query")

        with col_exp2:
            # Export feedback
            all_feedback = query_logger._read_jsonl(query_logger.feedback_path)
            if all_feedback:
                feedback_json = json.dumps(all_feedback, ensure_ascii=False, indent=2)
                st.download_button(
                    "💬 Télécharger feedback.json",
                    data=feedback_json,
                    file_name=f"feedback_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.caption(f"{len(all_feedback)} feedbacks")
            else:
                st.caption("Aucun feedback")

        with col_exp3:
            # Export alertes
            all_alerts_data = alerter.get_recent_alerts(n=1000)
            if all_alerts_data:
                alerts_json = json.dumps(all_alerts_data, ensure_ascii=False, indent=2)
                st.download_button(
                    "🚨 Télécharger alertes.json",
                    data=alerts_json,
                    file_name=f"alertes_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
                st.caption(f"{len(all_alerts_data)} alertes")
            else:
                st.caption("Aucune alerte")

        st.markdown("---")

        # Logs structurés
        app_log_path = project_root / "logs" / "app.jsonl"
        if app_log_path.exists():
            log_size = app_log_path.stat().st_size
            size_display = f"{log_size / 1024:.1f} KB" if log_size < 1024 * 1024 else f"{log_size / 1024 / 1024:.1f} MB"
            st.caption(f"📄 Logs structurés : `logs/app.jsonl` ({size_display})")

            with open(app_log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            st.download_button(
                "📄 Télécharger app.jsonl",
                data=log_content,
                file_name=f"app_logs_{datetime.now().strftime('%Y%m%d')}.jsonl",
                mime="application/jsonl",
                use_container_width=True,
            )


main()
