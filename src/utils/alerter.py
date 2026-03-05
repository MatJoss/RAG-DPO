"""
Alerter — Threshold-based alerting for RAG-DPO observability.

Checks query stats against configurable thresholds and sends alerts
via SMTP email or logs them locally as fallback.

Configuration in configs/config.yaml under `observability.alerting`.
SMTP is disabled by default — configure it to enable email alerts.

Usage:
    from src.utils.alerter import Alerter, load_alert_config
    
    config = load_alert_config()
    alerter = Alerter(config)
    alerter.check_and_alert(stats)
"""
import json
import logging
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils.paths import CONFIG_PATH as _CONFIG_PATH, LOGS_DIR as _LOGS_DIR

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = _CONFIG_PATH
ALERT_LOG_FILE = "alerts.jsonl"


def load_alert_config(config_path: Optional[Path] = None) -> Dict:
    """Load alerting config from config.yaml.
    
    Returns:
        Dict with alerting configuration, or defaults if not configured.
    """
    config_path = config_path or DEFAULT_CONFIG_PATH
    
    defaults = {
        "enabled": True,
        "thresholds": {
            "error_rate_pct": 20.0,
            "avg_response_time_s": 60.0,
            "satisfaction_below_pct": 50.0,
            "zero_citation_rate_pct": 30.0,
        },
        "check_interval_minutes": 60,
        "smtp": {
            "enabled": False,
            "host": "",
            "port": 587,
            "use_tls": True,
            "username": "",
            "password": "",
            "from_addr": "",
            "to_addrs": [],
        },
    }
    
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        
        obs_config = full_config.get("observability", {}).get("alerting", {})
        
        # Merge with defaults (config overrides defaults)
        merged = defaults.copy()
        if "enabled" in obs_config:
            merged["enabled"] = obs_config["enabled"]
        if "thresholds" in obs_config:
            merged["thresholds"].update(obs_config["thresholds"])
        if "check_interval_minutes" in obs_config:
            merged["check_interval_minutes"] = obs_config["check_interval_minutes"]
        if "smtp" in obs_config:
            merged["smtp"].update(obs_config["smtp"])
        
        return merged
        
    except Exception as e:
        logger.debug(f"Config alerting non trouvée, utilisation des défauts: {e}")
        return defaults


class Alerter:
    """Threshold-based alerter with SMTP email and local log fallback."""
    
    def __init__(self, config: Optional[Dict] = None, log_dir: Optional[Path] = None):
        self.config = config or load_alert_config()
        self.log_dir = log_dir or _LOGS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.alert_log_path = self.log_dir / ALERT_LOG_FILE
        self._last_check: Optional[datetime] = None
    
    def check_and_alert(self, stats: Dict) -> List[Dict]:
        """Check stats against thresholds and fire alerts if needed.
        
        Args:
            stats: Stats dict from QueryLogger.get_stats()
            
        Returns:
            List of triggered alert dicts (empty if none)
        """
        if not self.config.get("enabled", True):
            return []
        
        if stats.get("total_queries", 0) == 0:
            return []
        
        alerts = []
        thresholds = self.config.get("thresholds", {})
        
        # Check error rate
        error_rate = stats.get("error_rate", 0)
        threshold_error = thresholds.get("error_rate_pct", 20.0)
        if error_rate > threshold_error:
            alerts.append({
                "type": "error_rate",
                "severity": "critical" if error_rate > threshold_error * 2 else "warning",
                "message": f"Taux d'erreur élevé : {error_rate:.1f}% (seuil: {threshold_error}%)",
                "value": error_rate,
                "threshold": threshold_error,
            })
        
        # Check average response time
        avg_time = stats.get("avg_total_time", 0)
        threshold_time = thresholds.get("avg_response_time_s", 60.0)
        if avg_time > threshold_time:
            alerts.append({
                "type": "slow_response",
                "severity": "warning",
                "message": f"Temps de réponse moyen élevé : {avg_time:.1f}s (seuil: {threshold_time}s)",
                "value": avg_time,
                "threshold": threshold_time,
            })
        
        # Check satisfaction rate (only if feedback exists)
        satisfaction = stats.get("satisfaction_rate")
        threshold_satisfaction = thresholds.get("satisfaction_below_pct", 50.0)
        if satisfaction is not None and satisfaction < threshold_satisfaction:
            alerts.append({
                "type": "low_satisfaction",
                "severity": "warning",
                "message": f"Satisfaction faible : {satisfaction:.0f}% (seuil: {threshold_satisfaction}%)",
                "value": satisfaction,
                "threshold": threshold_satisfaction,
            })
        
        # Check zero-citation rate
        total = stats.get("total_queries", 1)
        zero_citations = stats.get("zero_citation_queries", 0)
        zero_rate = (zero_citations / total * 100) if total > 0 else 0
        threshold_zero = thresholds.get("zero_citation_rate_pct", 30.0)
        if zero_rate > threshold_zero:
            alerts.append({
                "type": "zero_citations",
                "severity": "warning",
                "message": f"Taux de questions sans citation : {zero_rate:.0f}% (seuil: {threshold_zero}%)",
                "value": zero_rate,
                "threshold": threshold_zero,
            })
        
        # Fire alerts
        if alerts:
            self._fire_alerts(alerts, stats)
        
        self._last_check = datetime.now()
        return alerts
    
    def _fire_alerts(self, alerts: List[Dict], stats: Dict):
        """Send alerts via configured channels."""
        # Always log locally
        self._log_alerts(alerts)
        
        # Try SMTP if configured
        smtp_config = self.config.get("smtp", {})
        if smtp_config.get("enabled", False):
            try:
                self._send_email(alerts, stats)
            except Exception as e:
                logger.warning(
                    f"⚠️  Échec envoi email alerte: {e}",
                    extra={"event": "alert_email_failed", "error": str(e)}
                )
        
        # Log to application logger
        for alert in alerts:
            severity = alert.get("severity", "warning")
            log_fn = logger.critical if severity == "critical" else logger.warning
            log_fn(
                f"🚨 ALERTE [{alert['type']}]: {alert['message']}",
                extra={
                    "event": "alert_triggered",
                    "alert_type": alert["type"],
                    "alert_severity": severity,
                    "component": "alerter",
                }
            )
    
    def _log_alerts(self, alerts: List[Dict]):
        """Append alerts to local JSONL log."""
        try:
            with open(self.alert_log_path, 'a', encoding='utf-8') as f:
                for alert in alerts:
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        **alert,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Erreur écriture alert log: {e}")
    
    def _send_email(self, alerts: List[Dict], stats: Dict):
        """Send alert email via SMTP."""
        smtp = self.config["smtp"]
        
        if not smtp.get("host") or not smtp.get("to_addrs"):
            logger.debug("SMTP non configuré (host/to_addrs manquants), skip email")
            return
        
        # Build email
        subject = f"🚨 RAG-DPO Alerte — {len(alerts)} problème(s) détecté(s)"
        
        body_lines = [
            "RAG-DPO — Rapport d'alerte",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "=" * 50,
            "",
        ]
        
        for alert in alerts:
            severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
            body_lines.append(f"{severity_icon} [{alert['type'].upper()}] {alert['message']}")
        
        body_lines.extend([
            "",
            "=" * 50,
            "Statistiques actuelles:",
            f"  Total queries: {stats.get('total_queries', 0)}",
            f"  Temps moyen: {stats.get('avg_total_time', 0):.1f}s",
            f"  Taux erreur: {stats.get('error_rate', 0):.1f}%",
            f"  Taux citation: {stats.get('citation_rate', 0):.0f}%",
            f"  Satisfaction: {stats.get('satisfaction_rate', 'N/A')}",
            "",
            "— RAG-DPO Alerter (automatique)",
        ])
        
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = smtp.get("from_addr", smtp.get("username", ""))
        msg["To"] = ", ".join(smtp["to_addrs"])
        msg.attach(MIMEText("\n".join(body_lines), "plain", "utf-8"))
        
        # Send
        with smtplib.SMTP(smtp["host"], smtp.get("port", 587)) as server:
            if smtp.get("use_tls", True):
                server.starttls()
            if smtp.get("username") and smtp.get("password"):
                server.login(smtp["username"], smtp["password"])
            server.send_message(msg)
        
        logger.info(
            f"📧 Alerte email envoyée à {msg['To']}",
            extra={"event": "alert_email_sent", "component": "alerter"}
        )
    
    def get_recent_alerts(self, n: int = 20) -> List[Dict]:
        """Read the N most recent alerts from the log.
        
        Returns:
            List of alert entries (most recent first)
        """
        if not self.alert_log_path.exists():
            return []
        
        entries = []
        try:
            with open(self.alert_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except Exception:
            pass
        
        return list(reversed(entries[-n:]))
