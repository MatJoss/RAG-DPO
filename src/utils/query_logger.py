"""
Query Logger — Structured logging for RAG pipeline queries.

Logs every query with timing, sources, and metadata to a JSONL file
for observability and analytics. Designed for zero-impact append.

Usage:
    from src.utils.query_logger import QueryLogger
    
    logger = QueryLogger()
    logger.log_query(question, response)
    stats = logger.get_stats()
"""
import json
import hashlib
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# ── Constantes ───────────────────────────────────────────────
from src.utils.paths import LOGS_DIR
DEFAULT_LOG_DIR = LOGS_DIR
QUERIES_FILE = "queries.jsonl"
FEEDBACK_FILE = "feedback.jsonl"
MAX_LOG_SIZE_MB = 10  # Rotation après 10 MB


class QueryLogger:
    """Structured query logger for RAG pipeline observability."""
    
    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.queries_path = self.log_dir / QUERIES_FILE
        self.feedback_path = self.log_dir / FEEDBACK_FILE
    
    # ── Logging ──────────────────────────────────────────────
    
    def log_query(
        self,
        question: str,
        response: Any,  # RAGResponse
        enterprise_tags: Optional[List[str]] = None,
        depth: str = "Normal",
        filter_nature: Optional[List[str]] = None,
    ):
        """Log a RAG query with full metadata.
        
        Args:
            question: User question
            response: RAGResponse object from pipeline
            enterprise_tags: Enterprise tags used for filtering
            depth: Search depth setting
            filter_nature: Document nature filter
        """
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer_preview": (response.answer or "")[:200],
                "answer_hash": hashlib.md5((response.answer or "").encode()).hexdigest()[:8],
                "retrieval_time": round(response.retrieval_time, 3),
                "generation_time": round(response.generation_time, 3),
                "total_time": round(response.total_time, 3),
                "n_sources": len(response.sources) if response.sources else 0,
                "n_cited": len(response.cited_sources) if response.cited_sources else 0,
                "model": response.model or "",
                "error": response.error,
                "enterprise_tags": enterprise_tags or [],
                "depth": depth,
                "filter_nature": filter_nature or [],
            }
            
            self._append_jsonl(self.queries_path, entry)
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur log query: {e}")
    
    def log_feedback(
        self,
        question: str,
        answer_hash: str,
        rating: int,  # 1 = 👍, -1 = 👎
        comment: str = "",
    ):
        """Log user feedback on a response.
        
        Args:
            question: Original question
            answer_hash: Hash of the answer (for matching)
            rating: 1 for positive, -1 for negative
            comment: Optional user comment
        """
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer_hash": answer_hash,
                "rating": rating,
                "comment": comment,
            }
            
            self._append_jsonl(self.feedback_path, entry)
            
        except Exception as e:
            logger.warning(f"⚠️  Erreur log feedback: {e}")
    
    # ── Analytics ─────────────────────────────────────────────
    
    def get_stats(self, hours: int = 24) -> Dict:
        """Get query statistics for the last N hours.
        
        Returns:
            Dict with stats: total_queries, avg_time, error_rate, etc.
        """
        entries = self._read_jsonl(self.queries_path)
        feedback = self._read_jsonl(self.feedback_path)
        
        if not entries:
            return self._empty_stats()
        
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # All-time stats
        total = len(entries)
        all_times = [e.get("total_time", 0) for e in entries if e.get("total_time")]
        all_errors = [e for e in entries if e.get("error")]
        all_cited = [e.get("n_cited", 0) for e in entries]
        all_sources = [e.get("n_sources", 0) for e in entries]
        
        # Recent stats (last N hours)
        recent = []
        for e in entries:
            try:
                ts = datetime.fromisoformat(e["timestamp"])
                if ts >= cutoff:
                    recent.append(e)
            except (KeyError, ValueError):
                pass
        
        recent_times = [e.get("total_time", 0) for e in recent if e.get("total_time")]
        recent_errors = [e for e in recent if e.get("error")]
        
        # Feedback stats
        positive = sum(1 for f in feedback if f.get("rating", 0) > 0)
        negative = sum(1 for f in feedback if f.get("rating", 0) < 0)
        
        # Queries without citations (potential issues)
        zero_cited = [e for e in entries if e.get("n_cited", 0) == 0 and not e.get("error")]
        
        return {
            # All-time
            "total_queries": total,
            "avg_total_time": round(sum(all_times) / len(all_times), 2) if all_times else 0,
            "avg_retrieval_time": round(
                sum(e.get("retrieval_time", 0) for e in entries) / total, 2
            ) if total else 0,
            "avg_generation_time": round(
                sum(e.get("generation_time", 0) for e in entries) / total, 2
            ) if total else 0,
            "error_count": len(all_errors),
            "error_rate": round(len(all_errors) / total * 100, 1) if total else 0,
            "avg_sources_cited": round(sum(all_cited) / total, 1) if total else 0,
            "avg_sources_total": round(sum(all_sources) / total, 1) if total else 0,
            "citation_rate": round(
                sum(1 for c in all_cited if c > 0) / total * 100, 1
            ) if total else 0,
            
            # Recent (last N hours)
            "recent_queries": len(recent),
            "recent_avg_time": round(sum(recent_times) / len(recent_times), 2) if recent_times else 0,
            "recent_errors": len(recent_errors),
            
            # Feedback
            "feedback_positive": positive,
            "feedback_negative": negative,
            "feedback_total": positive + negative,
            "satisfaction_rate": round(
                positive / (positive + negative) * 100, 1
            ) if (positive + negative) > 0 else None,
            
            # Issues
            "zero_citation_queries": len(zero_cited),
            "recent_zero_citations": [
                {"question": e["question"][:100], "timestamp": e["timestamp"]}
                for e in zero_cited[-5:]  # Last 5
            ],
        }
    
    def get_recent_queries(self, n: int = 10) -> List[Dict]:
        """Get the N most recent queries.
        
        Returns:
            List of query entries (most recent first)
        """
        entries = self._read_jsonl(self.queries_path)
        return list(reversed(entries[-n:]))
    
    def get_recent_feedback(self, n: int = 10) -> List[Dict]:
        """Get the N most recent feedback entries.
        
        Returns:
            List of feedback entries (most recent first)
        """
        entries = self._read_jsonl(self.feedback_path)
        return list(reversed(entries[-n:]))
    
    # ── Internal ─────────────────────────────────────────────
    
    def _append_jsonl(self, path: Path, entry: Dict):
        """Append a JSON entry to a JSONL file with rotation."""
        # Check rotation
        if path.exists() and path.stat().st_size > MAX_LOG_SIZE_MB * 1024 * 1024:
            self._rotate(path)
        
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def _read_jsonl(self, path: Path) -> List[Dict]:
        """Read all entries from a JSONL file."""
        if not path.exists():
            return []
        
        entries = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries
    
    def _rotate(self, path: Path):
        """Rotate log file: rename current to .1, delete old .1."""
        rotated = path.with_suffix(path.suffix + '.1')
        if rotated.exists():
            rotated.unlink()
        path.rename(rotated)
        logger.info(f"🔄 Log rotation: {path.name} → {rotated.name}")
    
    def _empty_stats(self) -> Dict:
        """Return empty stats structure."""
        return {
            "total_queries": 0,
            "avg_total_time": 0,
            "avg_retrieval_time": 0,
            "avg_generation_time": 0,
            "error_count": 0,
            "error_rate": 0,
            "avg_sources_cited": 0,
            "avg_sources_total": 0,
            "citation_rate": 0,
            "recent_queries": 0,
            "recent_avg_time": 0,
            "recent_errors": 0,
            "feedback_positive": 0,
            "feedback_negative": 0,
            "feedback_total": 0,
            "satisfaction_rate": None,
            "zero_citation_queries": 0,
            "recent_zero_citations": [],
        }
