# answer_analytics.py
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict
import sqlite3


class AnswerAnalytics:
    def __init__(self, db_path: str = "answer_analytics.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """Initialize analytics database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS answer_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                query TEXT,
                answer_type TEXT,
                confidence_score REAL,
                generation_time REAL,
                sources_used INTEGER,
                citation_count INTEGER,
                user_feedback INTEGER,  -- 1=positive, 0=neutral, -1=negative
                answer_length INTEGER,
                metadata TEXT
            )
        """)

        conn.commit()
        conn.close()

    def log_answer(self, result, user_feedback: int = 0):
        """Log answer generation result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO answer_logs (
                timestamp, query, answer_type, confidence_score,
                generation_time, sources_used, citation_count,
                user_feedback, answer_length, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            result.query,
            result.answer_type,
            result.confidence_score,
            result.generation_time,
            len(result.sources),
            len(result.citations),
            user_feedback,
            len(result.answer),
            json.dumps(result.metadata)
        ))

        conn.commit()
        conn.close()

    def get_performance_metrics(self, days: int = 7) -> Dict:
        """Get performance metrics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        since_date = (datetime.now() - timedelta(days=days)).isoformat()

        cursor.execute("""
            SELECT * FROM answer_logs 
            WHERE timestamp > ?
        """, (since_date,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"message": "No data available"}

        # Calculate metrics
        total_queries = len(rows)
        avg_confidence = sum(row[4] for row in rows) / total_queries
        avg_generation_time = sum(row[5] for row in rows) / total_queries
        avg_sources_used = sum(row[6] for row in rows) / total_queries

        # Answer type distribution
        answer_types = defaultdict(int)
        for row in rows:
            answer_types[row[3]] += 1

        # User feedback (if available)
        positive_feedback = sum(1 for row in rows if row[8] > 0)
        negative_feedback = sum(1 for row in rows if row[8] < 0)

        return {
            "period_days": days,
            "total_queries": total_queries,
            "avg_confidence": round(avg_confidence, 3),
            "avg_generation_time": round(avg_generation_time, 3),
            "avg_sources_used": round(avg_sources_used, 1),
            "answer_type_distribution": dict(answer_types),
            "user_feedback": {
                "positive": positive_feedback,
                "negative": negative_feedback,
                "neutral": total_queries - positive_feedback - negative_feedback
            }
        }

    def get_common_queries(self, limit: int = 10) -> List[Dict]:
        """Get most common queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT query, COUNT(*) as frequency, AVG(confidence_score) as avg_confidence
            FROM answer_logs 
            GROUP BY query 
            ORDER BY frequency DESC 
            LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "query": row[0],
                "frequency": row[1],
                "avg_confidence": round(row[2], 3)
            }
            for row in rows
        ]


# Usage in your answer system:
def enhanced_generate_answer(answer_generator, analytics, query: str):
    """Generate answer with analytics logging"""
    result = answer_generator.generate_answer(query)
    analytics.log_answer(result)
    return result