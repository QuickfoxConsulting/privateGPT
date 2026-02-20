from typing import Any, Dict, List, Optional
from sqlalchemy import func, and_
from sqlalchemy.orm import Session
from private_gpt.users.models.audit import Audit
from datetime import datetime, timedelta

class AuditService:
    @staticmethod
    def get_chat_metrics(db: Session, conversation_id: str) -> Dict[str, Any]:
        """
        Aggregate metrics for a specific conversation.
        Categorizes logs into Total, Answered, Unanswered, and Failed.
        """
        # Filter for the specific conversation
        audit_logs = db.query(Audit).filter(Audit.resource_id == conversation_id).all()
        
        total_attempts = 0
        answered = 0
        unanswered = 0
        failed = 0
        
        for log in audit_logs:
            action = log.action
            details = log.details or {}
            
            if action == "chat_completion_attempt":
                total_attempts += 1
            elif action in ["chat_completion_success", "chat_completion_success_streamed"]:
                if details.get("is_answered") is True or details.get("source_count", 0) > 0:
                    answered += 1
                else:
                    unanswered += 1
            elif action == "chat_completion_failure":
                failed += 1
                
        return {
            "conversation_id": conversation_id,
            "total_questions": total_attempts,
            "answered": answered,
            "unanswered": unanswered,
            "failed": failed,
            "success_rate": round(answered / total_attempts * 100, 2) if total_attempts > 0 else 0
        }

    @staticmethod
    def get_admin_dashboard_stats(db: Session, days: int = 7) -> Dict[str, Any]:
        """
        Aggregate overall workload and status for the admin dashboard.
        """
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Total workload (Total Questions Asked)
        total_queries = db.query(func.count(Audit.id)).filter(
            Audit.action == "chat_completion_attempt",
            Audit.timestamp >= start_date
        ).scalar() or 0
        
        # Answered queries
        answered_count = db.query(func.count(Audit.id)).filter(
            Audit.action == "chat_completion_success",
            Audit.timestamp >= start_date
        ).scalar() or 0
        
        # Unanswered queries (RAG refusal or no docs)
        unanswered_count = db.query(func.count(Audit.id)).filter(
            Audit.action == "chat_completion_unanswered",
            Audit.timestamp >= start_date
        ).scalar() or 0
        
        # Total failures (System errors)
        failures = db.query(func.count(Audit.id)).filter(
            Audit.action == "chat_completion_failure",
            Audit.timestamp >= start_date
        ).scalar() or 0
        
        # User adoption (Active users)
        active_users = db.query(func.count(func.distinct(Audit.user_id))).filter(
            Audit.timestamp >= start_date
        ).scalar() or 0
        
        return {
            "period_days": days,
            "total_workload": total_queries,
            "active_users": active_users,
            "answered_queries": answered_count,
            "unanswered_queries": unanswered_count,
            "failure_count": failures,
            "global_success_rate": round(answered_count / total_queries * 100, 2) if total_queries > 0 else 0
        }

audit_service = AuditService()
