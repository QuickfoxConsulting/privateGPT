from datetime import datetime
from sqlalchemy.orm import Session
from private_gpt.users.models.audit import Audit

def log_audit_entry(
    session: Session,
    model: str,
    action: str,
    details: dict,
    user_id: int = None,
    username: str = None,
    ip_address: str = None,
    user_agent: str = None,
    session_id: str = None,
    request_id: str = None,
    severity: str = "INFO",
    resource_id: str = None,
):
    """
    Log an audit entry with comprehensive information.
    
    Args:
        session: Database session
        model: The model/entity being audited (e.g., "Document", "User")
        action: The action being performed (e.g., "create", "update", "delete", "access")
        details: Additional details about the action
        user_id: ID of the user performing the action
        username: Username of the user performing the action
        ip_address: IP address of the client
        user_agent: User agent string of the client
        session_id: Session identifier
        request_id: Request identifier
        severity: Severity level (INFO, WARNING, ERROR)
        resource_id: ID of the resource being accessed
    """
    audit_entry = Audit(
        timestamp=datetime.utcnow(),
        user_id=user_id,
        username=username,
        model=model,
        action=action,
        details=details,
        ip_address=ip_address,
        user_agent=user_agent,
        session_id=session_id,
        request_id=request_id,
        severity=severity,
        resource_id=resource_id
    )
    
    session.add(audit_entry)
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        # Log the error but don't fail the main operation
        print(f"Failed to log audit entry: {str(e)}")