import io
import pandas as pd
from typing import List, Optional
from private_gpt.users import schemas

def generate_audit_log_report(audit_logs: List[schemas.Audit], username: Optional[str] = None) -> io.BytesIO:
    """
    Generate an Excel report of the QuickGPT audit logs.

    Args:
        audit_logs (List[schemas.Audit]): List of audit logs to include in the report.
        username (str): Username for whom the audit log report is generated.

    Returns:
        io.BytesIO: In-memory Excel file buffer containing the report.
    """
    intro_data = [
        ["Audit Log Report"],
        ["Date:", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
        [" "],  
    ]
    intro_df = pd.DataFrame(intro_data)

    if username:
        total_login_counts = get_total_login_counts(audit_logs, username)
        intro_data.extend([
            ["Username:", username],
            ["Total Login Counts:", total_login_counts],
            [" "], 
        ])

    # Convert audit logs to dictionary format with all fields
    audit_data = []
    for log in audit_logs:
        log_dict = log.dict()
        # Flatten the details dictionary for better Excel representation
        if log_dict.get('details'):
            details = log_dict.pop('details')
            if isinstance(details, dict):
                # Add each detail as a separate column
                for key, value in details.items():
                    log_dict[f"detail_{key}"] = value
        audit_data.append(log_dict)
    
    audit_df = pd.DataFrame(audit_data)
    excel_buffer = io.BytesIO()

    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        intro_df.to_excel(writer, index=False, header=False)
        if username:
            audit_df.to_excel(writer, index=False, startrow=len(intro_data) + 2)
        else:
            audit_df.to_excel(writer, index=False, startrow=len(intro_data))

    excel_buffer.seek(0)
    return excel_buffer

def get_total_login_counts(audit_logs: List[schemas.Audit], username: str) -> int:
    """
    Get the total login counts for the given username.

    Args:
        audit_logs (List[schemas.Audit]): List of audit logs to search for login events.
        username (str): Username to get the login counts for.

    Returns:
        int: Total number of login events for the given username.
    """
    login_events = [log for log in audit_logs if log.action and "login" in log.action and log.username == username]
    return len(login_events)