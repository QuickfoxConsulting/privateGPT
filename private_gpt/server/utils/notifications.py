
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from private_gpt.users.core.config import settings

logger = logging.getLogger(__name__)

class NotificationService:
    @staticmethod
    def send_vendor_notification_email(query: str, response: str) -> None:
        """
        Sends an email notification to the vendor when a query cannot be answered.
        """
        if not settings.SMTP_SERVER or not settings.VENDOR_NOTIFICATION_EMAIL:
            # Silently return if not configured to avoid log spam if feature is not used
            return

        subject = f"QuickRef Alert: Unanswered Query - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 8px;">
                <h2 style="color: #d9534f;">Unanswered Query Alert</h2>
                <p>The AI system encountered a query that could not be answered from the provided documentation.</p>
                
                <div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007bff; margin-bottom: 20px;">
                    <strong>User Query:</strong><br>
                    {query}
                </div>
                
                <div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #28a745; margin-bottom: 20px;">
                    <strong>AI Response:</strong><br>
                    {response}
                </div>
                
                <p style="font-size: 0.9em; color: #666;">
                    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                    System: QuickRef AI
                </p>
            </div>
        </body>
        </html>
        """

        msg = MIMEMultipart()
        msg['From'] = settings.SMTP_SENDER_EMAIL
        msg['To'] = settings.VENDOR_NOTIFICATION_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'html'))

        try:
            # Ensure port is an integer
            port = int(settings.SMTP_PORT) if settings.SMTP_PORT else 587
            
            with smtplib.SMTP(settings.SMTP_SERVER, port) as server:
                server.starttls()
                server.login(settings.SMTP_SENDER_EMAIL, settings.SMTP_PASSWORD)
                server.send_message(msg)
            logger.info(f"Vendor notification email sent to {settings.VENDOR_NOTIFICATION_EMAIL}")
        except Exception as e:
            logger.error(f"Failed to send vendor notification email: {e}")
