import base64
import logging
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Union

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from llama_index.core.tools import BaseTool, ToolMetadata

from private_gpt.server.tools.tool_interface import BaseMCPTool, ToolAuthConfig, ToolCapability

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']

class GmailTool(BaseMCPTool):
    """Tool for interacting with Gmail API"""
    
    def _create_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="gmail_tool",
            description=(
                "Interact with Gmail to search, read, reply and send emails.\n"
                "Capabilities:\n"
                "- Search emails: `search_emails(query='from:boss')`\n"
                "- Read email: `get_email(message_id='...')`\n"
                "- Create draft: `create_draft(to='...', subject='...', body='...', cc='...', bcc='...')`\n"
                "- Send email: `send_email(to='...', subject='...', body='...', cc='...', bcc='...')`\n"
                "- Reply to email: `reply_email(message_id='...', body='...')`"
            )
        )

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "token": {"type": "string"},
                "refresh_token": {"type": "string"},
                "token_uri": {"type": "string"},
                "client_id": {"type": "string"},
                "client_secret": {"type": "string"},
                "scopes": {"type": "array", "items": {"type": "string"}} 
            },
            "required": ["refresh_token"]
        }

    @classmethod
    def get_capabilities(cls) -> List[ToolCapability]:
        return [
            ToolCapability.READ,
            ToolCapability.WRITE,
            ToolCapability.SEARCH,
            ToolCapability.CREATE
        ]

    @classmethod
    def get_auth_config(cls) -> Optional[ToolAuthConfig]:
        return ToolAuthConfig(
            auth_type="oauth2",
            required_params={
                "client_id": "Google Client ID",
                "client_secret": "Google Client Secret"
            },
            oauth_scopes=SCOPES
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        # Minimal validation; Registry ensures credentials are filled
        return "refresh_token" in config

    def _get_service(self):
        creds = Credentials(
            token=self.config.get("token"),
            refresh_token=self.config.get("refresh_token"),
            token_uri=self.config.get("token_uri", "https://oauth2.googleapis.com/token"),
            client_id=self.config.get("client_id"),
            client_secret=self.config.get("client_secret"),
            scopes=self.config.get("scopes", SCOPES)
        )
        
        # Refresh token if expired
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # Update config with new token if possible
                # Note: self.config is usually static in instance, but update if setter exists
                if hasattr(self, 'config'):
                    self.config['token'] = creds.token
            except Exception as e:
                logger.error(f"Failed to refresh Gmail token: {e}")
                
        return build('gmail', 'v1', credentials=creds)

    def test_connection(self) -> bool:
        try:
            service = self._get_service()
            service.users().getProfile(userId='me').execute()
            return True
        except Exception as e:
            logger.error(f"Gmail connection test failed: {e}")
            return False

    def _get_body_from_payload(self, payload: Dict[str, Any]) -> str:
        """Recursively extract body from message payload."""
        if 'parts' in payload:
            # Look for text/plain first, then text/html
            text_parts = []
            html_parts = []
            
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    text_parts.append(part)
                elif part['mimeType'] == 'text/html':
                    html_parts.append(part)
                elif 'parts' in part:
                    # Recursive call for nested multipart
                    body = self._get_body_from_payload(part)
                    if body:
                        return body
            
            # Prefer plain text
            target_part = text_parts[0] if text_parts else (html_parts[0] if html_parts else None)
            if target_part:
                data = target_part['body'].get('data')
                if data:
                    return base64.urlsafe_b64decode(data).decode()
        else:
            # Single part message
            data = payload['body'].get('data')
            if data:
                return base64.urlsafe_b64decode(data).decode()
                
        return ""

    def search_emails(self, query: str, max_results: int = 10) -> Union[List[Dict[str, Any]], str]:
        """Search for emails matching the query"""
        try:
            service = self._get_service()
            results = service.users().messages().list(
                userId='me', q=query, maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            summaries = []
            
            for msg in messages:
                # Fetch minimal details for summary
                data = service.users().messages().get(
                    userId='me', id=msg['id'], format='metadata'
                ).execute()
                
                headers = {h['name']: h['value'] for h in data['payload']['headers']}
                summaries.append({
                    "id": msg['id'],
                    "subject": headers.get("Subject", "(No Subject)"),
                    "from": headers.get("From", "Unknown"),
                    "date": headers.get("Date", "")
                })
                
            return summaries
        except Exception as e:
            logger.error(f"Error searching emails: {e}")
            return f"Error searching emails: {str(e)}"

    def get_email(self, message_id: str) -> str:
        """Get full content of a specific email"""
        try:
            service = self._get_service()
            message = service.users().messages().get(
                userId='me', id=message_id, format='full'
            ).execute()
            
            headers = {h['name']: h['value'] for h in message['payload']['headers']}
            body = self._get_body_from_payload(message['payload']) or message.get('snippet', '')
            
            return (
                f"ID: {message_id}\n"
                f"From: {headers.get('From')}\n"
                f"Subject: {headers.get('Subject')}\n"
                f"Date: {headers.get('Date')}\n\n"
                f"{body}"
            )
        except Exception as e:
            logger.error(f"Error fetching email {message_id}: {e}")
            return f"Error fetching email: {str(e)}"

    def create_draft(
        self, 
        to: str, 
        subject: str, 
        body: str, 
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        confirm: bool = False
    ) -> str:
        """Create a draft email. 
        Set confirm=True only after the user has explicitly approved the content.
        """
        if not confirm:
            cc_str = f"CC: {cc}\n" if cc else ""
            bcc_str = f"BCC: {bcc}\n" if bcc else ""
            return (
                "ACTION_REQUIRED: Please confirm you want to create this draft:\n"
                "--------------------------------------------------\n"
                f"To: {to}\n"
                f"{cc_str}"
                f"{bcc_str}"
                f"Subject: {subject}\n"
                f"Body:\n{body}\n"
                "--------------------------------------------------\n"
                "Reply with 'Confirm draft' to proceed."
            )

        try:
            service = self._get_service()
            
            message = EmailMessage()
            message.set_content(body)
            message['To'] = to
            message['Subject'] = subject
            if cc:
                message['Cc'] = cc if isinstance(cc, str) else ", ".join(cc)
            if bcc:
                message['Bcc'] = bcc if isinstance(bcc, str) else ", ".join(bcc)
            
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {'message': {'raw': encoded_message}}
            
            draft = service.users().drafts().create(
                userId='me', body=create_message
            ).execute()
            
            return f"Draft created with ID: {draft['id']}"
        except Exception as e:
            logger.error(f"Error creating draft: {e}")
            return f"Error creating draft: {str(e)}"

    def send_email(
        self, 
        to: str, 
        subject: str, 
        body: str, 
        cc: Optional[Union[str, List[str]]] = None,
        bcc: Optional[Union[str, List[str]]] = None,
        confirm: bool = False
    ) -> str:
        """Send an email immediately.
        Set confirm=True only after the user has explicitly approved sending this email.
        """
        if not confirm:
            cc_str = f"CC: {cc}\n" if cc else ""
            bcc_str = f"BCC: {bcc}\n" if bcc else ""
            return (
                "ACTION_REQUIRED: Please confirm you want to send this email:\n"
                "--------------------------------------------------\n"
                f"To: {to}\n"
                f"{cc_str}"
                f"{bcc_str}"
                f"Subject: {subject}\n"
                f"Body:\n{body}\n"
                "--------------------------------------------------\n"
                "Reply with 'Confirm send' to proceed."
            )

        try:
            service = self._get_service()
            
            message = EmailMessage()
            message.set_content(body)
            message['To'] = to
            message['Subject'] = subject
            if cc:
                message['Cc'] = cc if isinstance(cc, str) else ", ".join(cc)
            if bcc:
                message['Bcc'] = bcc if isinstance(bcc, str) else ", ".join(bcc)
            
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {'raw': encoded_message}
            
            sent_message = service.users().messages().send(
                userId='me', body=create_message
            ).execute()
            
            return f"Email sent with ID: {sent_message['id']}"
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return f"Error sending email: {str(e)}"

    def reply_email(self, message_id: str, body: str, confirm: bool = False) -> str:
        """Reply to an existing email.
        Set confirm=True only after the user has explicitly approved the reply.
        """
        try:
            service = self._get_service()
            # Fetch original message to get headers for threading
            original = service.users().messages().get(
                userId='me', id=message_id, format='metadata'
            ).execute()
            
            headers = {h['name']: h['value'] for h in original['payload']['headers']}
            
            # Recipient is the 'From' of the original message OR 'Reply-To'
            to = headers.get('Reply-To') or headers.get('From')
            subject = headers.get('Subject', '')
            if not subject.lower().startswith('re:'):
                subject = f"Re: {subject}"
                
            if not confirm:
                return (
                    "ACTION_REQUIRED: Please confirm you want to send this reply:\n"
                    "--------------------------------------------------\n"
                    f"To: {to}\n"
                    f"Subject: {subject}\n"
                    f"Body:\n{body}\n"
                    "--------------------------------------------------\n"
                    "Reply with 'Confirm reply' to proceed."
                )

            message = EmailMessage()
            message.set_content(body)
            message['To'] = to
            message['Subject'] = subject
            
            # Threading headers
            message['In-Reply-To'] = headers.get('Message-ID')
            message['References'] = headers.get('References', '') + ' ' + headers.get('Message-ID', '')
            
            # Thread ID must be same as original
            thread_id = original.get('threadId')
            
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            create_message = {'raw': encoded_message, 'threadId': thread_id}
            
            sent_message = service.users().messages().send(
                userId='me', body=create_message
            ).execute()
            
            return f"Reply sent with ID: {sent_message['id']}"
        except Exception as e:
            logger.error(f"Error replying to email {message_id}: {e}")
            return f"Error replying to email: {str(e)}"
    
    def __call__(self, *args, **kwargs):
        pass

    def to_tool_list(self) -> List[BaseTool]:
        """Convert this MCP tool into a list of LlamaIndex FunctionTools"""
        from llama_index.core.tools import FunctionTool
        
        return [
            FunctionTool.from_defaults(
                fn=self.search_emails,
                name="gmail_search",
                description="Search for emails in Gmail. Query format: 'from:alice', 'subject:report', etc. (No approval needed for reading). Input parameters: query (string) - search query, max_results (int, optional, default=10)."
            ),
            FunctionTool.from_defaults(
                fn=self.get_email,
                name="gmail_read",
                description="Read full content of an email given its message ID. (No approval needed for reading). Input parameter: message_id (string) - the email message ID."
            ),
            FunctionTool.from_defaults(
                fn=self.create_draft,
                name="gmail_draft",
                description="Create a draft email. Input parameters: to (string), subject (string), body (string), cc (string, optional), bcc (string, optional), confirm (bool, default=False). **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            ),
            FunctionTool.from_defaults(
                fn=self.send_email,
                name="gmail_send",
                description="Send an email immediately. Input parameters: to (string), subject (string), body (string), cc (string, optional), bcc (string, optional), confirm (bool, default=False). **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            ),
            FunctionTool.from_defaults(
                fn=self.reply_email,
                name="gmail_reply",
                description="Reply to an existing email. Input parameters: message_id (string), body (string), confirm (bool, default=False). **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            )
        ]
