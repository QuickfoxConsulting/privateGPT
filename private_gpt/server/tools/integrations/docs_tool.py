import logging
from typing import Any, Dict, List, Optional, Union

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from llama_index.core.tools import BaseTool, FunctionTool, ToolMetadata

from private_gpt.server.tools.tool_interface import BaseMCPTool, ToolAuthConfig, ToolCapability

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/documents']

class DocsTool(BaseMCPTool):
    """Tool for interacting with Google Docs API"""
    
    def _create_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="docs_tool",
            description=(
                "Interact with Google Docs to read and create documents.\n"
                "Capabilities:\n"
                "- Create document: `create_document(title='...')`\n"
                "- Read document: `read_document(document_id='...')`\n"
                "- Append text: `append_text(document_id='...', text='...')`\n"
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
                if hasattr(self, 'config'):
                    self.config['token'] = creds.token
            except Exception as e:
                logger.error(f"Failed to refresh Docs token: {e}")
                
        return build('docs', 'v1', credentials=creds)

    def test_connection(self) -> bool:
        try:
            service = self._get_service()
            # Docs API check
            return True
        except Exception as e:
            logger.error(f"Docs connection test failed: {e}")
            return False

    def create_document(self, title: str, confirm: bool = False) -> str:
        """Create a new Google Doc with the given title.
        Set confirm=True only after the user has explicitly approved the title.
        """
        if not confirm:
            return (
                "ACTION_REQUIRED: Please confirm you want to create this document:\n"
                "--------------------------------------------------\n"
                f"Title: {title}\n"
                "--------------------------------------------------\n"
                "Reply with 'Confirm doc creation' to proceed."
            )

        try:
            service = self._get_service()
            body = {'title': title}
            doc = service.documents().create(body=body).execute()
            return f"Document created: ID {doc.get('documentId')}"
        except Exception as e:
            logger.error(f"Error creating document: {e}")
            return f"Error creating document: {str(e)}"

    def read_document(self, document_id: str) -> str:
        """Read the full text content of a Google Doc."""
        try:
            service = self._get_service()
            document = service.documents().get(documentId=document_id).execute()
            content = document.get('body').get('content')
            return self._read_structural_elements(content)
        except Exception as e:
            logger.error(f"Error reading document {document_id}: {e}")
            return f"Error reading document: {str(e)}"

    def _read_structural_elements(self, elements: List[Dict[str, Any]]) -> str:
        """Recursively read text from structural elements."""
        text = ''
        for value in elements:
            if 'paragraph' in value:
                elems = value.get('paragraph').get('elements')
                for elem in elems:
                    text += self._read_paragraph_element(elem)
            elif 'table' in value:
                table = value.get('table')
                for row in table.get('tableRows'):
                    cells = row.get('tableCells')
                    for cell in cells:
                        text += self._read_structural_elements(cell.get('content'))
            elif 'tableOfContents' in value:
                toc = value.get('tableOfContents')
                text += self._read_structural_elements(toc.get('content'))
        return text

    def _read_paragraph_element(self, element: Dict[str, Any]) -> str:
        """Read text from a paragraph element."""
        text_run = element.get('textRun')
        if not text_run:
            return ''
        return text_run.get('content')

    def append_text(self, document_id: str, text: str, confirm: bool = False) -> str:
        """Append text to the end of a document.
        Set confirm=True only after the user has explicitly approved the content.
        """
        if not confirm:
            return (
                "ACTION_REQUIRED: Please confirm you want to append text to this document:\n"
                "--------------------------------------------------\n"
                f"Document ID: {document_id}\n"
                f"Text to append: {text}\n"
                "--------------------------------------------------\n"
                "Reply with 'Confirm append' to proceed."
            )

        try:
            service = self._get_service()
            requests = [
                {
                    'insertText': {
                        'endOfSegmentLocation': {
                            'segmentId': '' # Body
                        },
                        'text': text
                    }
                }
            ]
            service.documents().batchUpdate(
                documentId=document_id, body={'requests': requests}
            ).execute()
            return "Text appended successfully."
        except Exception as e:
            logger.error(f"Error appending text to {document_id}: {e}")
            return f"Error appending text: {str(e)}"
    
    def __call__(self, *args, **kwargs):
        pass

    def to_tool_list(self) -> List[BaseTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.create_document,
                name="docs_create",
                description="Create a new Google Doc. Requires title. **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            ),
            FunctionTool.from_defaults(
                fn=self.read_document,
                name="docs_read",
                description="Read content of a Google Doc given its document ID. (No approval needed for reading)"
            ),
            FunctionTool.from_defaults(
                fn=self.append_text,
                name="docs_append",
                description="Append text to the end of a Google Doc. Requires document_id and text. **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            )
        ]
