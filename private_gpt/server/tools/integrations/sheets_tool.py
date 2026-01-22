import logging
from typing import Any, Dict, List, Optional, Union

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from llama_index.core.tools import BaseTool, FunctionTool, ToolMetadata

from private_gpt.server.tools.tool_interface import BaseMCPTool, ToolAuthConfig, ToolCapability

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

class SheetsTool(BaseMCPTool):
    """Tool for interacting with Google Sheets API"""
    
    def _create_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="sheets_tool",
            description=(
                "Interact with Google Sheets to read and write data.\n"
                "Capabilities:\n"
                "- Read range: `get_values(spreadsheet_id='...', range_name='Sheet1!A1:B10')`\n"
                "- Write range: `update_values(spreadsheet_id='...', range_name='Sheet1!A1', values=[['Heading1', 'Heading2']])`\n"
                "- Create sheet: `create_spreadsheet(title='...')`\n"
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
                logger.error(f"Failed to refresh Sheets token: {e}")
                
        return build('sheets', 'v4', credentials=creds)

    def test_connection(self) -> bool:
        try:
            service = self._get_service()
            return True
        except Exception as e:
            logger.error(f"Sheets connection test failed: {e}")
            return False

    def get_values(self, spreadsheet_id: str, range_name: str) -> Union[List[List[Any]], str]:
        """Read values from a specific range in a spreadsheet"""
        try:
            service = self._get_service()
            result = service.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id, range=range_name
            ).execute()
            return result.get('values', [])
        except Exception as e:
            logger.error(f"Error reading sheets values from {spreadsheet_id}: {e}")
            return f"Error reading sheets values: {str(e)}"

    def update_values(
        self, 
        spreadsheet_id: str, 
        range_name: str, 
        values: List[List[Any]],
        confirm: bool = False
    ) -> str:
        """Write values to a specific range. 'values' must be a list of lists.
        Set confirm=True only after the user has explicitly approved the update.
        """
        if not confirm:
            return (
                "ACTION_REQUIRED: Please confirm you want to update these values in the spreadsheet:\n"
                "--------------------------------------------------\n"
                f"Spreadsheet ID: {spreadsheet_id}\n"
                f"Range: {range_name}\n"
                f"Values (first row): {values[0] if values else '[]'}\n"
                "--------------------------------------------------\n"
                "Reply with 'Confirm update' to proceed."
            )

        try:
            service = self._get_service()
            body = {'values': values}
            result = service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id, range=range_name,
                valueInputOption='USER_ENTERED', body=body
            ).execute()
            return f"{result.get('updatedCells')} cells updated successfully."
        except Exception as e:
            logger.error(f"Error updating sheets values in {spreadsheet_id}: {e}")
            return f"Error updating sheets values: {str(e)}"

    def create_spreadsheet(self, title: str, confirm: bool = False) -> str:
        """Create a new spreadsheet with the given title.
        Set confirm=True only after the user has explicitly approved the creation.
        """
        if not confirm:
            return (
                "ACTION_REQUIRED: Please confirm you want to create this spreadsheet:\n"
                "--------------------------------------------------\n"
                f"Title: {title}\n"
                "--------------------------------------------------\n"
                "Reply with 'Confirm spreadsheet creation' to proceed."
            )

        try:
            service = self._get_service()
            spreadsheet_body = {'properties': {'title': title}}
            spreadsheet = service.spreadsheets().create(
                body=spreadsheet_body, fields='spreadsheetId'
            ).execute()
            return f"Spreadsheet created: ID {spreadsheet.get('spreadsheetId')}"
        except Exception as e:
            logger.error(f"Error creating spreadsheet: {e}")
            return f"Error creating spreadsheet: {str(e)}"
    
    def __call__(self, *args, **kwargs):
        pass

    def to_tool_list(self) -> List[BaseTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.get_values,
                name="sheets_read",
                description="Read data from a Google Sheet range (e.g. 'Sheet1!A1:B5'). Returns list of rows."
            ),
            FunctionTool.from_defaults(
                fn=self.update_values,
                name="sheets_write",
                description="Write data to a Google Sheet range. Values must be a list of lists. **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            ),
            FunctionTool.from_defaults(
                fn=self.create_spreadsheet,
                name="sheets_create",
                description="Create a new empty Google Sheet. Requires title. **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            )
        ]
