import datetime
import logging
from typing import Any, Dict, List, Optional, Union

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from llama_index.core.tools import BaseTool, FunctionTool, ToolMetadata

from private_gpt.server.tools.tool_interface import BaseMCPTool, ToolAuthConfig, ToolCapability

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/calendar']

class CalendarTool(BaseMCPTool):
    """Tool for interacting with Google Calendar API"""
    
    def _create_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="calendar_tool",
            description=(
                "Interact with Google Calendar to list, create, and delete events.\n"
                "Capabilities:\n"
                "- List events: `list_upcoming_events(max_results=10)`\n"
                "- Create event: `create_event(summary='...', start_time='...', end_time='...', location='...')`\n"
                "- Delete event: `delete_event(event_id='...')`"
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
                if hasattr(self, 'config'):
                    self.config['token'] = creds.token
            except Exception as e:
                logger.error(f"Failed to refresh Calendar token: {e}")
                
        return build('calendar', 'v3', credentials=creds)

    def test_connection(self) -> bool:
        try:
            service = self._get_service()
            service.calendarList().list(maxResults=1).execute()
            return True
        except Exception as e:
            logger.error(f"Calendar connection test failed: {e}")
            return False

    def list_upcoming_events(self, max_results: int = 10) -> Union[List[Dict[str, Any]], str]:
        """List the next upcoming events from the primary calendar"""
        try:
            service = self._get_service()
            now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
            
            events_result = service.events().list(
                calendarId='primary', timeMin=now,
                maxResults=max_results, singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])
            
            event_list = []
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                event_list.append({
                    "id": event['id'],
                    "summary": event.get("summary", "(No Title)"),
                    "start": start,
                    "status": event.get("status"),
                    "link": event.get("htmlLink")
                })
                
            return event_list
        except Exception as e:
            logger.error(f"Error listing calendar events: {e}")
            return f"Error listing calendar events: {str(e)}"

    def create_event(
        self, 
        summary: str, 
        start_time: str, 
        end_time: str, 
        location: Optional[str] = None,
        description: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        confirm: bool = False
    ) -> str:
        """
        Create a new event in the primary calendar.
        Format times as ISO 8601 strings (e.g., '2023-10-25T14:00:00Z').
        Set confirm=True only after the user has explicitly approved the details.
        """
        if not confirm:
            attendees_str = f"Attendees: {', '.join(attendees)}\n" if attendees else ""
            location_str = f"Location: {location}\n" if location else ""
            return (
                "ACTION_REQUIRED: Please confirm you want to create this calendar event:\n"
                "--------------------------------------------------\n"
                f"Summary: {summary}\n"
                f"Start: {start_time}\n"
                f"End: {end_time}\n"
                f"{location_str}"
                f"{attendees_str}"
                f"Description: {description or '(No description)'}\n"
                "--------------------------------------------------\n"
                "Reply with 'Confirm event' to proceed."
            )

        try:
            service = self._get_service()
            
            event_body = {
                'summary': summary,
                'location': location or '',
                'description': description or '',
                'start': {
                    'dateTime': start_time,
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time,
                    'timeZone': 'UTC',
                },
            }
            if attendees:
                event_body['attendees'] = [{'email': email} for email in attendees]
            
            event = service.events().insert(calendarId='primary', body=event_body).execute()
            return f"Event created: {event.get('htmlLink')}"
        except Exception as e:
            logger.error(f"Error creating calendar event: {e}")
            return f"Error creating calendar event: {str(e)}"

    def delete_event(self, event_id: str, confirm: bool = False) -> str:
        """Delete an event from the primary calendar by its ID."""
        try:
            service = self._get_service()
            # Try to get event details for confirmation
            event = service.events().get(calendarId='primary', eventId=event_id).execute()
            
            if not confirm:
                return (
                    "ACTION_REQUIRED: Please confirm you want to delete this event:\n"
                    "--------------------------------------------------\n"
                    f"Summary: {event.get('summary')}\n"
                    f"Start: {event.get('start', {}).get('dateTime', event.get('start', {}).get('date'))}\n"
                    "--------------------------------------------------\n"
                    "Reply with 'Confirm delete event' to proceed."
                )
            
            service.events().delete(calendarId='primary', eventId=event_id).execute()
            return f"Event {event_id} successfully deleted."
        except Exception as e:
            logger.error(f"Error deleting calendar event {event_id}: {e}")
            return f"Error deleting calendar event: {str(e)}"
    
    def __call__(self, *args, **kwargs):
        pass

    def to_tool_list(self) -> List[BaseTool]:
        """Convert this MCP tool into a list of LlamaIndex FunctionTools"""
        
        return [
            FunctionTool.from_defaults(
                fn=self.list_upcoming_events,
                name="calendar_list_upcoming",
                description="List upcoming events from Google Calendar."
            ),
            FunctionTool.from_defaults(
                fn=self.create_event,
                name="calendar_create_event",
                description="Create a new event in Google Calendar. Requires summary, start_time, and end_time (ISO format). Optional location, description, attendees. **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            ),
            FunctionTool.from_defaults(
                fn=self.delete_event,
                name="calendar_delete_event",
                description="Delete an event from Google Calendar given its event ID. **IMPORTANT: First call WITHOUT confirm to get approval, then call WITH confirm=True ONLY after user explicitly approves.**"
            )
        ]
