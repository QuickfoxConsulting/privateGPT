import logging
import requests
from typing import Any, Dict, List, Optional
from private_gpt.users.core.config import settings
from llama_index.core.tools import BaseTool, FunctionTool, ToolMetadata, ToolOutput
from private_gpt.server.tools.tool_interface import BaseMCPTool, ToolAuthConfig, ToolCapability

logger = logging.getLogger(__name__)

class SerperTool(BaseMCPTool):
    """Serper.dev search tool integration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Try to get API key from config, fallback to env var if not in config
        # (Though BaseMCPTool usually relies on explicit config)
        self.api_key = settings.SERPER_API
        if not self.api_key:
             # Try getting from settings if available globally, though this breaks pure isolation
             # Ideally, the user configures this tool with the key.
             # We'll stick to config-first.
             pass

    def _create_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="serper_tool",
            description="Serper.dev search tools (Web, Instant, News)."
        )

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "api_key": {"type": "string", "description": "Serper.dev API Key"}
            },
            "required": ["api_key"]
        }

    @classmethod
    def get_capabilities(cls) -> List[ToolCapability]:
        return [ToolCapability.SEARCH]

    @classmethod
    def get_auth_config(cls) -> Optional[ToolAuthConfig]:
        return ToolAuthConfig(
            auth_type="basic",
            required_params={"api_key": "Serper API Key"}
        )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return "api_key" in config

    def _get_api_key(self) -> str:
        if self.api_key:
            return self.api_key
        # Check environment as fallback
        import os
        return os.environ.get("SERPER_API_KEY", "")

    def _make_request(self, query: str, search_type: str = "search") -> Dict:
        """Make a request to the Serper API."""
        api_key = self._get_api_key()
        if not api_key:
             raise ValueError("Serper API key not configured")

        url = f"https://google.serper.dev/{search_type}"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Serper API request failed: {str(e)}")

    def web_search(self, query: str) -> str:
        """Perform a web search using Serper API. Returns web search results with titles, links, and snippets."""
        try:
            data = self._make_request(query, "search")
            results = []
            organic_results = data.get("organic", [])[:10]
            
            for i, result in enumerate(organic_results, 1):
                results.append(f"{i}. {result.get('title', '')}\n{result.get('link', '')}\n{result.get('snippet', '')}")
                
            return "\n\n".join(results) if results else "No web search results found."
        except Exception as e:
            return f"Search failed: {str(e)}"

    def instant_search(self, query: str) -> str:
        """Get instant answers and knowledge graph information from Serper API."""
        try:
            data = self._make_request(query, "search")
            results = []
            
            # Answer box results
            if "answerBox" in data:
                answer_box = data["answerBox"]
                results.append(f"Answer: {answer_box.get('answer', '')}")
                if answer_box.get('title'):
                    results.append(f"Title: {answer_box.get('title')}")
            
            # Knowledge graph results
            if "knowledgeGraph" in data:
                kg = data["knowledgeGraph"]
                results.append(f"Knowledge Graph - {kg.get('title', '')}: {kg.get('description', '')}")
            
            return "\n\n".join(results) if results else "No instant answers found."
        except Exception as e:
            return f"Instant search failed: {str(e)}"

    def news_search(self, query: str) -> str:
        """Search for recent news articles using Serper API."""
        try:
            data = self._make_request(query, "news")
            results = []
            news_results = data.get("news", [])[:10]
            
            for i, result in enumerate(news_results, 1):
                date_info = f" ({result.get('date', '')})" if result.get('date') else ""
                results.append(f"{i}. {result.get('title', '')}{date_info}\n{result.get('link', '')}\n{result.get('snippet', '')}")
                
            return "\n\n".join(results) if results else "No news results found."
        except Exception as e:
            return f"News search failed: {str(e)}"

    def to_tool_list(self) -> List[BaseTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.web_search,
                name="serper_web_search",
                description="Perform a web search using Serper API. Returns web search results with titles, links, and snippets."
            ),
            FunctionTool.from_defaults(
                fn=self.instant_search,
                name="serper_instant_search",
                description="Get instant answers and knowledge graph information from Serper API."
            ),
            FunctionTool.from_defaults(
                fn=self.news_search,
                name="serper_news_search",
                description="Search for recent news articles using Serper API."
            )
        ]

    def __call__(self, *args, **kwargs):
        pass

    async def test_connection(self) -> bool:
        """Test connection to the Serper API."""
        try:
             # If no API key, we can't really test, but that's a config issue not connection.
             # Ideally we make a cheap call.
             if not self._get_api_key():
                 return False # Config missing, so 'connection' (authed access) fails
             
             # We could do a dummy search or just check if key exists. 
             # For now, let's assume if key is there, it's 'connected' enough for installation.
             # Real connection test would be making a request.
             self._make_request("test", "search")
             return True
        except Exception:
             return False
