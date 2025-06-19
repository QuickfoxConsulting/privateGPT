import requests
from typing import Any, Dict, List, Optional
from llama_index.core.tools import ToolMetadata, BaseTool, ToolOutput
from private_gpt.users.core.config import settings


class SerperSearchToolSpec:
    """Serper.dev search tool spec with multiple search modes."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Serper search tool.
        
        Args:
            api_key (Optional[str]): Serper API key. If not provided, will look for SERPER_API_KEY env var.
        """
        self.api_key = settings.SERPER_API
        if not self.api_key:
            raise ValueError(
                "SerperSearchToolSpec requires a Serper API key. "
                "Provide it as an argument or set the SERPER_API_KEY environment variable."
            )

    def _make_request(self, query: str, search_type: str = "search") -> Dict:
        """Make a request to the Serper API."""
        url = f"https://google.serper.dev/{search_type}"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Serper API request failed: {str(e)}")

    def to_tool_list(self) -> List[BaseTool]:
        """Convert to LlamaIndex tools."""
        return [
            SerperWebSearchTool(self),
            SerperInstantSearchTool(self),
            SerperNewsSearchTool(self)
        ]

class SerperWebSearchTool(BaseTool):
    def __init__(self, serper_spec: SerperSearchToolSpec):
        self.serper_spec = serper_spec

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="serper_web_search",
            description="Perform a web search using Serper API. Returns web search results with titles, links, and snippets."
        )

    def __call__(self, input: Any) -> ToolOutput:
        try:
            if isinstance(input, str):
                query = input
            elif isinstance(input, dict):
                query = input.get("query", str(input))
            else:
                query = str(input)

            data = self.serper_spec._make_request(query, "search")
            results = []
            organic_results = data.get("organic", [])[:10]
            
            for i, result in enumerate(organic_results, 1):
                results.append(f"{i}. {result.get('title', '')}\n{result.get('link', '')}\n{result.get('snippet', '')}")
                
            content = "\n\n".join(results) if results else "No web search results found."
            
            return ToolOutput(
                content=content,
                raw_input={"query": query},
                raw_output=data,
                tool_name="serper_web_search"
            )
        except Exception as e:
            return ToolOutput(
                content=f"Search failed: {str(e)}",
                raw_input={"query": query if 'query' in locals() else str(input)},
                raw_output=None,
                tool_name="serper_web_search",
                is_error=True
            )

class SerperInstantSearchTool(BaseTool):
    def __init__(self, serper_spec: SerperSearchToolSpec):
        self.serper_spec = serper_spec

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="serper_instant_search",
            description="Get instant answers and knowledge graph information from Serper API."
        )

    def __call__(self, input: Any) -> ToolOutput:
        try:
            if isinstance(input, str):
                query = input
            elif isinstance(input, dict):
                query = input.get("query", str(input))
            else:
                query = str(input)

            data = self.serper_spec._make_request(query, "search")
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
            
            content = "\n\n".join(results) if results else "No instant answers found."
            
            return ToolOutput(
                content=content,
                raw_input={"query": query},
                raw_output=data,
                tool_name="serper_instant_search"
            )
        except Exception as e:
            return ToolOutput(
                content=f"Instant search failed: {str(e)}",
                raw_input={"query": query if 'query' in locals() else str(input)},
                raw_output=None,
                tool_name="serper_instant_search",
                is_error=True
            )

class SerperNewsSearchTool(BaseTool):
    def __init__(self, serper_spec: SerperSearchToolSpec):
        self.serper_spec = serper_spec

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="serper_news_search",
            description="Search for recent news articles using Serper API."
        )

    def __call__(self, input: Any) -> ToolOutput:
        try:
            if isinstance(input, str):
                query = input
            elif isinstance(input, dict):
                query = input.get("query", str(input))
            else:
                query = str(input)

            data = self.serper_spec._make_request(query, "news")
            results = []
            news_results = data.get("news", [])[:10]
            
            for i, result in enumerate(news_results, 1):
                date_info = f" ({result.get('date', '')})" if result.get('date') else ""
                results.append(f"{i}. {result.get('title', '')}{date_info}\n{result.get('link', '')}\n{result.get('snippet', '')}")
                
            content = "\n\n".join(results) if results else "No news results found."
            
            return ToolOutput(
                content=content,
                raw_input={"query": query},
                raw_output=data,
                tool_name="serper_news_search"
            )
        except Exception as e:
            return ToolOutput(
                content=f"News search failed: {str(e)}",
                raw_input={"query": query if 'query' in locals() else str(input)},
                raw_output=None,
                tool_name="serper_news_search",
                is_error=True
            )
