import logging
import re
import json
import asyncio
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, RegexChunking
from crawl4ai.async_configs import CacheMode
from crawl4ai.extraction_strategy import NoExtractionStrategy
from llama_index.core.tools import BaseTool, FunctionTool, ToolMetadata, ToolOutput

from private_gpt.server.tools.tool_interface import BaseMCPTool, ToolAuthConfig, ToolCapability

logger = logging.getLogger(__name__)

class Crawl4AITool(BaseMCPTool):
    """Tool to extract readable content from a webpage."""

    def __init__(self, verbose: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self._executor = ThreadPoolExecutor(max_workers=1)


    async def test_connection(self) -> bool:
        """Test if the crawler is ready."""
        return True

    def _create_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="crawl4ai_scraper",
            description="Extract readable content from a webpage using its URL."
        )

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "verbose": {"type": "boolean"}
            },
            "required": []
        }

    @classmethod
    def get_capabilities(cls) -> List[ToolCapability]:
        return [ToolCapability.READ]

    @classmethod
    def get_auth_config(cls) -> Optional[ToolAuthConfig]:
        return None

    def validate_config(self, config: Dict[str, Any]) -> bool:
        return True

    def _validate_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            return all([parsed.scheme, parsed.netloc])
        except:
            return False

    async def _async_crawl(self, url: str) -> str:
        """Async crawling logic using Crawl4AI."""
        try:
            async with AsyncWebCrawler(verbose=self.verbose) as crawler:
                config = CrawlerRunConfig(
                    word_count_threshold=10,
                    extraction_strategy=NoExtractionStrategy(),
                    chunking_strategy=RegexChunking(),
                    cache_mode=CacheMode.BYPASS
                )
                result = await crawler.arun(
                    url=url,
                    config=config
                )
                if result.success and result.markdown:
                    content = re.sub(r'\n\s*\n', '\n\n', result.markdown)
                    return f"Content from {url}:\n\n{content}"
                return f"No readable content found at {url}"
        except Exception as e:
            return f"Error crawling {url}: {str(e)}"

    def _run_in_thread(self, url: str) -> str:
        """Run the async crawl in a separate thread with its own event loop."""
        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_crawl(url))
            finally:
                loop.close()
        
        return self._executor.submit(_run).result()

    def crawl_url(self, url: str) -> str:
        """Extract readable content from a webpage."""
        if not self._validate_url(url):
            return f"Invalid URL format: {url}"

        # Run the crawl in a separate thread with its own event loop
        # This mirrors the original implementation's behavior
        return self._run_in_thread(url)
    
    def __call__(self, input: Any) -> ToolOutput:
        # Maintain original __call__ behavior for backward compatibility if needed,
        # but typical BaseMCPTool usage goes through to_tool_list -> FunctionTool -> fn
        # However, checking the original code, it had a complex __call__ handling string/dict input.
        # FunctionTool usually handles argument parsing.
        # We'll implement the logic in crawl_url and expose that.
        # If explicitly called as an object (not via FunctionTool wrapper), we use this:
        try:
            if isinstance(input, str):
                try:
                    input_dict = json.loads(input) if input.strip().startswith("{") else {"url": input}
                except Exception:
                    input_dict = {"url": input}
            elif isinstance(input, dict):
                input_dict = input
            else:
                input_dict = {"url": str(input)}

            url = input_dict.get("url", "").strip()
            result = self.crawl_url(url)
            
            return ToolOutput(
                content=result,
                raw_input=input_dict,
                raw_output=result,
                tool_name="crawl4ai_scraper"
            )

        except Exception as e:
            msg = f"Error in Crawl4AITool: {str(e)}"
            logger.error(msg, exc_info=True)
            return ToolOutput(
                content=msg,
                raw_input={"error_input": str(input)},
                raw_output=msg,
                tool_name="crawl4ai_scraper"
            )

    def to_tool_list(self) -> List[BaseTool]:
        return [
            FunctionTool.from_defaults(
                fn=self.crawl_url,
                name="crawl4ai_scraper",
                description="Extract readable content from a webpage using its URL. Input parameter: url (string) - the webpage URL to scrape."
            )
        ]
