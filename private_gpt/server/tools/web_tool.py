import re
import json
import asyncio
import logging
from typing import Any, Dict
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler
from concurrent.futures import ThreadPoolExecutor

from llama_index.core.tools import BaseTool
from llama_index.core.tools.types import ToolMetadata, ToolOutput

logger = logging.getLogger(__name__)

class Crawl4AITool(BaseTool):
    """Tool to extract readable content from a webpage."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="crawl4ai_scraper",
            description="Extract readable content from a webpage using its URL."
        )

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
                result = await crawler.arun(
                    url=url,
                    word_count_threshold=10,
                    extraction_strategy="NoExtractionStrategy",
                    chunking_strategy="RegexChunking",
                    bypass_cache=True
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

    def __call__(self, input: Any) -> ToolOutput:
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

            if not self._validate_url(url):
                msg = f"Invalid URL format: {url}"
                return ToolOutput(
                    content=msg,
                    raw_input=input_dict,
                    raw_output=msg,
                    tool_name="crawl4ai_scraper"
                )

            # Run the crawl in a separate thread with its own event loop
            result = self._run_in_thread(url)

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
