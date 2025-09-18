"""Adaptive node parser that selects the best strategy based on content type."""

from typing import Any, Callable, List, Optional, Sequence
import re

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits, default_id_func
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable

from private_gpt.components.nodeparser.MarkdownAwareNodeParser import MarkdownAwareNodeParser
from private_gpt.components.nodeparser.SentenceChunkNodeParser import SentenceChunkWindowNodeParser


class AdaptiveNodeParser(NodeParser):
    """Adaptive node parser that selects the best strategy based on content analysis."""

    chunk_size: int = Field(default=1024, description="Target chunk size in characters", gt=0)
    chunk_overlap: int = Field(default=100, description="Overlap between chunks in characters", ge=0)
    markdown_threshold: float = Field(
        default=0.3,
        description="Threshold for markdown content detection (ratio of markdown elements)",
        ge=0.0,
        le=1.0,
    )

    @classmethod
    def class_name(cls) -> str:
        return "AdaptiveNodeParser"

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        markdown_threshold: float = 0.3,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "AdaptiveNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        id_func = id_func or default_id_func

        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            markdown_threshold=markdown_threshold,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse document into nodes using adaptive strategy."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            # Analyze content to determine best parsing strategy
            parser = self._select_parser(node.text)
            parsed_nodes = parser._parse_nodes([node], show_progress=False)
            all_nodes.extend(parsed_nodes)

        return all_nodes

    def _select_parser(self, text: str) -> NodeParser:
        """Select the appropriate parser based on content analysis."""
        # Calculate markdown elements ratio
        markdown_elements = len(re.findall(r'^#{1,6}\s+.*$|^[\*\-\+]\s+.*$|^(\d+\.)\s+.*$|^```.*?```$', 
                                          text, re.MULTILINE))
        total_lines = len(text.split('\n'))
        
        markdown_ratio = markdown_elements / max(total_lines, 1)
        
        # Select parser based on content type
        if markdown_ratio >= self.markdown_threshold:
            # Use markdown-aware parser for markdown-like content
            return MarkdownAwareNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:
            # Use sentence-based parser for plain text
            return SentenceChunkWindowNodeParser.from_defaults(
                chunk_size=max(1, self.chunk_size // 100),  # Convert to sentences
                window_size=3,
                chunk_overlap=self.chunk_overlap,
            )

    def build_nodes_from_document(self, document: Document) -> List[BaseNode]:
        """Build nodes from document using adaptive strategy."""
        parser = self._select_parser(document.text)
        if hasattr(parser, 'build_nodes_from_document'):
            return parser.build_nodes_from_document(document)
        else:
            # Fallback to standard parsing
            return parser._parse_nodes([document], show_progress=False)