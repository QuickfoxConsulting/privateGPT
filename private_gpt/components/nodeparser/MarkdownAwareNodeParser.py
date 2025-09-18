"""Markdown-aware node parser for better handling of LlamaParse converted PDFs."""

from typing import Any, Callable, List, Optional, Sequence
import re

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core.utils import get_tqdm_iterable


class MarkdownAwareNodeParser(NodeParser):
    """Markdown-aware node parser that respects document structure from LlamaParse.
    
    This parser is designed to handle markdown content from LlamaParse conversion
    of unstructured PDFs, preserving page references and document structure.
    """

    chunk_size: int = Field(
        default=1024,
        description="Target chunk size in characters",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=100,
        description="Overlap between chunks in characters",
        ge=0,
    )
    preserve_headers: bool = Field(
        default=True,
        description="Whether to preserve markdown headers as chunk boundaries",
    )
    min_chunk_size: int = Field(
        default=100,
        description="Minimum chunk size in characters",
        gt=0,
    )

    @classmethod
    def class_name(cls) -> str:
        return "MarkdownAwareNodeParser"

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = 1024,
        chunk_overlap: int = 100,
        preserve_headers: bool = True,
        min_chunk_size: int = 100,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "MarkdownAwareNodeParser":
        callback_manager = callback_manager or CallbackManager([])

        id_func = id_func or default_id_func

        return cls(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preserve_headers=preserve_headers,
            min_chunk_size=min_chunk_size,
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
        """Parse document into nodes."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            nodes = self.build_nodes_from_document(node)
            all_nodes.extend(nodes)

        return all_nodes

    def _split_by_headers(self, text: str) -> List[str]:
        """Split text by markdown headers while preserving them."""
        # Find all header positions
        header_pattern = r'^#{1,6}\s+.*$'
        header_matches = list(re.finditer(header_pattern, text, re.MULTILINE))
        
        if not header_matches:
            return [text]
        
        # Split text at headers
        splits = []
        last_end = 0
        
        for match in header_matches:
            # Add text before header (if any)
            if match.start() > last_end:
                splits.append(text[last_end:match.start()])
            
            # Add header and text until next header
            header_start = match.start()
            next_header_start = len(text)
            for next_match in header_matches:
                if next_match.start() > match.start():
                    next_header_start = next_match.start()
                    break
            
            splits.append(text[header_start:next_header_start])
            last_end = next_header_start
        
        # Add remaining text
        if last_end < len(text):
            splits.append(text[last_end:])
        
        return [split.strip() for split in splits if split.strip()]

    def _chunk_text_with_overlap(self, text: str) -> List[str]:
        """Chunk text with overlap, respecting sentence boundaries."""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine chunk end
            end = min(start + self.chunk_size, len(text))
            
            # Try to find a sentence boundary near the end
            if end < len(text):
                # Look for sentence ending punctuation followed by space or end of text
                for i in range(end, start + self.min_chunk_size, -1):
                    if i < len(text) and text[i] in '.!?;' and (i + 1 >= len(text) or text[i + 1].isspace()):
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
            if start <= 0:  # Prevent infinite loop
                start = end
        
        return chunks

    def build_nodes_from_document(self, document: Document) -> List[BaseNode]:
        """Build nodes from document, preserving page references and structure."""
        text = document.text
        
        # Extract page information from metadata if available
        page_info = ""
        if document.metadata:
            page_num = document.metadata.get("page")
            filename = document.metadata.get("filename", "")
            if page_num:
                page_info = f"[Page {page_num}"
                if filename:
                    page_info += f" of {filename}"
                page_info += "] "
        
        chunks = []
        
        if self.preserve_headers:
            # First split by headers
            header_sections = self._split_by_headers(text)
            
            # Then chunk each section
            for section in header_sections:
                if len(section) <= self.chunk_size:
                    chunks.append(section)
                else:
                    section_chunks = self._chunk_text_with_overlap(section)
                    chunks.extend(section_chunks)
        else:
            # Direct chunking
            chunks = self._chunk_text_with_overlap(text)
        
        # Build nodes from chunks
        nodes = build_nodes_from_splits(
            chunks,
            document,
            id_func=self.id_func,
        )
        
        # Add page reference to each node and preserve document structure
        for i, node in enumerate(nodes):
            # Add page information to the beginning of each chunk
            if page_info and not node.text.startswith('['):
                node.text = page_info + node.text
            
            # Preserve original metadata
            if document.metadata:
                node.metadata.update(document.metadata)
            
            # Add chunk sequence information
            node.metadata["chunk_sequence"] = i
            node.metadata["total_chunks"] = len(nodes)
            
            # Preserve relationships
            if i > 0:
                node.relationships["previous"] = nodes[i-1].node_id
            if i < len(nodes) - 1:
                node.relationships["next"] = nodes[i+1].node_id
        
        return nodes