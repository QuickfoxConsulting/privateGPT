"""
Focused Context Window node parser using Chonkie for single-document processing.

This parser is optimized for the use case where it receives a sequence of 
page-wise Document objects all belonging to the same source file. It combines 
text across page boundaries to create semantically meaningful chunks with 
Chonkie, and then maps each chunk back to its original source page metadata.
"""
from typing import Any, Callable, List, Optional, Sequence, Tuple

from chonkie import LateChunker
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_WINDOW_SIZE = 3

class LateChunkNodeParser(NodeParser):
    """
    An optimized node parser that uses Chonkie for semantic chunking across
    page boundaries of a single document and adds a context window.

    It is designed to work on a sequence of page-nodes from one document at a time.
    """
    
    chunker: LateChunker = Field(
        default_factory=lambda: LateChunker(),
        description="The chunking engine from Chonkie.",
        exclude=True,
    )
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The number of sentences to include in each chunk.",
        gt=0,
    )
    window_size: int = Field(
        default=DEFAULT_WINDOW_SIZE, 
        description="The number of chunks on each side of a chunk to capture.",
        gt=0,
    )
    
    window_metadata_key: str = Field(
        default=DEFAULT_WINDOW_METADATA_KEY,
        description="The metadata key for the context window.",
    )
    original_text_metadata_key: str = Field(
        default=DEFAULT_OG_TEXT_METADATA_KEY,
        description="The metadata key to store the original chunk text in.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "LateChunkNodeParser"

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        window_size: int = DEFAULT_WINDOW_SIZE, 
        window_metadata_key: str = DEFAULT_WINDOW_METADATA_KEY,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "LateChunkNodeParser":

        chunker = LateChunker(chunk_size=chunk_size)
        return cls(
            chunker=chunker,
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager or CallbackManager([]),
            id_func=id_func or default_id_func,
        )
    
    
    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Parse a sequence of page-nodes from a single document into context-aware chunks,
        adding a 'page_range' key to the metadata.
        This version robustly checks for 'page_label' or 'page' in metadata.
        """
        if not nodes:
            return []

        # 1. Combine text and track offsets
        combined_text = ""
        page_offsets: List[Tuple[int, BaseNode]] = []
        page_nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Combining pages"
        )
        for page_node in page_nodes_with_progress:
            page_text = page_node.get_content()
            page_offsets.append((len(combined_text), page_node))
            combined_text += page_text + "\n\n"

        # 2. Perform chunking
        splits = self.chunker(combined_text)

        # 3. Create a temporary document to use with build_nodes_from_splits
        # We need to find a representative source document for metadata inheritance
        source_doc = None
        if page_offsets:
            first_node = page_offsets[0][1]
            # Create a temporary document that preserves the source metadata
            source_doc = Document(
                text=combined_text,
                metadata=first_node.metadata.copy(),
                id_=getattr(first_node, 'ref_doc_id', None) or first_node.metadata.get('doc_id', None)
            )
        
        # Extract text splits for build_nodes_from_splits
        text_splits = [split.text for split in splits]
        
        # Use LlamaIndex's built-in function to create nodes properly
        chunk_nodes = build_nodes_from_splits(
            text_splits,
            source_doc,
            id_func=self.id_func,
        )
        
        # 4. Add page range metadata to each chunk
        current_offset = 0
        for i, (node, split) in enumerate(zip(chunk_nodes, splits)):
            split_text = split.text
            start_offset = combined_text.find(split_text, current_offset)
            if start_offset == -1:
                current_offset += len(split_text)
                continue
            
            end_offset = start_offset + len(split_text)

            start_node = self._find_source_node(start_offset, page_offsets)
            end_node = self._find_source_node(end_offset, page_offsets)

            # Check for 'page_label' first, then fall back to 'page'
            start_page = start_node.metadata.get("page_label") or start_node.metadata.get("page")
            end_page = end_node.metadata.get("page_label") or end_node.metadata.get("page")

            # Add page range metadata while preserving existing metadata
            if start_page is not None:
                start_page_str = str(start_page)
                if end_page is not None and str(end_page) != start_page_str:
                    node.metadata["page_range"] = f"{start_page_str}-{str(end_page)}"
                else:
                    node.metadata["page_range"] = start_page_str
            
            current_offset = start_offset + len(split_text)

        # 5. Add context window and excluded metadata keys
        self._add_context_window_to_nodes(chunk_nodes)

        return chunk_nodes

    def _find_source_node(self, offset: int, page_offsets: List[Tuple[int, BaseNode]]) -> BaseNode:
        """Find the document corresponding to a given character offset."""
        for start_offset, doc in reversed(page_offsets):
            if offset >= start_offset:
                return doc
        return page_offsets[0][1] # Fallback to the first page

    def _add_context_window_to_nodes(self, nodes: List[BaseNode]):
        """Iterate through nodes and add the 'window' metadata."""
        for i, node in enumerate(nodes):
            # Store the original chunk text for semantic search
            node.metadata[self.original_text_metadata_key] = node.get_content()
            
            # Create context window with surrounding chunks
            window_nodes = nodes[
                max(0, i - self.window_size) :
                min(len(nodes), i + self.window_size + 1)
            ]
            window_text = " ".join([n.get_content() for n in window_nodes])
            node.metadata[self.window_metadata_key] = window_text
            
            # IMPORTANT: Only exclude window metadata from embeddings
            # Keep original_text available for semantic search embedding
            node.excluded_embed_metadata_keys.extend([
                self.window_metadata_key
            ])
            node.excluded_llm_metadata_keys.extend([
                self.window_metadata_key,
                self.original_text_metadata_key
            ])