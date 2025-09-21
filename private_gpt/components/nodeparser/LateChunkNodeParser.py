"""
Focused Context Window node parser using Chonkie for single-document processing.

This parser is optimized for the use case where it receives a sequence of 
page-wise Document objects all belonging to the same source file. It combines 
text across page boundaries to create semantically meaningful chunks with 
Chonkie, and then maps each chunk back to its original source page metadata.
"""
import uuid
import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple

from chonkie import LateChunker
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CONTEXT_WINDOW = 3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class LateChunkNodeParser(NodeParser):
    """
    An optimized node parser that uses Chonkie for semantic chunking across
    page boundaries of a single document and adds a context window.

    It is designed to work on a sequence of page-nodes from one document at a time.
    """
    
    chunker: LateChunker = Field(
        description="The chunking engine from Chonkie.",
        exclude=True,
    )
    
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The number of sentences to include in each chunk.",
        gt=0,
    )
    window_size: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
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
        window_size: int = DEFAULT_CONTEXT_WINDOW,
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
            chunk_size=chunk_size,
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
        Parse a sequence of page-nodes from a single document into context-aware chunks.
        """
        if not nodes:
            return []

        # 1. Combine text and track page offsets
        combined_text = ""
        page_offsets: List[Tuple[int, BaseNode]] = []
        for page_node in get_tqdm_iterable(nodes, show_progress, "Combining pages"):
            page_text = page_node.get_content()
            page_offsets.append((len(combined_text), page_node))
            combined_text += page_text + "\n\n"

        # 2. Perform chunking
        splits = self.chunker(combined_text)
        text_splits = [split.text for split in splits]

        # 3. Create a temporary source document for ID and relationship linking
        first_node = nodes[0]
        source_doc_id = first_node.ref_doc_id or str(uuid.uuid4())
        
        # We pass a copy of the first node's metadata. This will be corrected later.
        source_doc = Document(
            text=combined_text,
            id_=source_doc_id,
            metadata=first_node.metadata.copy(),
        )

        # 4. Use LlamaIndex utility to build nodes with correct relationships and ref_doc_id
        chunk_nodes = build_nodes_from_splits(
            text_splits,
            source_doc,
            id_func=self.id_func,
        )
        
        # 5. Correct the metadata for each chunk node
        current_offset = 0
        for node in get_tqdm_iterable(chunk_nodes, show_progress, "Correcting metadata"):
            node_text = node.get_content()
            start_offset = combined_text.find(node_text, current_offset)
            if start_offset == -1:
                continue # Should not happen in practice
            end_offset = start_offset + len(node_text)
            
            start_page_node = self._find_source_node(start_offset, page_offsets)
            end_page_node = self._find_source_node(end_offset, page_offsets)

            start_page = start_page_node.metadata.get("page_label") or start_page_node.metadata.get("page")
            end_page = end_page_node.metadata.get("page_label") or end_page_node.metadata.get("page")

            # --- METADATA CORRECTION ---
            # Overwrite the inherited metadata with accurate, chunk-specific values.
            if start_page is not None:
                node.metadata["page"] = start_page
                start_page_str = str(start_page)
                if end_page is not None and str(end_page) != start_page_str:
                    node.metadata["page_range"] = f"{start_page_str}-{str(end_page)}"
                else:
                    node.metadata["page_range"] = start_page_str

            # Correct the page-level relationships
            node.metadata["prev_page"] = start_page_node.metadata.get("prev_page")
            node.metadata["next_page"] = end_page_node.metadata.get("next_page")
            
            current_offset = end_offset

        # 6. Add context window and configure metadata for embedding/LLM
        self._add_context_window_to_nodes(chunk_nodes)

        return chunk_nodes

    def _find_source_node(self, offset: int, page_offsets: List[Tuple[int, BaseNode]]) -> BaseNode:
        """Find the source page node corresponding to a given character offset."""
        for start_offset, doc in reversed(page_offsets):
            if offset >= start_offset:
                return doc
        return page_offsets[0][1]

    def _add_context_window_to_nodes(self, nodes: List[BaseNode]):
        """
        Iterate through nodes to add a context window and set metadata exclusion keys.
        """
        for i, node in enumerate(nodes):
            original_text = node.get_content()
            node.metadata[self.original_text_metadata_key] = original_text

            window_nodes = nodes[
                max(0, i - self.window_size) : 
                min(len(nodes), i + self.window_size + 1)
            ]
            window_text = " ".join([n.get_content() for n in window_nodes])
            node.metadata[self.window_metadata_key] = window_text

            if self.window_metadata_key not in node.excluded_embed_metadata_keys:
                node.excluded_embed_metadata_keys.append(self.window_metadata_key)
            
            if self.original_text_metadata_key not in node.excluded_llm_metadata_keys:
                node.excluded_llm_metadata_keys.append(self.original_text_metadata_key)