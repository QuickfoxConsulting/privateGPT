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
from llama_index.core.node_parser.node_utils import default_id_func
from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_CONTEXT_WINDOW = 3

class ChonkieContextWindowNodeParser(NodeParser):
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
    
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The number of surrounding chunks to include on each side.",
        gt=0,
    )
    
    window_metadata_key: str = Field(
        default=DEFAULT_WINDOW_METADATA_KEY,
        description="The metadata key for the context window.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "ChonkieContextWindowNodeParser"

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "ChonkieContextWindowNodeParser":
        chunker = LateChunker(chunk_size=chunk_size)
        return cls(
            chunker=chunker,
            context_window=context_window,
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

        # 3. Map chunks back to source pages
        chunk_nodes: List[TextNode] = []
        current_offset = 0
        splits_with_progress = get_tqdm_iterable(
            splits, show_progress, "Creating nodes"
        )
        for i, split in enumerate(splits_with_progress):
            split_text = split.text
            start_offset = combined_text.find(split_text, current_offset)
            if start_offset == -1:
                continue
            
            # --- REVISED LOGIC TO CHECK FOR 'page' AND 'page_label' ---
            end_offset = start_offset + len(split_text)

            start_node = self._find_source_node(start_offset, page_offsets)
            end_node = self._find_source_node(end_offset, page_offsets)

            # Check for 'page_label' first, then fall back to 'page'
            start_page = start_node.metadata.get("page_label") or start_node.metadata.get("page")
            end_page = end_node.metadata.get("page_label") or end_node.metadata.get("page")

            # Inherit metadata from the start node
            new_metadata = start_node.metadata.copy()
            
            # Ensure document IDs are properly preserved using consistent key "doc_id"
            if "doc_id" in start_node.metadata and start_node.metadata["doc_id"]:
                new_metadata["doc_id"] = start_node.metadata["doc_id"]
            elif "document_id" in start_node.metadata and start_node.metadata["document_id"]:
                new_metadata["doc_id"] = start_node.metadata["document_id"]
            elif hasattr(start_node, 'ref_doc_id') and start_node.ref_doc_id:
                new_metadata["doc_id"] = start_node.ref_doc_id
            
            # Also preserve the original document_id for compatibility
            if "doc_id" in new_metadata:
                new_metadata["document_id"] = new_metadata["doc_id"]

            # Create and add the new 'page_range' key
            if start_page is not None:
                # Ensure pages are converted to string for consistent formatting
                start_page_str = str(start_page)
                if end_page is not None and str(end_page) != start_page_str:
                    new_metadata["page_range"] = f"{start_page_str}-{str(end_page)}"
                else:
                    new_metadata["page_range"] = start_page_str
            # --- END REVISED LOGIC ---

            # Create the new node with the augmented metadata
            node = TextNode(
                text=split_text,
                metadata=new_metadata,
            )
            node.id_ = self.id_func(i, start_node)
            # Ensure the node has the correct ref_doc_id set
            if "doc_id" in new_metadata:
                node.ref_doc_id = new_metadata["doc_id"]
            elif hasattr(start_node, 'ref_doc_id') and start_node.ref_doc_id:
                node.ref_doc_id = start_node.ref_doc_id
            chunk_nodes.append(node)
            
            current_offset = start_offset + len(split_text)

        # 4. Add context window
        self._add_context_window_to_nodes(chunk_nodes)

        return chunk_nodes

    def _find_source_node(self, offset: int, page_offsets: List[Tuple[int, BaseNode]]) -> BaseNode:
        """Find the document corresponding to a given character offset."""
        for start_offset, doc in reversed(page_offsets):
            if offset >= start_offset:
                return doc
        return page_offsets[0][1] # Fallback to the first page

    def _add_context_window_to_nodes(self, nodes: List[TextNode]):
        """Iterate through nodes and add the 'window' metadata."""
        for i, node in enumerate(nodes):
            window_nodes = nodes[
                max(0, i - self.context_window) : 
                min(len(nodes), i + self.context_window + 1)
            ]
            window_text = " ".join([n.get_content() for n in window_nodes])
            node.metadata[self.window_metadata_key] = window_text