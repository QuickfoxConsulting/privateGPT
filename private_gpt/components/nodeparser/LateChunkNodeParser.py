import uuid
import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple

from chonkie import LateChunker
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    default_id_func,
)
from llama_index.core.schema import BaseNode, Document, NodeRelationship, RelatedNodeInfo, TextNode
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"
DEFAULT_CHUNK_SIZE = 512
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
        
        doc_ids = {node.ref_doc_id for node in nodes}
        if len(doc_ids) > 1:
            raise ValueError("LateChunkNodeParser is designed to process nodes from a single document at a time.")
        
        # 1. Combine text and track page offsets
        combined_text = ""
        page_offsets: List[Tuple[int, BaseNode]] = []
        for page_node in get_tqdm_iterable(nodes, show_progress, "Combining pages"):
            page_text = page_node.get_content()
            page_offsets.append((len(combined_text), page_node))
            combined_text += page_text + "\n\n"

        # 2. Perform chunking using Chonkie
        splits = self.chunker(combined_text)

        # 3. Create nodes with correct metadata from the start
        chunk_nodes: List[BaseNode] = []
        
        # This will be the reference document for all new chunk nodes
        first_node = nodes[0]
        ref_doc_id = first_node.ref_doc_id or str(uuid.uuid4())
        base_metadata = first_node.metadata.copy()

        # Create source_doc for ID generation
        source_doc = Document(
            id_=ref_doc_id,
            metadata=base_metadata,
            text=combined_text,
        )

        for i, split in enumerate(get_tqdm_iterable(splits, show_progress, "Creating chunk nodes")):
            start_char_idx = split.start_index
            end_char_idx = split.end_index
            
            # Use helper to find the corresponding source page nodes
            start_page_node = self._find_source_node(start_char_idx, page_offsets)
            end_page_node = self._find_source_node(end_char_idx - 1, page_offsets)

            # Construct metadata for the new chunk node
            metadata = {}
            if self.include_metadata:
                metadata = base_metadata.copy()
                start_page = start_page_node.metadata.get("page_label", start_page_node.metadata.get("page"))
                end_page = end_page_node.metadata.get("page_label", end_page_node.metadata.get("page"))

                if start_page is not None:
                    metadata["page_label"] = start_page # Use page_label as the standard
                    if end_page is not None and str(end_page) != str(start_page):
                        metadata["page_range"] = f"{start_page}-{end_page}"
                    else:
                        metadata["page_range"] = str(start_page)

            node_id = self.id_func(i, source_doc)
            node = TextNode(
                id_=node_id,
                text=split.text,
                metadata=metadata,
                relationships={
                    NodeRelationship.SOURCE: RelatedNodeInfo(node_id=source_doc.id_)
                },
            )
            chunk_nodes.append(node)
        
        # 4. Add chunk-level relationships (prev/next) if enabled
        if self.include_prev_next_rel:
            for i, node in enumerate(chunk_nodes):
                if i > 0:
                    node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                        node_id=chunk_nodes[i-1].node_id
                    )
                if i < len(chunk_nodes) - 1:
                    node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                        node_id=chunk_nodes[i+1].node_id
                    )

        # 5. Add context window and configure metadata for embedding/LLM
        self._add_context_window_to_nodes(chunk_nodes)

        return chunk_nodes

    def _find_source_node(self, offset: int, page_offsets: List[Tuple[int, BaseNode]]) -> BaseNode:
        """Find the source page node corresponding to a given character offset."""
        # This logic is correct. No changes needed.
        for start_offset, doc in reversed(page_offsets):
            if offset >= start_offset:
                return doc
        return page_offsets[0][1]

    def _add_context_window_to_nodes(self, nodes: List[TextNode]):
        """
        Iterate through nodes to add a context window and set metadata exclusion keys.
        """
        for i, node in enumerate(nodes):
            original_text = node.get_content()
            
            window_start = max(0, i - self.window_size)
            window_end = min(len(nodes), i + self.window_size + 1)
            
            window_nodes = nodes[window_start:window_end]
            # Use a more robust joiner than just a space
            window_text = "\n\n---\n\n".join(
                [n.get_content() for n in window_nodes]
            )
            
            node.set_content(original_text)  # Keep original content as primary
            
            if self.include_metadata:
                node.metadata[self.original_text_metadata_key] = original_text
                node.metadata[self.window_metadata_key] = window_text

                # Exclude window metadata from embedding to avoid diluting the vector representations
                if self.window_metadata_key not in node.excluded_embed_metadata_keys:
                    node.excluded_embed_metadata_keys.append(self.window_metadata_key)
                
                # Exclude original text from LLM metadata to avoid redundancy
                if self.original_text_metadata_key not in node.excluded_llm_metadata_keys:
                    node.excluded_llm_metadata_keys.append(self.original_text_metadata_key)