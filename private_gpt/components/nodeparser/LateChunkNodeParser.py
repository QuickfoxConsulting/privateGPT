"""Late chunking node parser using Chonkie."""

from typing import Any, Callable, List, Optional, Sequence
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

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 128
DEFAULT_CONTEXT_WINDOW = 3
DEFAULT_CHUNK_METADATA_KEY = "chunk_info"
DEFAULT_ORIGINAL_TEXT_KEY = "original_text"

class LateChunkingNodeParser(NodeParser):
    """Late chunking node parser using Chonkie.
    
    Splits documents into nodes with dynamic late chunking capability.
    Each node contains contextual information about its chunks and neighboring content.
    
    Args:
        chunk_size (int): Maximum size of each chunk
        chunk_overlap (int): Overlap between consecutive chunks
        context_window (int): Number of surrounding chunks to include as context
        include_metadata (bool): Whether to include metadata in nodes
        include_prev_next_rel (bool): Whether to include prev/next relationships
    """

    chunker: LateChunker = Field(
        default_factory=lambda: LateChunker(),
        description="The chunking engine to use for text splitting.",
        exclude=True,
    )
    
    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The maximum size of each chunk.",
        gt=0,
    )
    
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        description="The overlap size between consecutive chunks.",
        ge=0,
    )
    
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The number of surrounding chunks to include as context.",
        gt=0,
    )
    
    chunk_metadata_key: str = Field(
        default=DEFAULT_CHUNK_METADATA_KEY,
        description="The metadata key to store chunk information.",
    )
    
    original_text_key: str = Field(
        default=DEFAULT_ORIGINAL_TEXT_KEY,
        description="The metadata key to store the original text.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "LateChunkingNodeParser"

    @classmethod
    def from_defaults(
        cls,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        chunk_metadata_key: str = DEFAULT_CHUNK_METADATA_KEY,
        original_text_key: str = DEFAULT_ORIGINAL_TEXT_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "LateChunkingNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func
        
        chunker = LateChunker(chunk_size=chunk_size, mode="sentence")

        return cls(
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            context_window=context_window,
            chunk_metadata_key=chunk_metadata_key,
            original_text_key=original_text_key,
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
        """Parse documents into nodes with late chunking."""
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            chunked_nodes = self.build_chunked_nodes_from_documents([node])
            all_nodes.extend(chunked_nodes)

        return all_nodes

    def build_chunked_nodes_from_documents(
        self, documents: Sequence[Document]
    ) -> List[BaseNode]:
        """Build nodes with late chunking from documents."""
        all_nodes: List[BaseNode] = []
        
        for doc in documents:
            text = doc.text
            
            # Perform late chunking
            chunks = self.chunker(text)
            
            # Build base nodes from chunks
            nodes = build_nodes_from_splits(
                chunks,
                doc,
                id_func=self.id_func,
            )

            # Add chunk context and metadata to each node
            for i, node in enumerate(nodes):
                # Get surrounding chunks as context
                context_start = max(0, i - self.context_window)
                context_end = min(i + self.context_window + 1, len(nodes))
                context_chunks = nodes[context_start:context_end]
                
                # Store chunk metadata
                chunk_info = {
                    "chunk_index": i,
                    "total_chunks": len(nodes),
                    "context_chunks": [n.text for n in context_chunks],
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
                
                node.metadata[self.chunk_metadata_key] = chunk_info
                node.metadata[self.original_text_key] = node.text

                # Exclude certain metadata from embedding and LLM
                node.excluded_embed_metadata_keys.extend(
                    [self.chunk_metadata_key, self.original_text_key]
                )
                node.excluded_llm_metadata_keys.extend(
                    [self.chunk_metadata_key, self.original_text_key]
                )

            all_nodes.extend(nodes)

        return all_nodes