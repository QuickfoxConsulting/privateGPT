"""Page-based window node parser."""

from typing import Any, Callable, List, Optional, Sequence

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import (
    build_nodes_from_splits,
    default_id_func,
)
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable

DEFAULT_WINDOW_SIZE = 1  # Number of pages on each side of a page to include in window
DEFAULT_WINDOW_METADATA_KEY = "window"
DEFAULT_OG_TEXT_METADATA_KEY = "original_text"


class PageWindowNodeParser(NodeParser):
    """Page window node parser.

    This parser takes a list of Documents, where each Document is assumed to be a page.
    It creates a single Node per page.

    For each node, it attaches a "window" of text from the surrounding pages in
    the metadata. This is useful for retrieval, where a page's content might be
    better understood in the context of its neighboring pages.

    Args:
        window_size (int): The number of pages on each side of a given page
            to include in its window.
        window_metadata_key (str): The key in the metadata to store the window.
        original_text_metadata_key (str): The key in the metadata to store the
            original page text.
    """

    window_size: int = Field(
        default=DEFAULT_WINDOW_SIZE,
        description="The number of pages on each side of a page to capture.",
        gt=0,
    )
    window_metadata_key: str = Field(
        default=DEFAULT_WINDOW_METADATA_KEY,
        description="The metadata key to store the page window under.",
    )
    original_text_metadata_key: str = Field(
        default=DEFAULT_OG_TEXT_METADATA_KEY,
        description="The metadata key to store the original page text in.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "PageWindowNodeParser"

    @classmethod
    def from_defaults(
        cls,
        window_size: int = DEFAULT_WINDOW_SIZE,
        window_metadata_key: str = DEFAULT_WINDOW_METADATA_KEY,
        original_text_metadata_key: str = DEFAULT_OG_TEXT_METADATA_KEY,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "PageWindowNodeParser":
        """
        Initializes the parser with default settings.

        Args:
            window_size (int): The number of pages on each side to include in the window.
            window_metadata_key (str): The metadata key for the window text.
            original_text_metadata_key (str): The metadata key for the original page text.
            include_metadata (bool): Whether to include metadata in nodes.
            include_prev_next_rel (bool): Whether to include prev/next relationships.
            callback_manager (Optional[CallbackManager]): The callback manager.
            id_func (Optional[Callable]): The function to generate node IDs.
        """
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func

        return cls(
            window_size=window_size,
            window_metadata_key=window_metadata_key,
            original_text_metadata_key=original_text_metadata_key,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _parse_nodes(
        self,
        documents: Sequence[Document],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Parse a list of documents (pages) into nodes with window metadata.

        Args:
            documents (Sequence[Document]): The documents to parse, where each
                document represents a page.
            show_progress (bool): Whether to display a progress bar.

        Returns:
            List[BaseNode]: A list of nodes, each with window metadata.
        """
        # First, convert each document into a single node.
        # The text of the document is the "chunk"
        all_nodes: List[BaseNode] = []
        docs_with_progress = get_tqdm_iterable(
            documents, show_progress, "Parsing documents into nodes"
        )
        for doc in docs_with_progress:
            # The 'splits' are just the single text content of the document
            nodes = build_nodes_from_splits(
                [doc.text],
                doc,
                id_func=self.id_func,
            )
            all_nodes.extend(nodes)

        # Now, iterate through the generated nodes to add the window metadata
        for i, node in enumerate(all_nodes):
            window_start = max(0, i - self.window_size)
            window_end = min(i + self.window_size + 1, len(all_nodes))
            
            # The window is the concatenated text of the nodes in the window range
            window_nodes = all_nodes[window_start:window_end]
            window_text = " ".join([n.text for n in window_nodes])

            # Store the window and original text in metadata
            node.metadata[self.window_metadata_key] = window_text
            node.metadata[self.original_text_metadata_key] = node.text

            # Exclude the new metadata from embedding and LLM inputs
            # to prevent contamination of the core node content
            node.excluded_embed_metadata_keys.extend(
                [self.window_metadata_key, self.original_text_metadata_key]
            )
            node.excluded_llm_metadata_keys.extend(
                [self.window_metadata_key, self.original_text_metadata_key]
            )
            
        return all_nodes