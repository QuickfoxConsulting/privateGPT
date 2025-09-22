"""Passthrough node parser for already chunked documents."""

from typing import Any, List, Optional, Sequence, Callable

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits, default_id_func
from llama_index.core.schema import BaseNode, Document
from llama_index.core.utils import get_tqdm_iterable


class PageByPageNodeParser(NodeParser):
    """A node parser that assumes documents are already pre-chunked.

    It wraps each document's text into a single node, preserving existing metadata.
    """

    @classmethod
    def class_name(cls) -> str:
        return "PageByPageNodeParser"

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        id_func: Optional[Callable[[int, Document], str]] = None,
    ) -> "PageByPageNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or default_id_func

        return cls(
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
        """Wrap each pre-chunked document into a node."""
        all_nodes: List[BaseNode] = []
        docs_with_progress = get_tqdm_iterable(documents, show_progress, "Parsing documents")

        for doc in docs_with_progress:
            nodes = self.build_window_nodes_from_documents([doc])
            all_nodes.extend(nodes)

        return all_nodes

    def build_window_nodes_from_documents(
        self, documents: Sequence[Document]
    ) -> List[BaseNode]:
        """Convert each document to a single node without re-chunking."""
        all_nodes: List[BaseNode] = []

        for doc in documents:
            nodes = build_nodes_from_splits(
                [doc.text],
                doc,
                id_func=self.id_func,
            )
            
            # Ensure document IDs are preserved
            for node in nodes:
                if hasattr(doc, 'doc_id') and doc.doc_id:
                    node.metadata["document_id"] = doc.doc_id
                    node.metadata["doc_id"] = doc.doc_id
                elif doc.metadata and "document_id" in doc.metadata:
                    node.metadata["document_id"] = doc.metadata["document_id"]
                    node.metadata["doc_id"] = doc.metadata["document_id"]
                elif doc.metadata and "doc_id" in doc.metadata:
                    node.metadata["document_id"] = doc.metadata["doc_id"]
                    node.metadata["doc_id"] = doc.metadata["doc_id"]
            all_nodes.extend(nodes)

        return all_nodes
