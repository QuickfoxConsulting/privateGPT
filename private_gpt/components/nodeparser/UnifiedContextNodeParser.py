from typing import Any, List, Optional, Sequence
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, Document, TextNode
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.bridge.pydantic import Field

class UnifiedContextNodeParser(NodeParser):
    """
    The Unified Context Node Parser.
    Combines:
    1. Late Chunking (Page Stitching)
    2. Hierarchical Splitting (Parent-Child)
    3. Metadata Backfilling (Cross-context)
    """
    hierarchical_parser: HierarchicalNodeParser = Field(description="The underlying hierarchical parser.")
    
    @classmethod
    def from_defaults(
        cls,
        chunk_sizes: Optional[List[int]] = None,
        chunk_overlap: int = 50,
        **kwargs: Any
    ) -> "UnifiedContextNodeParser":
        chunk_sizes = chunk_sizes or [1024, 512, 256]
        hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes,
            chunk_overlap=chunk_overlap,
            include_metadata=True,
            include_prev_next_rel=True
        )
        return cls(hierarchical_parser=hierarchical_parser)

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        if not nodes:
            return []

        # 1. Late Chunking: combine all pages into one virtual document
        # This prevents the "Page Boundary Gap"
        combined_text = ""
        full_metadata = nodes[0].metadata.copy()
        
        # Track pages for backfilling
        pages = []
        for node in nodes:
            combined_text += node.get_content() + "\n\n"
            p = node.metadata.get("page_label") or node.metadata.get("page")
            if p:
                pages.append(str(p))
        
        if pages:
            full_metadata["all_pages"] = ", ".join(pages)
            full_metadata["page_count"] = len(pages)

        # Create one giant document for the hierarchy to work on
        master_doc = Document(
            text=combined_text,
            metadata=full_metadata,
        )

        # 2. Hierarchical Parsing
        # This creates the Parent (Broad) and Child (Precise) nodes
        all_nodes = self.hierarchical_parser.get_nodes_from_documents([master_doc])
        
        return all_nodes

    @classmethod
    def class_name(cls) -> str:
        return "UnifiedContextNodeParser"
