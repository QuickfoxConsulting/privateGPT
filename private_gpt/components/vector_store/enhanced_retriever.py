"""Enhanced retriever for better handling of cross-page content and page-aware retrieval."""

from typing import List, Optional, Dict, Any
from llama_index.core.schema import NodeWithScore
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQuery

class EnhancedFileRetriever(BaseRetriever):
    """Enhanced file retriever that handles cross-page content and preserves page context."""

    def __init__(
        self,
        index: VectorStoreIndex,
        file_name: str,
        similarity_top_k: int = 5,
        enable_cross_page_retrieval: bool = True,
        preserve_page_context: bool = True,
        context_window_expansion: float = 1.5,
    ):
        self.index = index
        self.file_name = file_name
        self.similarity_top_k = similarity_top_k
        self.enable_cross_page_retrieval = enable_cross_page_retrieval
        self.preserve_page_context = preserve_page_context
        self.context_window_expansion = context_window_expansion
        
        # Get the base retriever from the index
        self.base_retriever = index.as_retriever(similarity_top_k=similarity_top_k)

    def _retrieve(self, query_bundle: Any) -> List[NodeWithScore]:
        """Retrieve nodes with enhanced cross-page context handling."""
        # Get initial results from base retriever
        initial_nodes = self.base_retriever.retrieve(query_bundle)
        
        # Filter nodes by file name
        filtered_nodes = [
            node for node in initial_nodes 
            if node.node.metadata.get("filename") == self.file_name or 
               node.node.metadata.get("file_name") == self.file_name
        ]
        
        if not self.enable_cross_page_retrieval:
            return filtered_nodes
            
        # Enhance with cross-page context if enabled
        enhanced_nodes = self._enhance_with_cross_page_context(filtered_nodes)
        
        # Sort by similarity score and return top k
        enhanced_nodes.sort(key=lambda x: x.score, reverse=True)
        return enhanced_nodes[:self.similarity_top_k]

    def _enhance_with_cross_page_context(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Enhance nodes with cross-page context and better page handling."""
        if not nodes:
            return nodes
            
        # Group nodes by document and page information
        page_groups = self._group_nodes_by_page_context(nodes)
        
        # Expand context for each group if needed
        enhanced_nodes = []
        for group in page_groups:
            if self.preserve_page_context:
                # Add page context information to node metadata
                for node_with_score in group:
                    node = node_with_score.node
                    # Extract page information
                    page_info = node.metadata.get("page_info") or node.metadata.get("page")
                    if page_info:
                        # Update node text to include page information
                        if not node.text.startswith("[Page"):
                            node.text = f"[Page {page_info}] {node.text}"
                    
                    enhanced_nodes.append(node_with_score)
            else:
                enhanced_nodes.extend(group)
                
        return enhanced_nodes

    def _group_nodes_by_page_context(self, nodes: List[NodeWithScore]) -> List[List[NodeWithScore]]:
        """Group nodes by their page context for better handling of cross-page content."""
        groups = []
        processed_node_ids = set()
        
        for node_with_score in nodes:
            if node_with_score.node.node_id in processed_node_ids:
                continue
                
            # Get page information for this node
            current_page = self._extract_page_info(node_with_score.node)
            
            if not current_page:
                # No page info, treat as individual group
                groups.append([node_with_score])
                processed_node_ids.add(node_with_score.node.node_id)
                continue
            
            # Find related nodes (same document, adjacent pages)
            related_nodes = [node_with_score]
            processed_node_ids.add(node_with_score.node.node_id)
            
            # Look for nodes from adjacent pages
            for other_node_with_score in nodes:
                if other_node_with_score.node.node_id in processed_node_ids:
                    continue
                    
                other_page = self._extract_page_info(other_node_with_score.node)
                if other_page and self._are_pages_adjacent(current_page, other_page):
                    related_nodes.append(other_node_with_score)
                    processed_node_ids.add(other_node_with_score.node.node_id)
            
            groups.append(related_nodes)
            
        return groups

    def _extract_page_info(self, node) -> Optional[str]:
        """Extract page information from node metadata."""
        # Try different metadata keys for page information
        page_info = (
            node.metadata.get("page_info") or
            node.metadata.get("page_ranges") or
            node.metadata.get("page_label") or
            node.metadata.get("page")
        )
        return str(page_info) if page_info else None

    def _are_pages_adjacent(self, page1: str, page2: str) -> bool:
        """Check if two pages are adjacent or part of the same range."""
        try:
            # Handle simple page numbers
            if page1.isdigit() and page2.isdigit():
                return abs(int(page1) - int(page2)) <= 1
                
            # Handle page ranges (e.g., "5-7")
            if "-" in page1 and "-" in page2:
                # Extract first page from each range
                first_page1 = int(page1.split("-")[0])
                first_page2 = int(page2.split("-")[0])
                return abs(first_page1 - first_page2) <= 2
                
            # Handle mixed cases
            first_page1 = int(page1.split("-")[0]) if "-" in page1 else int(page1)
            first_page2 = int(page2.split("-")[0]) if "-" in page2 else int(page2)
            return abs(first_page1 - first_page2) <= 2
            
        except (ValueError, IndexError):
            # If we can't parse page numbers, assume they might be related
            return page1 == page2

    def __setattr__(self, name: str, value: Any) -> None:
        """Forward attribute setting to base retriever when appropriate."""
        if hasattr(self, 'base_retriever') and hasattr(self.base_retriever, name):
            setattr(self.base_retriever, name, value)
        super().__setattr__(name, value)