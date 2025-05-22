from typing import Dict, List, Optional, ClassVar
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.storage.docstore import BaseDocumentStore
from pydantic import Field

class PrevNextPagePostprocessor(BaseNodePostprocessor):
    """
    Previous/Next Page post-processor for PDF documents.
    
    This postprocessor retrieves adjacent page chunks based on 'prev_chunk_id' and 'next_chunk_id'
    metadata fields that were added during document processing. This improves retrieval context
    by providing surrounding document pages.
    
    Args:
        docstore (BaseDocumentStore): The document store containing all document chunks.
        prev_pages (int): Number of previous pages to include (default: 1).
        next_pages (int): Number of next pages to include (default: 1).
        mode (str): The mode of the post-processor - "previous", "next", or "both".
    """
    docstore: BaseDocumentStore
    prev_pages: int = Field(default=1, description="Number of previous pages to include")
    next_pages: int = Field(default=1, description="Number of next pages to include")
    mode: str = Field(default="both", description="Mode: 'previous', 'next', or 'both'")
    
    
    def class_name(self) -> str:
        return "PrevNextPagePostprocessor"
    
    def _get_next_pages(self, node: NodeWithScore, num_pages: int) -> Dict[str, NodeWithScore]:
        """Get the specified number of next page chunks."""
        result = {}
        current_node = node
        
        for _ in range(num_pages):
            # Check if there's a next page
            if (not hasattr(current_node.node, 'metadata') or 
                'next_chunk_id' not in current_node.node.metadata or 
                current_node.node.metadata['next_chunk_id'] is None):
                break
                
            next_id = current_node.node.metadata['next_chunk_id']
            
            # Try to get the next node from the document store
            try:
                next_node = self.docstore.get_node(next_id)
                # Create a NodeWithScore with the same score as the original node
                next_node_with_score = NodeWithScore(node=next_node, score=current_node.score)
                result[next_id] = next_node_with_score
                current_node = next_node_with_score
            except (KeyError, ValueError):
                # Node not found, stop the chain
                break
                
        return result
    
    def _get_previous_pages(self, node: NodeWithScore, num_pages: int) -> Dict[str, NodeWithScore]:
        """Get the specified number of previous page chunks."""
        result = {}
        current_node = node
        
        for _ in range(num_pages):
            # Check if there's a previous page
            if (not hasattr(current_node.node, 'metadata') or 
                'prev_chunk_id' not in current_node.node.metadata or 
                current_node.node.metadata['prev_chunk_id'] is None):
                break
                
            prev_id = current_node.node.metadata['prev_chunk_id']
            
            # Try to get the previous node from the document store
            try:
                prev_node = self.docstore.get_node(prev_id)
                # Create a NodeWithScore with the same score as the original node
                prev_node_with_score = NodeWithScore(node=prev_node, score=current_node.score)
                result[prev_id] = prev_node_with_score
                current_node = prev_node_with_score
            except (KeyError, ValueError):
                # Node not found, stop the chain
                break
                
        return result
    
    def _sort_nodes_by_page_order(self, nodes_dict: Dict[str, NodeWithScore]) -> List[NodeWithScore]:
        """Sort nodes based on their chunk_index to maintain document order."""
        nodes_list = list(nodes_dict.values())
        
        def get_chunk_index(node):
            if hasattr(node.node, 'metadata') and 'chunk_index' in node.node.metadata:
                return node.node.metadata['chunk_index']
            return float('inf')  # Place nodes without chunk_index at the end
            
        return sorted(nodes_list, key=get_chunk_index)
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes by adding previous and next page chunks."""
        # Dictionary to store all nodes, using node_id as key to avoid duplicates
        all_nodes: Dict[str, NodeWithScore] = {node.node.node_id: node for node in nodes}
        
        # Process each node to add previous/next pages based on mode
        for node in nodes:
            if self.mode in ["next", "both"]:
                next_pages = self._get_next_pages(node, self.next_pages)
                all_nodes.update(next_pages)
                
            if self.mode in ["previous", "both"]:
                prev_pages = self._get_previous_pages(node, self.prev_pages)
                all_nodes.update(prev_pages)
        
        sorted_nodes = self._sort_nodes_by_page_order(all_nodes)
        return sorted_nodes


class DocumentAwarePrevNextPostprocessor(PrevNextPagePostprocessor):
    """
    Enhanced version of the PrevNextPagePostprocessor that is aware of document boundaries.
    
    This postprocessor only retrieves adjacent pages from the same document, ensuring
    that we don't accidentally pull pages from different documents.
    """
    
    def _get_next_pages(self, node: NodeWithScore, num_pages: int) -> Dict[str, NodeWithScore]:
        """Get next page chunks, ensuring they're from the same document."""
        result = {}
        current_node = node
        
        # Get the document_id from the current node
        if not hasattr(current_node.node, 'metadata') or 'document_id' not in current_node.node.metadata:
            return result
            
        doc_id = current_node.node.metadata['document_id']
        
        for _ in range(num_pages):
            # Check if there's a next page
            if (not hasattr(current_node.node, 'metadata') or 
                'next_chunk_id' not in current_node.node.metadata or 
                current_node.node.metadata['next_chunk_id'] is None):
                break
                
            next_id = current_node.node.metadata['next_chunk_id']
            
            # Try to get the next node
            try:
                next_node = self.docstore.get_node(next_id)
                
                # Check if the next node is from the same document
                if not hasattr(next_node, 'metadata') or next_node.metadata.get('document_id') != doc_id:
                    break
                    
                # Create a NodeWithScore with a slightly lower score (0.95 of original)
                next_node_with_score = NodeWithScore(
                    node=next_node, 
                    score=current_node.score * 0.95
                )
                result[next_id] = next_node_with_score
                current_node = next_node_with_score
                
            except (KeyError, ValueError):
                break
                
        return result
    
    def _get_previous_pages(self, node: NodeWithScore, num_pages: int) -> Dict[str, NodeWithScore]:
        """Get previous page chunks, ensuring they're from the same document."""
        result = {}
        current_node = node
        
        # Get the document_id from the current node
        if not hasattr(current_node.node, 'metadata') or 'document_id' not in current_node.node.metadata:
            return result
            
        doc_id = current_node.node.metadata['document_id']
        
        for _ in range(num_pages):
            # Check if there's a previous page
            if (not hasattr(current_node.node, 'metadata') or 
                'prev_chunk_id' not in current_node.node.metadata or 
                current_node.node.metadata['prev_chunk_id'] is None):
                break
                
            prev_id = current_node.node.metadata['prev_chunk_id']
            
            # Try to get the previous node
            try:
                prev_node = self.docstore.get_node(prev_id)
                
                # Check if the previous node is from the same document
                if not hasattr(prev_node, 'metadata') or prev_node.metadata.get('document_id') != doc_id:
                    break
                    
                # Create a NodeWithScore with a slightly lower score (0.95 of original)
                prev_node_with_score = NodeWithScore(
                    node=prev_node, 
                    score=current_node.score * 0.95
                )
                result[prev_id] = prev_node_with_score
                current_node = prev_node_with_score
                
            except (KeyError, ValueError):
                break
                
        return result

# Example usage
"""
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import Document

# First, create your document index
pdf_reader = CustomMarkerPDFReader()
documents = pdf_reader.load_data("your_pdf_file.pdf")
storage_context = StorageContext.from_defaults()
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Then, when querying, use the post-processor
retriever = index.as_retriever()
postprocessor = PrevNextPagePostprocessor(
    docstore=storage_context.docstore,
    prev_pages=1,
    next_pages=1,
    mode="both"
)

# Apply the postprocessor to the retriever
retriever.node_postprocessors = [postprocessor]

# Now query with improved context
query_engine = index.as_query_engine(
    retriever=retriever,
    response_mode="tree_summarize"  # Use tree_summarize to handle multiple nodes effectively
)
response = query_engine.query("your question here")
"""
