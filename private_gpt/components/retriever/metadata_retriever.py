from typing import List, Optional, Tuple
import re
from dataclasses import dataclass
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever

@dataclass
class MetadataFilters:
    file_names: List[str]
    tags: List[str]
    original_query: str

class MetadataFilterRetriever(BaseRetriever):
    """Retriever that filters nodes based on file names and tags from the query"""
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
    ) -> None:
        self.base_retriever = base_retriever
        super().__init__()

    def _parse_metadata_filters(self, query: str) -> MetadataFilters:
        """Extract file names and tags from query"""
        file_names = re.findall(r'@(\S+)', query)
        
        # Find all #tags
        tags = re.findall(r'#(\S+)', query)
        
        # Remove the metadata markers from the query
        clean_query = re.sub(r'[@#]\S+\s*', '', query).strip()
        
        return MetadataFilters(
            file_names=file_names,
            tags=tags,
            original_query=clean_query
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # Parse metadata filters from query
        filters = self._parse_metadata_filters(query_bundle.query_str)
        
        # Create new query bundle with clean query
        clean_bundle = QueryBundle(
            query_str=filters.original_query,
            embedding=query_bundle.embedding
        )
        
        # Get base results
        results = self.base_retriever.retrieve(clean_bundle)
        
        # Filter by metadata
        filtered_results = []
        for node_with_score in results:
            node = node_with_score.node
            metadata = node.metadata or {}
            
            # Check if node matches file name filter
            matches_file = (
                not filters.file_names or
                any(f.lower() in metadata.get('file_name', '').lower() 
                    for f in filters.file_names)
            )
            matches_file = (
                not filters.file_names or
                any(f.lower() in metadata.get('filename', '').lower() 
                    for f in filters.file_names)
            )
            
            # Check if node matches tag filter
            matches_tags = (
                not filters.tags or
                any(tag.lower() in metadata.get('tags', [])
                    for tag in filters.tags)
            )
            
            # Include node if it matches all filters
            if matches_file and matches_tags:
                filtered_results.append(node_with_score)
                
        return filtered_results
