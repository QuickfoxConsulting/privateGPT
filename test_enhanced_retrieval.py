#!/usr/bin/env python3
"""
Test script for the enhanced retrieval system.
"""

from private_gpt.components.vector_store.enhanced_retriever import EnhancedFileRetriever
from llama_index.core.schema import TextNode, NodeWithScore

def test_enhanced_retriever():
    """Test the enhanced file retriever with sample data."""
    # Create sample nodes with page information
    node1 = TextNode(
        text="This is content from page 1 of the document.",
        metadata={
            "filename": "test_document.pdf",
            "file_name": "test_document.pdf",
            "page": 1,
            "page_info": "Page 1"
        }
    )
    
    node2 = TextNode(
        text="This is related content from page 2 of the document.",
        metadata={
            "filename": "test_document.pdf",
            "file_name": "test_document.pdf",
            "page": 2,
            "page_info": "Page 2"
        }
    )
    
    node3 = TextNode(
        text="This is content from a different document.",
        metadata={
            "filename": "other_document.pdf",
            "file_name": "other_document.pdf",
            "page": 1,
            "page_info": "Page 1"
        }
    )
    
    # Create node with score objects
    nodes_with_score = [
        NodeWithScore(node=node1, score=0.95),
        NodeWithScore(node=node2, score=0.85),
        NodeWithScore(node=node3, score=0.90),
    ]
    
    # Test the enhanced retriever
    print("=== Testing Enhanced File Retriever ===")
    
    # Test with cross-page retrieval enabled
    retriever = EnhancedFileRetriever(
        index=None,  # We're not actually using the index in this test
        file_name="test_document.pdf",
        similarity_top_k=5,
        enable_cross_page_retrieval=True,
        preserve_page_context=True
    )
    
    # Simulate the enhancement process
    enhanced_nodes = retriever._enhance_with_cross_page_context(nodes_with_score[:2])
    
    print(f"Enhanced {len(nodes_with_score[:2])} nodes to {len(enhanced_nodes)} nodes")
    for i, node_with_score in enumerate(enhanced_nodes):
        print(f"Node {i+1}:")
        print(f"  Text: {node_with_score.node.text}")
        print(f"  Score: {node_with_score.score}")
        print(f"  Metadata: {node_with_score.node.metadata}")
        print()
    
    # Test page grouping
    page_groups = retriever._group_nodes_by_page_context(nodes_with_score[:2])
    print(f"Grouped nodes into {len(page_groups)} groups")
    for i, group in enumerate(page_groups):
        print(f"Group {i+1}: {len(group)} nodes")
    
    # Test page adjacency
    print("\n=== Testing Page Adjacency ===")
    print(f"Pages 1 and 2 adjacent: {retriever._are_pages_adjacent('1', '2')}")
    print(f"Pages 1 and 3 adjacent: {retriever._are_pages_adjacent('1', '3')}")
    print(f"Page ranges 5-7 and 8-10 adjacent: {retriever._are_pages_adjacent('5-7', '8-10')}")

if __name__ == "__main__":
    print("Testing enhanced retrieval system...")
    test_enhanced_retriever()
    print("\nTesting completed!")