#!/usr/bin/env python3
"""
Test script for the new chunking strategies.
"""

from private_gpt.components.nodeparser.MarkdownAwareNodeParser import MarkdownAwareNodeParser
from private_gpt.components.nodeparser.AdaptiveNodeParser import AdaptiveNodeParser
from llama_index.core.schema import Document

def test_markdown_aware_parser():
    """Test the markdown aware parser with sample content."""
    # Sample markdown content similar to what LlamaParse might produce
    sample_text = """# Introduction
This is the introduction to our document.

## Chapter 1
This is the first chapter content. It contains important information that spans multiple sentences and should be kept together when possible.

### Section 1.1
This is a subsection with detailed information that is related to the previous content.

## Chapter 2
This is the second chapter which has its own distinct content.

### Section 2.1
More detailed information in section 2.1.

### Section 2.2
Additional content that continues from the previous section but is still related."""

    # Create a document with sample metadata
    doc = Document(
        text=sample_text,
        metadata={
            "filename": "sample.pdf",
            "page": 1,
            "document_id": "test_doc_001"
        }
    )
    
    # Test markdown aware parser
    parser = MarkdownAwareNodeParser.from_defaults(
        chunk_size=300,
        chunk_overlap=50,
        preserve_headers=True
    )
    
    nodes = parser.build_nodes_from_document(doc)
    
    print("=== Markdown Aware Parser Results ===")
    print(f"Generated {len(nodes)} nodes:")
    for i, node in enumerate(nodes):
        print(f"\nNode {i+1}:")
        print(f"Text: {node.text[:100]}...")
        print(f"Metadata: {node.metadata}")
    
    return nodes

def test_adaptive_parser():
    """Test the adaptive parser with different content types."""
    # Markdown-like content
    markdown_text = """# Title
## Section 1
Content with headers.

## Section 2
More content."""

    # Plain text content
    plain_text = """This is plain text content without any special formatting.
It should be parsed differently than markdown content.
Each sentence should be considered for chunking."""

    markdown_doc = Document(text=markdown_text, metadata={"filename": "markdown.pdf"})
    plain_doc = Document(text=plain_text, metadata={"filename": "plain.txt"})
    
    # Test adaptive parser
    parser = AdaptiveNodeParser.from_defaults(
        chunk_size=200,
        chunk_overlap=30
    )
    
    print("\n=== Adaptive Parser Results ===")
    
    # Test with markdown content
    print("\n--- Markdown Content ---")
    markdown_nodes = parser.build_nodes_from_document(markdown_doc)
    print(f"Generated {len(markdown_nodes)} nodes from markdown content")
    
    # Test with plain text content
    print("\n--- Plain Text Content ---")
    plain_nodes = parser.build_nodes_from_document(plain_doc)
    print(f"Generated {len(plain_nodes)} nodes from plain text content")

if __name__ == "__main__":
    print("Testing new chunking strategies...")
    test_markdown_aware_parser()
    test_adaptive_parser()
    print("\nTesting completed!")