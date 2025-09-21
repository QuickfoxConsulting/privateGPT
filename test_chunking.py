#!/usr/bin/env python3

import asyncio
import tempfile
from pathlib import Path
from private_gpt.server.ingest.ingest_service import IngestService, ChunkingStrategy
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import VectorStoreComponent
from private_gpt.settings.settings import settings

async def test_dynamic_chunking():
    """Test dynamic chunking strategies."""
    print("Testing dynamic chunking strategies...")
    
    # Initialize components (this would normally be done by the DI system)
    llm_component = LLMComponent()
    vector_store_component = VectorStoreComponent()
    embedding_component = EmbeddingComponent()
    node_store_component = NodeStoreComponent()
    
    # Create ingest service
    ingest_service = IngestService(
        llm_component,
        vector_store_component,
        embedding_component,
        node_store_component
    )
    
    # Create a simple test file
    test_content = """
    This is a test document with multiple sentences. 
    It is designed to test the dynamic chunking functionality.
    The document has several paragraphs to ensure proper chunking.
    We want to verify that different chunking strategies work correctly.
    This includes late chunking, sentence window, semantic, and page by page strategies.
    Each strategy should produce different results based on its parameters.
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = Path(f.name)
    
    try:
        # Test Late Chunking
        print("Testing Late Chunking...")
        result = await ingest_service.ingest_file(
            "test_late_chunking.txt",
            temp_file_path,
            chunk_size=100,
            strategy=ChunkingStrategy.LATE_CHUNKING
        )
        print(f"Late Chunking produced {len(result)} documents")
        
        # Test Sentence Window
        print("Testing Sentence Window...")
        result = await ingest_service.ingest_file(
            "test_sentence_window.txt",
            temp_file_path,
            chunk_size=50,
            window_size=2,
            strategy=ChunkingStrategy.SENTENCE_WINDOW
        )
        print(f"Sentence Window produced {len(result)} documents")
        
        # Test Semantic
        print("Testing Semantic...")
        result = await ingest_service.ingest_file(
            "test_semantic.txt",
            temp_file_path,
            strategy=ChunkingStrategy.SEMANTIC
        )
        print(f"Semantic produced {len(result)} documents")
        
        # Test Page By Page
        print("Testing Page By Page...")
        result = await ingest_service.ingest_file(
            "test_page_by_page.txt",
            temp_file_path,
            strategy=ChunkingStrategy.PAGE_BY_PAGE
        )
        print(f"Page By Page produced {len(result)} documents")
        
        print("All tests completed successfully!")
        
    finally:
        # Clean up temp file
        temp_file_path.unlink()

if __name__ == "__main__":
    asyncio.run(test_dynamic_chunking())