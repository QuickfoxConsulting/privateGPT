"""Models for chunking parameters."""

from typing import Optional
from pydantic import BaseModel, Field
from private_gpt.server.ingest.ingest_service import ChunkingStrategy


class ChunkingParameters(BaseModel):
    """Parameters for controlling chunking behavior."""
    
    chunk_size: int = Field(
        512,
        description="The maximum size of each chunk in tokens",
        ge=100,
        le=2048
    )
    
    chunk_overlap: int = Field(
        100,
        description="The overlap size between consecutive chunks",
        ge=0,
        le=512
    )
    
    strategy: ChunkingStrategy = Field(
        ChunkingStrategy.LATE_CHUNKING,
        description="The chunking strategy to use"
    )