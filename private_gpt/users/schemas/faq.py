from datetime import datetime
from typing import Optional, List, Dict, Union
from pydantic import BaseModel, Field, Json


class FAQBase(BaseModel):
    """Base FAQ schema with common fields."""
    question: str = Field(..., description="The FAQ question", min_length=1, max_length=1000)
    answer: Union[str, Dict] = Field(..., description="The FAQ answer in JSON format")
    category: Optional[str] = Field(None, description="Category for the FAQ", max_length=100)

class FAQCreate(FAQBase):
    """Schema for creating a new FAQ."""
    pass


class FAQUpdate(FAQBase):
    """Schema for updating an existing FAQ."""
    question: Optional[str] = Field(None, description="The FAQ question", min_length=1, max_length=1000)
    answer: Optional[str] = Field(None, description="The FAQ answer", min_length=1, max_length=5000)
    category: Optional[str] = Field(None, description="Category for the FAQ", max_length=100)
    is_active: Optional[bool] = Field(None, description="Whether the FAQ is active")


class FAQInDBBase(FAQBase):
    """Base schema for FAQ stored in database."""
    id: int
    is_active: bool
    frequency: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class FAQ(FAQInDBBase):
    """Schema for FAQ response."""
    # Add similarity attribute for debugging (not stored in DB)
    similarity: Optional[float] = Field(None, description="Similarity score for search results")


class FAQInDB(FAQInDBBase):
    """Schema for FAQ stored in database with additional fields."""
    created_by_id: Optional[int]
    updated_by_id: Optional[int]


class FAQList(BaseModel):
    """Schema for FAQ list response."""
    faqs: List[FAQ]
    total: int


class FAQSearch(BaseModel):
    """Schema for FAQ search requests."""
    query: str = Field(..., description="Search query", min_length=1, max_length=100)
    limit: int = Field(10, description="Maximum number of results to return", ge=1, le=100)
    similarity_threshold: float = Field(0.7, description="Minimum similarity score (0-1) for results", ge=0.0, le=1.0)


class FAQCategory(BaseModel):
    """Schema for FAQ category."""
    category: str
    count: int