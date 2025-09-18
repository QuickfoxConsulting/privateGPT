from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from private_gpt.users.db.base_class import Base


class FAQ(Base):
    """Models a FAQ table"""
    __tablename__ = "faqs"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(JSONB, nullable=False)
    category = Column(String(100), nullable=True, index=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    frequency = Column(Integer, default=0, nullable=False)  # Track how often this FAQ is accessed
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationship to track which admin created/updated the FAQ
    created_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    updated_by_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    
    created_by = relationship("User", foreign_keys=[created_by_id])
    updated_by = relationship("User", foreign_keys=[updated_by_id])
    
    def __repr__(self):
        return f"<FAQ {self.id}: {self.question[:50]}...>"