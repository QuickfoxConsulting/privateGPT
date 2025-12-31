from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime, Text, Enum
from sqlalchemy.orm import relationship

from private_gpt.users.db.base_class import Base

class Prompt(Base):
    """Models a prompt table for dynamic prompt management."""
    __tablename__ = "prompts"
    
    id = Column(Integer, nullable=False, primary_key=True)
    name = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    description = Column(String(255), nullable=True)
    
    # Types: "system", "qa", "condense", "decompose", "context"
    type = Column(String(50), nullable=False)
    
    # Modes: "chat", "search", "agentic"
    mode = Column(String(50), nullable=False)
    
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", backref="prompts")
    
    version = Column(Integer, default=1)
    
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(
        DateTime,
        default=datetime.now,
        onupdate=datetime.now,
    )

    def __repr__(self):
        return f"<Prompt {self.name} (v{self.version})>"
