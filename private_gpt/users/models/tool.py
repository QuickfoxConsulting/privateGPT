from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, JSON, UniqueConstraint
from sqlalchemy.orm import relationship
from datetime import datetime

from private_gpt.users.db.base_class import Base

class ToolCategory(Base):
    """Tool categories (Document, Web, Integration, Utility)"""
    __tablename__ = "tool_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)  # e.g., "integration", "document"
    description = Column(Text)
    icon = Column(String, nullable=True)
    
    tools = relationship("ToolDefinition", back_populates="category")

class ToolDefinition(Base):
    """Available tools that can be installed"""
    __tablename__ = "tool_definitions"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)  # e.g., "gmail_tool"
    display_name = Column(String)
    category_id = Column(Integer, ForeignKey("tool_categories.id"))
    version = Column(String)
    
    # Tool implementation
    module_path = Column(String)  # e.g., "private_gpt.server.tools.integrations.gmail"
    class_name = Column(String)   # e.g., "GmailTool"
    
    # Metadata
    description = Column(Text)
    author = Column(String, nullable=True)
    icon = Column(String, nullable=True)
    
    # Configuration schema (JSON)
    config_schema = Column(JSON, nullable=True)  # Required auth keys, API endpoints
    
    # Dependencies
    dependencies = Column(JSON, nullable=True)  # Required Python packages, other tools
    
    # Status
    is_official = Column(Boolean, default=False)
    is_verified = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    category = relationship("ToolCategory", back_populates="tools")
    installations = relationship("UserToolInstallation", back_populates="tool")
    execution_logs = relationship("ToolExecutionLog", back_populates="tool")

class UserToolInstallation(Base):
    """Track which tools are installed per user"""
    __tablename__ = "user_tool_installations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tool_id = Column(Integer, ForeignKey("tool_definitions.id"))
    
    # User-specific configuration
    config = Column(JSON, nullable=True)  # API keys, custom settings (should be encrypted in real impl)
    
    # Status
    is_enabled = Column(Boolean, default=True)
    installed_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0)
    
    user = relationship("User", backref="tool_installations")
    tool = relationship("ToolDefinition", back_populates="installations")
    
    __table_args__ = (UniqueConstraint('user_id', 'tool_id', name='uq_user_tool'),)

class ToolExecutionLog(Base):
    """Audit trail for tool usage"""
    __tablename__ = "tool_execution_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    tool_id = Column(Integer, ForeignKey("tool_definitions.id"))
    session_id = Column(String, nullable=True)
    
    input_data = Column(JSON, nullable=True)
    output_data = Column(JSON, nullable=True)
    status = Column(String)  # success, error, timeout
    error_message = Column(Text, nullable=True)
    
    execution_time_ms = Column(Integer, nullable=True)
    executed_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", backref="tool_execution_logs")
    tool = relationship("ToolDefinition", back_populates="execution_logs")
