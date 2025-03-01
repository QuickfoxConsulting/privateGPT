from __future__ import annotations
import uuid
from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, event, Index, JSON, Enum as SQLAlchemyEnum, UniqueConstraint, func
from sqlalchemy.exc import SQLAlchemyError
from private_gpt.users.db.base_class import Base
from enum import Enum as PythonEnum
import logging

logger = logging.getLogger(__name__)

class ChatContentSchema(BaseModel):
    text: str = Field(..., description="The text content of the message")
    
    class Config:
        extra = "allow" 

class Rating(PythonEnum):
    """User feedback ratings for chat messages"""
    THUMBS_UP = 'THUMBS_UP'
    THUMBS_DOWN = 'THUMBS_DOWN'
    HELPFUL = 'HELPFUL'
    NOT_HELPFUL = 'NOT_HELPFUL'
    IRRELEVANT = 'IRRELEVANT'

class MessageStatus(PythonEnum):
    """Status indicators for messages"""
    DELIVERED = 'DELIVERED'
    READ = 'READ'
    DELETED = 'DELETED'
    EDITED = 'EDITED'


class ChatHistory(Base):
    """Models a conversation history between a user and the system"""
    __tablename__ = "chat_history"
    
    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    is_archived = Column(Boolean, default=False, index=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    # Define relationships
    user = relationship("User", back_populates="chat_histories")
    chat_items = relationship(
        "ChatItem", 
        back_populates="chat_history", 
        cascade="all, delete-orphan",
        order_by="ChatItem.index",
        lazy="selectin"  # Optimize for common access pattern
    )
    
    title_generated = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_chat_history_user_created', 'user_id', 'created_at'),
        Index('idx_chat_history_active', 'is_deleted', 'is_archived'),
    )
        
    def generate_title(self) -> None:
        """Sets title based on the first user message"""
        if self.title_generated:
            return
            
        user_chat_items = [item for item in self.chat_items if item.sender == "user"]
        
        if user_chat_items:
            try:
                first_message_content = user_chat_items[0].content
                if isinstance(first_message_content, dict) and "text" in first_message_content:
                    text = first_message_content["text"]
                    self.title = text[:50] + "..." if len(text) > 50 else text
                elif isinstance(first_message_content, str):
                    self.title = first_message_content[:50] + "..." if len(first_message_content) > 50 else first_message_content
                else:
                    self.title = "New Chat"
            except (KeyError, TypeError, IndexError) as e:
                logger.warning(f"Error generating title: {str(e)}")
                self.title = "New Chat"
        else:
            self.title = "New Chat"
            
        self.title_generated = True
    
    def soft_delete(self) -> None:
        """Mark the chat history as deleted without removing from database"""
        self.is_deleted = True
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the chat history"""
        self.is_archived = True
        self.updated_at = datetime.utcnow()
    
    def unarchive(self) -> None:
        """Unarchive the chat history"""
        self.is_archived = False
        self.updated_at = datetime.utcnow()
    
    def get_message_count(self) -> Dict[str, int]:
        """Get counts of messages by sender"""
        counts = {}
        for item in self.chat_items:
            if item.sender in counts:
                counts[item.sender] += 1
            else:
                counts[item.sender] = 1
        return counts
    
    def get_duration(self) -> Optional[float]:
        """Get conversation duration in seconds"""
        if not self.chat_items:
            return None
        
        first_message = min(self.chat_items, key=lambda x: x.created_at)
        last_message = max(self.chat_items, key=lambda x: x.created_at)
        
        return (last_message.created_at - first_message.created_at).total_seconds()
    
    def __repr__(self) -> str:
        """Returns string representation of model instance"""
        return f"<ChatHistory conversation_id={self.conversation_id} title={self.title!r} messages={len(self.chat_items)}>"


class ChatItem(Base):
    """Models an individual message in a chat conversation"""
    __tablename__ = "chat_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)    
    index = Column(Integer, nullable=False)
    
    sender = Column(String(225), nullable=False, index=True)
    content = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    rating = Column(SQLAlchemyEnum(Rating), nullable=True)
    status = Column(SQLAlchemyEnum(MessageStatus), default=MessageStatus.DELIVERED)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("chat_items.id"), nullable=True)
    conversation_id = Column(
        UUID(as_uuid=True), 
        ForeignKey("chat_history.conversation_id", ondelete="CASCADE"), 
        nullable=False,
        index=True
    )
    
    chat_history = relationship("ChatHistory", back_populates="chat_items")
    __table_args__ = (
        Index('idx_chat_items_conv_index', 'conversation_id', 'index'),
        Index('idx_chat_items_sender', 'conversation_id', 'sender'),
        UniqueConstraint('conversation_id', 'index', name='uq_chat_items_conversation_index'),
    )
    
    def validate_content(self) -> bool:
        """Validate the content format"""
        try:
            if isinstance(self.content, dict):
                ChatContentSchema(**self.content)
                return True
            elif isinstance(self.content, str):
                return True
            return False
        except Exception as e:
            logger.warning(f"Content validation failed for ChatItem {self.id}: {str(e)}")
            return False
    
    def set_rating(self, rating: Rating) -> None:
        """Set user feedback rating"""
        self.rating = rating
        self.updated_at = datetime.utcnow()
    
    def edit_content(self, new_content: Union[Dict, str]) -> None:
        """Edit message content and update status"""
        # Validate new content
        if isinstance(new_content, dict):
            try:
                ChatContentSchema(**new_content)
            except Exception as e:
                logger.warning(f"New content validation failed: {str(e)}")
        
        # Update content and status
        self.content = new_content
        self.status = MessageStatus.EDITED
        self.updated_at = datetime.utcnow()
    
    def __repr__(self) -> str:
        """Returns string representation of model instance"""
        return f"<ChatItem id={self.id} sender={self.sender} index={self.index} status={self.status.name}>"


def get_next_index(db: Session, conversation_id: uuid.UUID) -> int:
    """Get the next index value for the given conversation_id."""
    try:        
        # Use MAX function to get the highest index directly
        result = db.query(func.max(ChatItem.index)).filter(
            ChatItem.conversation_id == conversation_id
        ).scalar()
        
        # Return 0 if there are no items yet, otherwise increment the highest index
        return 0 if result is None else result + 1
    except SQLAlchemyError as e:
        logger.error(f"Error getting next index for conversation {conversation_id}: {str(e)}")
        raise


@event.listens_for(ChatItem, "before_insert")
def set_chat_item_index(mapper, connection, target):
    """Set the index value before inserting a new ChatItem if not already set."""
    # Only set the index if it hasn't been explicitly set already
    if target.conversation_id and target.index is None:
        session = Session.object_session(target)
        if session:
            try:
                # Check if there are any existing items with this conversation_id
                existing_count = session.query(ChatItem).filter(
                    ChatItem.conversation_id == target.conversation_id
                ).count()
                
                if existing_count == 0:
                    # If this is the first item, set index to 0
                    target.index = 0
                else:
                    # Get the highest index without relying on order
                    highest_index = session.query(func.max(ChatItem.index)).filter(
                        ChatItem.conversation_id == target.conversation_id
                    ).scalar() or -1
                    
                    # Set index to highest + 1 to avoid gaps
                    target.index = highest_index + 1
                
                logger.debug(f"Setting index via event listener: {target.index}")
            except SQLAlchemyError as e:
                logger.error(f"Error in event listener: {str(e)}")
                target.index = 0
        else:
            target.index = 0


@event.listens_for(ChatItem, "before_delete")
def log_chat_item_deletion(mapper, connection, target):
    logger.warning(f"Deleting ChatItem: {target}")


def get_recent_chats(db: Session, user_id: int, limit: int = 10) -> List[ChatHistory]:
    """Get the most recent active chat histories for a user"""
    return db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id,
        ChatHistory.is_deleted == False
    ).order_by(
        ChatHistory.updated_at.desc()
    ).limit(limit).all()


def search_chats(db: Session, user_id: int, query: str, limit: int = 10) -> List[ChatHistory]:
    """
    Search chat histories by title or content for a user
    """
    search_query = f"%{query}%"
    
    # First search by title
    title_matches = db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id,
        ChatHistory.is_deleted == False,
        ChatHistory.title.ilike(search_query)
    )
    
    # Then by content
    content_matches = db.query(ChatHistory).join(
        ChatItem, ChatHistory.conversation_id == ChatItem.conversation_id
    ).filter(
        ChatHistory.user_id == user_id,
        ChatHistory.is_deleted == False,
        ChatItem.content.cast(String).ilike(search_query)
    ).distinct()
    
    combined = title_matches.union(content_matches).order_by(
        ChatHistory.updated_at.desc()
    ).limit(limit)
    
    return combined.all()


# def batch_insert_chat_items(db: Session, items: List[ChatItem]) -> None:

#     items_by_conversation = {}
#     for item in items:
#         if item.conversation_id not in items_by_conversation:
#             items_by_conversation[item.conversation_id] = []
#         items_by_conversation[item.conversation_id].append(item)
    
#     # Get starting indexes for each conversation in a single query per conversation
#     for conversation_id, conv_items in items_by_conversation.items():
#         next_index = get_next_index(db, conversation_id)        
#         for i, item in enumerate(conv_items):
#             item.index = next_index + i
    
#     db.add_all(items)

# import uuid
# from datetime import datetime
# from sqlalchemy.orm import relationship, Session
# from sqlalchemy.dialects.postgresql import UUID 
# from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, event, JSON

# from private_gpt.users.db.base_class import Base

# class ChatHistory(Base):
#     """Models a chat history table"""

#     __tablename__ = "chat_history"

#     conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     title = Column(String(255), nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
#     user = relationship("User", back_populates="chat_histories")
#     chat_items = relationship(
#         "ChatItem", back_populates="chat_history", cascade="all, delete-orphan"
#     )
#     _title_generated = Column(Boolean, default=False)

#     def __init__(self, user_id, chat_items=None, **kwargs):
#         super().__init__(**kwargs)
#         self.user_id = user_id
#         self.chat_items = chat_items or []
#         for item in self.chat_items:
#             item.chat_history = self 

#         self.generate_title()

#     def generate_title(self) -> None:
#         """Sets title based on the first user message"""
#         if self.title_generated:
#             return
            
#         user_chat_items = [item for item in self.chat_items if item.sender == "user"]
        
#         if user_chat_items:
#             first_message_content = user_chat_items[0].content
#             if isinstance(first_message_content, dict) and "text" in first_message_content:
#                 text = first_message_content["text"]
#                 self.title = text[:50] + "..." if len(text) > 50 else text
#             else:
#                 self.title = "New Chat"
#         else:
#             self.title = "New Chat"
            
#         self.title_generated = True

#     def __repr__(self):
#         """Returns string representation of model instance"""
#         return f"<ChatHistory {self.conversation_id!r}>"


# class ChatItem(Base):
#     """Models a chat item table"""

#     __tablename__ = "chat_items"

#     id = Column(Integer, nullable=False, primary_key=True)
#     index = Column(Integer, nullable=False)
#     sender = Column(String(225), nullable=False)
#     content = Column(JSON, nullable=True)
#     created_at = Column(DateTime, default=datetime.now)
#     updated_at = Column(DateTime, default=datetime.now,
#                         onupdate=datetime.now)
#     like = Column(Boolean, default=None, nullable=True)
#     conversation_id = Column(UUID(as_uuid=True), ForeignKey(
#         "chat_history.conversation_id"), nullable=False)
#     chat_history = relationship("ChatHistory", back_populates="chat_items")

#     def __repr__(self):
#         """Returns string representation of model instance"""
#         return f"<ChatItem {self.id!r}>"

# def get_next_index(db: Session, conversation_id: uuid.UUID) -> int:
#     """Get the next index value for the given conversation_id."""
#     result = db.query(ChatItem).filter(
#         ChatItem.conversation_id == conversation_id
#     ).order_by(
#         ChatItem.index.desc()
#     ).first()
    
#     return 0 if result is None else result.index + 1


# @event.listens_for(ChatItem, "before_insert")
# def set_chat_item_index(mapper, connection, target):
#     """Set the index value before inserting a new ChatItem."""
#     if target.conversation_id and target.index is None:
#         session = Session.object_session(target)
#         if session:
#             target.index = get_next_index(session, target.conversation_id)
#         else:
#             target.index = 0


# @event.listens_for(ChatHistory, "after_insert")
# def update_chat_history_title(mapper, connection, target):
#     """Update title after insertion if not already generated"""
#     if not target.title_generated:
#         target.generate_title()
#         target.title_generated = True
