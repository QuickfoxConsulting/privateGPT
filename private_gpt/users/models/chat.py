# from __future__ import annotations
# import uuid
# from typing import List, Dict, Any, Optional, Union, Literal
# from datetime import datetime
# from pydantic import BaseModel, Field

# from sqlalchemy.orm import relationship, Session
# from sqlalchemy.dialects.postgresql import UUID, JSONB
# from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Boolean, event, Index, JSON, Enum as SQLAlchemyEnum, UniqueConstraint, func
# from sqlalchemy.exc import SQLAlchemyError
# from private_gpt.users.db.base_class import Base
# from enum import Enum as PythonEnum
# import logging

# logger = logging.getLogger(__name__)

# class ChatContentSchema(BaseModel):
#     text: str = Field(..., description="The text content of the message")
    
#     class Config:
#         extra = "allow" 

# class Rating(PythonEnum):
#     """User feedback ratings for chat messages"""
#     THUMBS_UP = 'THUMBS_UP'
#     THUMBS_DOWN = 'THUMBS_DOWN'
#     HELPFUL = 'HELPFUL'
#     NOT_HELPFUL = 'NOT_HELPFUL'
#     IRRELEVANT = 'IRRELEVANT'

# class MessageStatus(PythonEnum):
#     """Status indicators for messages"""
#     DELIVERED = 'DELIVERED'
#     READ = 'READ'
#     DELETED = 'DELETED'
#     EDITED = 'EDITED'


# class ChatHistory(Base):
#     """Models a conversation history between a user and the system"""
#     __tablename__ = "chat_history"
    
#     conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     title = Column(Text, nullable=True, index=True)
#     created_at = Column(DateTime, default=datetime.utcnow, index=True)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     deleted_at = Column(DateTime, nullable=True, index=True)  
#     user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
#     is_archived = Column(Boolean, default=False, index=True)
#     is_deleted = Column(Boolean, default=False, index=True)
#     chat_summary = Column(Text, nullable=True)

#     # Define relationships
#     user = relationship("User", back_populates="chat_histories")
#     chat_items = relationship(
#         "ChatItem", 
#         back_populates="chat_history", 
#         cascade="save-update, merge, delete, delete-orphan",
#         order_by="ChatItem.created_at.desc()",
#         lazy="dynamic",
#     )
    
#     title_generated = Column(Boolean, default=False)
    
#     __table_args__ = (
#         Index('idx_chat_history_user_created', 'user_id', 'created_at'),
#         Index('idx_chat_history_active', 'is_deleted', 'is_archived'),
#     )
        
#     def generate_title(self) -> None:
#         """Sets title based on the first user message"""
#         if self.title_generated:
#             return
            
#         user_chat_items = [item for item in self.chat_items if item.sender == "user"]
        
#         if user_chat_items:
#             try:
#                 first_message_content = user_chat_items[0].content
#                 if isinstance(first_message_content, dict) and "text" in first_message_content:
#                     text = first_message_content["text"]
#                     self.title = text[:50] + "..." if len(text) > 50 else text
#                 elif isinstance(first_message_content, str):
#                     self.title = first_message_content[:50] + "..." if len(first_message_content) > 50 else first_message_content
#                 else:
#                     self.title = "New Chat"
#             except (KeyError, TypeError, IndexError) as e:
#                 logger.warning(f"Error generating title: {str(e)}")
#                 self.title = "New Chat"
#         else:
#             self.title = "New Chat"
            
#         self.title_generated = True
    
#     def soft_delete(self) -> None:
#         """Mark the chat history as deleted without removing from database"""
#         self.is_deleted = True
#         self.deleted_at = datetime.utcnow()
#         self.updated_at = datetime.utcnow()
    
#     def archive(self) -> None:
#         """Archive the chat history"""
#         self.is_archived = True
#         self.updated_at = datetime.utcnow()
    
#     def unarchive(self) -> None:
#         """Unarchive the chat history"""
#         self.is_archived = False
#         self.updated_at = datetime.utcnow()
    
#     def get_message_count(self, db: Session) -> Dict[str, int]:
#         """Get counts of messages by sender using SQL aggregate"""
#         try:
#             counts = dict(db.query(
#                 ChatItem.sender, func.count(ChatItem.id)
#             ).filter(
#                 ChatItem.conversation_id == self.conversation_id
#             ).group_by(ChatItem.sender).all())
#             return counts
#         except SQLAlchemyError as e:
#             logger.error(f"Error getting message count for conversation {self.conversation_id}: {str(e)}")
#             raise
    
#     def get_duration(self, db: Session) -> Optional[float]:
#         """Get conversation duration in seconds using SQL min/max"""
#         try:
#             result = db.query(
#                 func.min(ChatItem.created_at).label('first'),
#                 func.max(ChatItem.created_at).label('last')
#             ).filter(
#                 ChatItem.conversation_id == self.conversation_id
#             ).one_or_none()
            
#             if result and result.first and result.last:
#                 return (result.last - result.first).total_seconds()
#             return None
#         except SQLAlchemyError as e:
#             logger.error(f"Error getting duration for conversation {self.conversation_id}: {str(e)}")
#             raise
    
#     def __repr__(self) -> str:
#         """Returns string representation of model instance"""
#         return f"<ChatHistory conversation_id={self.conversation_id} title={self.title!r} messages={len(self.chat_items)}>"


# class ChatItem(Base):
#     """Models an individual message in a chat conversation"""
#     __tablename__ = "chat_items"
    
#     id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)    
#     sender = Column(String(225), nullable=False, index=True)
#     content = Column(JSONB, nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow, index=True) 
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     rating = Column(SQLAlchemyEnum(Rating), nullable=True)
#     status = Column(SQLAlchemyEnum(MessageStatus), default=MessageStatus.DELIVERED)
#     parent_id = Column(UUID(as_uuid=True), ForeignKey("chat_items.id"), nullable=True)
#     conversation_id = Column(
#         UUID(as_uuid=True), 
#         ForeignKey("chat_history.conversation_id", ondelete="CASCADE"), 
#         nullable=False,
#         index=True
#     )
#     edit_history = relationship("ChatItemVersion", back_populates="chat_item", cascade="all, delete-orphan")
#     chat_history = relationship("ChatHistory", back_populates="chat_items")

#     __table_args__ = (
#         Index('idx_chat_items_sender', 'conversation_id', 'sender'),
#         Index('idx_chat_items_created', 'conversation_id', 'created_at'),
#         Index('gin_chat_items_content', 'content', postgresql_ops={'content': 'jsonb_path_ops'}, postgresql_using='gin'),  # New: GIN for JSONB
#         CheckConstraint("sender IN ('user', 'assistant', 'system')", name='check_sender_type'),  # New: Enforce sender values
#     )
    
#     def validate_content(self) -> bool:
#         """Validate the content format"""
#         try:
#             if isinstance(self.content, dict):
#                 ChatContentSchema(**self.content)
#                 return True
#             elif isinstance(self.content, str):
#                 return True
#             return False
#         except Exception as e:
#             logger.warning(f"Content validation failed for ChatItem {self.id}: {str(e)}")
#             return False
    
#     def set_rating(self, rating: Rating) -> None:
#         """Set user feedback rating"""
#         self.rating = rating
#         self.updated_at = datetime.utcnow()
    
#     def edit_content(self, new_content: Union[Dict, str], db: Session) -> None:
#         """Edit message content, save version history, and update status"""
#         try:
#             # Validate new content
#             if isinstance(new_content, dict):
#                 ChatContentSchema(**new_content)
            
#             # Save old version
#             version = ChatItemVersion(
#                 chat_item_id=self.id,
#                 old_content=self.content,
#                 edited_at=datetime.utcnow()
#             )
#             db.add(version)
            
#             # Update current
#             self.content = new_content
#             self.status = MessageStatus.EDITED
#             self.updated_at = datetime.utcnow()
            
#             db.commit()
#         except SQLAlchemyError as e:
#             logger.error(f"Error editing content for ChatItem {self.id}: {str(e)}")
#             db.rollback()
#             raise
    
#     def get_thread(self, db: Session) -> List["ChatItem"]:
#         """Get threaded replies starting from this item"""
#         try:
#             return db.query(ChatItem).filter(
#                 ChatItem.parent_id == self.id
#             ).order_by(ChatItem.created_at.asc()).all()
#         except SQLAlchemyError as e:
#             logger.error(f"Error getting thread for ChatItem {self.id}: {str(e)}")
#             raise
    
#     def __repr__(self) -> str:
#         """Returns string representation of model instance"""
#         return f"<ChatItem id={self.id} sender={self.sender} status={self.status.name}>"


# @event.listens_for(ChatItem, "before_delete")
# def log_chat_item_deletion(mapper, connection, target):
#     logger.warning(f"Deleting ChatItem: {target}")


# def get_recent_chats(
#     db: Session, 
#     user_id: int, 
#     limit: int = 10, 
#     offset: int = 0  # New: Pagination
# ) -> List[ChatHistory]:
#     """Get the most recent active chat histories for a user with pagination"""
#     try:
#         return db.query(ChatHistory).options(
#             selectinload(ChatHistory.chat_items)  # Eager load if needed
#         ).filter(
#             ChatHistory.user_id == user_id,
#             ChatHistory.is_deleted == False
#         ).order_by(
#             ChatHistory.updated_at.desc()
#         ).limit(limit).offset(offset).all()
#     except SQLAlchemyError as e:
#         logger.error(f"Error getting recent chats for user {user_id}: {str(e)}")
#         raise


# def search_chats(
#     db: Session, 
#     user_id: int, 
#     query: str, 
#     limit: int = 10,
#     offset: int = 0, 
#     min_date: Optional[datetime] = None,  
#     rating: Optional[Rating] = None  
# ) -> List[ChatHistory]:
#     """
#     Search chat histories by title or content for a user with enhanced filters
#     """
#     search_query = f"%{query}%"
    
#     base_query = db.query(ChatHistory).filter(
#         ChatHistory.user_id == user_id,
#         ChatHistory.is_deleted == False
#     )
    
#     if min_date:
#         base_query = base_query.filter(ChatHistory.created_at >= min_date)
    
#     # Title matches
#     title_matches = base_query.filter(
#         ChatHistory.title.ilike(search_query)
#     )
    
#     # Content matches with JSONB operator
#     content_matches = base_query.join(ChatItem).filter(
#         cast(ChatItem.content.op('->>')('text'), SQLString).ilike(search_query)
#     )
    
#     if rating:
#         content_matches = content_matches.filter(ChatItem.rating == rating)
    
#     combined = title_matches.union(content_matches.distinct()).order_by(
#         ChatHistory.updated_at.desc()
#     ).limit(limit).offset(offset)
    
#     return combined.all()


from __future__ import annotations
import uuid
from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field

from sqlalchemy.orm import relationship, Session, selectinload
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Text, Boolean, event, Index, JSON, Enum as SQLAlchemyEnum, 
    UniqueConstraint, func, CheckConstraint
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.expression import cast
from sqlalchemy.types import String as SQLString
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
    deleted_at = Column(DateTime, nullable=True)  # New: For paranoid soft deletes
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    is_archived = Column(Boolean, default=False, index=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    
    # Define relationships
    user = relationship("User", back_populates="chat_histories")
    chat_items = relationship(
        "ChatItem", 
        back_populates="chat_history", 
        cascade="save-update, merge, delete, delete-orphan",
        order_by="ChatItem.created_at.desc()",
        lazy="dynamic",
    )
    
    title_generated = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_chat_history_user_created', 'user_id', 'created_at'),
        Index('idx_chat_history_active', 'is_deleted', 'is_archived'),
    )
    
    def generate_title(self, db: Session) -> None:
        """Sets title based on the first user message, querying efficiently"""
        if self.title_generated:
            return
            
        try:
            first_user_item = db.query(ChatItem).filter(
                ChatItem.conversation_id == self.conversation_id,
                ChatItem.sender == "user"
            ).order_by(ChatItem.created_at.asc()).first()
            
            if first_user_item:
                content = first_user_item.content
                if isinstance(content, dict) and "text" in content:
                    text = content["text"]
                elif isinstance(content, str):
                    text = content
                else:
                    text = None
                
                if text:
                    self.title = text[:50] + "..." if len(text) > 50 else text
                else:
                    self.title = "New Chat"
            else:
                self.title = "New Chat"
            
            self.title_generated = True
            db.commit()
        except SQLAlchemyError as e:
            logger.error(f"Error generating title for conversation {self.conversation_id}: {str(e)}")
            db.rollback()
            raise
    
    def soft_delete(self) -> None:
        """Mark the chat history as deleted without removing from database"""
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def archive(self) -> None:
        """Archive the chat history"""
        self.is_archived = True
        self.updated_at = datetime.utcnow()
    
    def unarchive(self) -> None:
        """Unarchive the chat history"""
        self.is_archived = False
        self.updated_at = datetime.utcnow()
    
    def get_message_count(self, db: Session) -> Dict[str, int]:
        """Get counts of messages by sender using SQL aggregate"""
        try:
            counts = dict(db.query(
                ChatItem.sender, func.count(ChatItem.id)
            ).filter(
                ChatItem.conversation_id == self.conversation_id
            ).group_by(ChatItem.sender).all())
            return counts
        except SQLAlchemyError as e:
            logger.error(f"Error getting message count for conversation {self.conversation_id}: {str(e)}")
            raise
    
    def get_duration(self, db: Session) -> Optional[float]:
        """Get conversation duration in seconds using SQL min/max"""
        try:
            result = db.query(
                func.min(ChatItem.created_at).label('first'),
                func.max(ChatItem.created_at).label('last')
            ).filter(
                ChatItem.conversation_id == self.conversation_id
            ).one_or_none()
            
            if result and result.first and result.last:
                return (result.last - result.first).total_seconds()
            return None
        except SQLAlchemyError as e:
            logger.error(f"Error getting duration for conversation {self.conversation_id}: {str(e)}")
            raise
    
    def __repr__(self) -> str:
        """Returns string representation of model instance"""
        return f"<ChatHistory conversation_id={self.conversation_id} title={self.title!r} messages={len(self.chat_items)}>"


class ChatItem(Base):
    """Models an individual message in a chat conversation"""
    __tablename__ = "chat_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)    
    sender = Column(String(225), nullable=False, index=True)
    content = Column(JSONB, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
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
    
    # New: For edit history
    edit_history = relationship("ChatItemVersion", back_populates="chat_item", cascade="all, delete-orphan")
    
    chat_history = relationship("ChatHistory", back_populates="chat_items")
    
    __table_args__ = (
        Index('idx_chat_items_sender', 'conversation_id', 'sender'),
        Index('idx_chat_items_created', 'conversation_id', 'created_at'),
        Index('gin_chat_items_content', 'content', postgresql_ops={'content': 'jsonb_path_ops'}, postgresql_using='gin'),  # New: GIN for JSONB
        CheckConstraint("sender IN ('user', 'assistant', 'system')", name='check_sender_type'),  # New: Enforce sender values
    )
    
    def validate_content(self) -> bool:
        """Validate the content format"""
        try:
            # Allow None content
            if self.content is None:
                return True
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
    
    def edit_content(self, new_content: Union[Dict, str], db: Session) -> None:
        """Edit message content, save version history, and update status"""
        try:
            # Validate new content
            if isinstance(new_content, dict):
                ChatContentSchema(**new_content)
            
            # Save old version
            version = ChatItemVersion(
                chat_item_id=self.id,
                old_content=self.content,
                edited_at=datetime.utcnow()
            )
            db.add(version)
            
            # Update current
            self.content = new_content
            self.status = MessageStatus.EDITED
            self.updated_at = datetime.utcnow()
            
            db.commit()
        except SQLAlchemyError as e:
            logger.error(f"Error editing content for ChatItem {self.id}: {str(e)}")
            db.rollback()
            raise
    
    def get_thread(self, db: Session) -> List["ChatItem"]:
        """Get threaded replies starting from this item"""
        try:
            return db.query(ChatItem).filter(
                ChatItem.parent_id == self.id
            ).order_by(ChatItem.created_at.asc()).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting thread for ChatItem {self.id}: {str(e)}")
            raise
    
    def __repr__(self) -> str:
        """Returns string representation of model instance"""
        return f"<ChatItem id={self.id} sender={self.sender} status={self.status.name}>"


class ChatItemVersion(Base):
    """Stores edit history for chat items"""
    __tablename__ = "chat_item_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_item_id = Column(UUID(as_uuid=True), ForeignKey("chat_items.id", ondelete="CASCADE"), nullable=False)
    old_content = Column(JSONB, nullable=False)
    edited_at = Column(DateTime, default=datetime.utcnow)
    
    chat_item = relationship("ChatItem", back_populates="edit_history")
    
    __table_args__ = (
        Index('idx_chat_item_versions_item_id', 'chat_item_id'),
    )


@event.listens_for(ChatItem, "before_delete")
def log_chat_item_deletion(mapper, connection, target):
    logger.warning(f"Deleting ChatItem: {target}")


# @event.listens_for(ChatItem, "before_insert")
# def validate_before_insert(mapper, connection, target):
#     if not target.validate_content():
#         item_id = target.id if target.id is not None else 'new'
#         raise ValueError(f"Invalid content for ChatItem {item_id}")

def get_recent_chats(
    db: Session, 
    user_id: int, 
    limit: int = 10, 
    offset: int = 0 
) -> List[ChatHistory]:
    """Get the most recent active chat histories for a user with pagination"""
    try:
        return db.query(ChatHistory).options(
            selectinload(ChatHistory.chat_items)  # Eager load if needed
        ).filter(
            ChatHistory.user_id == user_id,
            ChatHistory.is_deleted == False
        ).order_by(
            ChatHistory.updated_at.desc()
        ).limit(limit).offset(offset).all()
    except SQLAlchemyError as e:
        logger.error(f"Error getting recent chats for user {user_id}: {str(e)}")
        raise


def search_chats(
    db: Session, 
    user_id: int, 
    query: str, 
    limit: int = 10,
    offset: int = 0, 
    min_date: Optional[datetime] = None,  # New: Date filter
    rating: Optional[Rating] = None  # New: Rating filter
) -> List[ChatHistory]:
    """
    Search chat histories by title or content for a user with enhanced filters
    """
    search_query = f"%{query}%"
    
    base_query = db.query(ChatHistory).filter(
        ChatHistory.user_id == user_id,
        ChatHistory.is_deleted == False
    )
    
    if min_date:
        base_query = base_query.filter(ChatHistory.created_at >= min_date)
    
    # Title matches
    title_matches = base_query.filter(
        ChatHistory.title.ilike(search_query)
    )
    
    # Content matches with JSONB operator
    content_matches = base_query.join(ChatItem).filter(
        cast(ChatItem.content.op('->>')('text'), SQLString).ilike(search_query)
    )
    
    if rating:
        content_matches = content_matches.filter(ChatItem.rating == rating)
    
    combined = title_matches.union(content_matches.distinct()).order_by(
        ChatHistory.updated_at.desc()
    ).limit(limit).offset(offset)
    
    return combined.all()