# import uuid
# from typing import Optional, List
# from sqlalchemy.orm import Session
# from sqlalchemy.sql.expression import desc, asc

# from private_gpt.users.crud.base import CRUDBase
# from private_gpt.users.models.chat import ChatHistory, ChatItem
# from private_gpt.users.schemas.chat import ChatHistoryCreate, ChatHistoryCreate, ChatItemCreate, ChatItemUpdate


# class CRUDChat(CRUDBase[ChatHistory, ChatHistoryCreate, ChatHistoryCreate]):
#     def get_by_id(self, db: Session, *, id: uuid.UUID, skip: int=0, limit: int=10) -> Optional[ChatHistory]:
#         chat_history = (
#             db.query(self.model)
#             .filter(ChatHistory.conversation_id == id)
#             .order_by(asc(getattr(ChatHistory, 'created_at')))
#             .first()
#         )
#         if chat_history:
#             chat_history.chat_items = (
#                 db.query(ChatItem)
#                 .filter(ChatItem.conversation_id == id)
#                 .order_by(desc(getattr(ChatItem, 'index')))
#                 .offset(skip)
#                 .limit(limit)
#                 .all()
#             )
#         return chat_history
        
#     def get_conversation(self, db: Session, conversation_id: uuid.UUID) -> Optional[ChatHistory]:
#          return (
#                 db.query(self.model)
#                 .filter(ChatHistory.conversation_id == conversation_id)
#                 .first()
#             )
    
#     def get_chat_history(
#             self, db: Session, *,user_id:int
#         ) -> List[ChatHistory]:
#             return (
#                 db.query(self.model)
#                 .filter(ChatHistory.user_id == user_id)
#                 .order_by(desc(getattr(ChatHistory, 'created_at')))
#                 .all()
#             )
    
# class CRUDChatItem(CRUDBase[ChatItem, ChatItemCreate, ChatItemUpdate]):
#     pass


# chat = CRUDChat(ChatHistory)
# chat_item = CRUDChatItem(ChatItem)



import uuid
from typing import Optional, List, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import desc, asc
from sqlalchemy import or_, and_, func, text

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.chat import ChatHistory, ChatItem, Rating, MessageStatus
from private_gpt.users.schemas.chat import (
    ChatHistoryCreate, 
    ChatHistoryUpdate, 
    ChatItemCreate, 
    ChatItemUpdate,
    ChatSearch
)


class CRUDChat(CRUDBase[ChatHistory, ChatHistoryCreate, ChatHistoryUpdate]):
    # def get_by_id(
    #     self, 
    #     db: Session, 
    #     *, 
    #     id: uuid.UUID, 
    #     skip: int = 0, 
    #     limit: int = 10,
    #     include_deleted: bool = False
    # ) -> Optional[ChatHistory]:
    #     """
    #     Get a chat history by ID with pagination support for chat items
        
    #     Args:
    #         db: Database session
    #         id: Conversation UUID
    #         skip: Number of chat items to skip
    #         limit: Maximum number of chat items to return
    #         include_deleted: Whether to include soft-deleted chats
            
    #     Returns:
    #         ChatHistory object if found, None otherwise
    #     """
    #     query = db.query(self.model).filter(ChatHistory.conversation_id == id)
        
    #     if not include_deleted:
    #         query = query.filter(ChatHistory.is_deleted == False)
            
    #     chat_history = query.first()
        
    #     if chat_history:
    #         chat_history.chat_items = (
    #             db.query(ChatItem)
    #             .filter(ChatItem.conversation_id == id)
    #             .order_by(asc(getattr(ChatItem, 'created_at')))  # Changed to ascending order by index
    #             .offset(skip)
    #             .limit(limit)
    #             .all()
    #         )
            
    #     return chat_history

    def get_by_id(  # New name for clarity, or modify get_by_id
        self,
        db: Session,
        *,
        id: uuid.UUID,
        include_deleted: bool = False
    ) -> Optional[ChatHistory]:
        """
        Get a ChatHistory ORM object by ID.
        This method DOES NOT load or assign paginated chat_items to the instance.
        Use the 'chat_items' dynamic relationship on the returned object for querying items.
        """
        query = db.query(self.model).filter(self.model.conversation_id == id)

        if not include_deleted:
            query = query.filter(self.model.is_deleted == False)

        chat_history_orm = query.first()
        return chat_history_orm
    
    def get_conversation(
        self, 
        db: Session, 
        conversation_id: uuid.UUID,
        include_deleted: bool = False
    ) -> Optional[ChatHistory]:
        """Get conversation without loading chat items"""
        query = db.query(self.model).filter(ChatHistory.conversation_id == conversation_id)
        
        if not include_deleted:
            query = query.filter(ChatHistory.is_deleted == False)
            
        return query.first()
    
    def get_chat_history(
        self, 
        db: Session, 
        *,
        user_id: int,
        skip: int = 0,
        limit: int = 20,
        include_archived: bool = False,
        include_deleted: bool = False
    ) -> List[ChatHistory]:
        """
        Get all chat histories for a user with filtering options
        
        Args:
            db: Database session
            user_id: User ID
            skip: Number of records to skip
            limit: Maximum number of records to return
            include_archived: Whether to include archived chats
            include_deleted: Whether to include soft-deleted chats
            
        Returns:
            List of ChatHistory objects
        """
        query = db.query(self.model).filter(ChatHistory.user_id == user_id)
        
        if not include_deleted:
            query = query.filter(ChatHistory.is_deleted == False)
            
        if not include_archived:
            query = query.filter(ChatHistory.is_archived == False)
            
        return (
            query
            .order_by(desc(getattr(ChatHistory, 'updated_at')))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def search_chat_history(
        self,
        db: Session,
        *,
        user_id: int,
        search_params: ChatSearch
    ) -> List[ChatHistory]:
        """
        Search chat histories with various filters
        
        Args:
            db: Database session
            user_id: User ID
            search_params: Search parameters
            
        Returns:
            List of ChatHistory objects matching the search criteria
        """
        query = db.query(self.model).filter(ChatHistory.user_id == user_id)
        
        # Apply filters
        if not search_params.include_deleted:
            query = query.filter(ChatHistory.is_deleted == False)
            
        if not search_params.include_archived:
            query = query.filter(ChatHistory.is_archived == False)
        
        # Search by text if provided
        if search_params.query:
            search_text = f"%{search_params.query}%"
            
            # First search by title
            title_matches = query.filter(ChatHistory.title.ilike(search_text))
            
            # Then search by content
            content_matches = db.query(self.model).join(
                ChatItem, ChatHistory.conversation_id == ChatItem.conversation_id
            ).filter(
                ChatHistory.user_id == user_id,
                or_(
                    func.cast(ChatItem.content, text("VARCHAR")).ilike(search_text),
                    # For JSONB content with text field
                    func.cast(func.jsonb_extract_path_text(ChatItem.content, 'text'), text("VARCHAR")).ilike(search_text)
                )
            )
            
            if not search_params.include_deleted:
                content_matches = content_matches.filter(ChatHistory.is_deleted == False)
                
            if not search_params.include_archived:
                content_matches = content_matches.filter(ChatHistory.is_archived == False)
            
            # Combine results
            query = title_matches.union(content_matches)
        
        # Filter by date range if provided
        if search_params.start_date:
            query = query.filter(ChatHistory.created_at >= search_params.start_date)
            
        if search_params.end_date:
            query = query.filter(ChatHistory.created_at <= search_params.end_date)
        
        # Apply sorting
        if search_params.sort_by == "title":
            query = query.order_by(
                desc(ChatHistory.title) if search_params.sort_desc else asc(ChatHistory.title)
            )
        else:  # Default to updated_at
            query = query.order_by(
                desc(ChatHistory.updated_at) if search_params.sort_desc else asc(ChatHistory.updated_at)
            )
        
        return query.offset(search_params.skip).limit(search_params.limit).all()
    
    def archive_chat(self, db: Session, *, id: uuid.UUID, user_id: int) -> Optional[ChatHistory]:
        """Archive a chat history"""
        chat_history = self.get_conversation(db, conversation_id=id)
        if chat_history and chat_history.user_id == user_id:
            chat_history.archive()
            db.commit()
            db.refresh(chat_history)
            return chat_history
        return None
    
    def unarchive_chat(self, db: Session, *, id: uuid.UUID, user_id: int) -> Optional[ChatHistory]:
        """Unarchive a chat history"""
        chat_history = self.get_conversation(db, conversation_id=id, include_deleted=False)
        if chat_history and chat_history.user_id == user_id:
            chat_history.unarchive()
            db.commit()
            db.refresh(chat_history)
            return chat_history
        return None
    
    def soft_delete(self, db: Session, *, id: uuid.UUID, user_id: int) -> bool:
        """Soft delete a chat history"""
        chat_history = self.get_conversation(db, conversation_id=id)
        if chat_history and chat_history.user_id == user_id:
            chat_history.soft_delete()
            db.commit()
            return True
        return False
    
    def get_chat_statistics(self, db: Session, *, user_id: int) -> Dict[str, Any]:
        """Get chat statistics for a user"""
        total_conversations = (
            db.query(func.count(ChatHistory.conversation_id))
            .filter(
                ChatHistory.user_id == user_id,
                ChatHistory.is_deleted == False
            )
            .scalar()
        )
        
        # Total number of messages
        total_messages = (
            db.query(func.count(ChatItem.id))
            .join(ChatHistory, ChatItem.conversation_id == ChatHistory.conversation_id)
            .filter(
                ChatHistory.user_id == user_id,
                ChatHistory.is_deleted == False
            )
            .scalar()
        )
        
        # Messages by sender
        messages_by_sender = (
            db.query(
                ChatItem.sender,
                func.count(ChatItem.id).label('count')
            )
            .join(ChatHistory, ChatItem.conversation_id == ChatHistory.conversation_id)
            .filter(
                ChatHistory.user_id == user_id,
                ChatHistory.is_deleted == False
            )
            .group_by(ChatItem.sender)
            .all()
        )
        
        # Rating statistics
        rating_stats = (
            db.query(
                ChatItem.rating,
                func.count(ChatItem.id).label('count')
            )
            .join(ChatHistory, ChatItem.conversation_id == ChatHistory.conversation_id)
            .filter(
                ChatHistory.user_id == user_id,
                ChatHistory.is_deleted == False,
                ChatItem.rating.isnot(None)
            )
            .group_by(ChatItem.rating)
            .all()
        )
        
        return {
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "messages_by_sender": {item[0]: item[1] for item in messages_by_sender},
            "rating_stats": {item[0].name: item[1] for item in rating_stats},
            "archived_conversations": (
                db.query(func.count(ChatHistory.conversation_id))
                .filter(
                    ChatHistory.user_id == user_id,
                    ChatHistory.is_archived == True,
                    ChatHistory.is_deleted == False
                )
                .scalar()
            )
        }


class CRUDChatItem(CRUDBase[ChatItem, ChatItemCreate, ChatItemUpdate]):
    def create_with_chat(
        self, 
        db: Session, 
        *, 
        obj_in: Union[ChatItemCreate, Dict[str, Any]], 
        conversation_id: uuid.UUID
    ) -> ChatItem:
        """Create a new chat item and update the chat history's updated_at timestamp"""
        chat_history = db.query(ChatHistory).filter(
            ChatHistory.conversation_id == conversation_id
        ).first()
        
        if not chat_history:
            raise ValueError(f"Chat history with ID {conversation_id} not found")
        
        # Get data from obj_in
        if isinstance(obj_in, dict):
            create_data = obj_in.copy()
        else:
            create_data = obj_in.dict(exclude_unset=True)
        
        create_data["conversation_id"] = conversation_id
        
        db_obj = self.model(**create_data)
        db.add(db_obj)
        
        # Update chat history timestamp
        chat_history.updated_at = func.now()
        
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def set_rating(
        self, 
        db: Session, 
        *, 
        id: uuid.UUID, 
        rating: Rating, 
        user_id: int
    ) -> Optional[ChatItem]:
        """Set a rating for a chat item"""
        chat_item = db.query(self.model).filter(ChatItem.id == id).first()
        
        if not chat_item:
            return None
            
        # Check if the chat item belongs to the user
        chat_history = db.query(ChatHistory).filter(
            ChatHistory.conversation_id == chat_item.conversation_id
        ).first()
        
        if not chat_history or chat_history.user_id != user_id:
            return None
        
        # Set the rating
        chat_item.set_rating(rating)
        db.commit()
        db.refresh(chat_item)
        return chat_item
    
    def get_chat_items(
        self, 
        db: Session, 
        *, 
        conversation_id: uuid.UUID, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[ChatItem]:
        """Get all chat items for a conversation with pagination"""
        return (
            db.query(self.model)
            .filter(ChatItem.conversation_id == conversation_id)
            .order_by(asc(ChatItem.index))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def add_reply(
        self, 
        db: Session, 
        *, 
        parent_id: uuid.UUID, 
        obj_in: Union[ChatItemCreate, Dict[str, Any]]
    ) -> ChatItem:
        """Add a reply to a chat item"""
        parent = db.query(self.model).filter(ChatItem.id == parent_id).first()
        
        if not parent:
            raise ValueError(f"Parent chat item with ID {parent_id} not found")
        
        if isinstance(obj_in, dict):
            create_data = obj_in.copy()
        else:
            create_data = obj_in.dict(exclude_unset=True)
        
        create_data["conversation_id"] = parent.conversation_id
        create_data["parent_id"] = parent_id
        
        db_obj = self.model(**create_data)
        db.add(db_obj)
        
        chat_history = db.query(ChatHistory).filter(
            ChatHistory.conversation_id == parent.conversation_id
        ).first()
        
        if chat_history:
            chat_history.updated_at = func.now()
        
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def edit_content(
        self, 
        db: Session, 
        *, 
        id: uuid.UUID, 
        new_content: Dict[str, Any], 
        user_id: int
    ) -> Optional[ChatItem]:
        """Edit the content of a chat item"""
        chat_item = db.query(self.model).filter(ChatItem.id == id).first()
        
        if not chat_item:
            return None
            
        chat_history = db.query(ChatHistory).filter(
            ChatHistory.conversation_id == chat_item.conversation_id
        ).first()
        
        if not chat_history or chat_history.user_id != user_id:
            return None
        
        chat_item.edit_content(new_content)
        db.commit()
        db.refresh(chat_item)
        return chat_item


# Instantiate CRUD objects
chat = CRUDChat(ChatHistory)
chat_item = CRUDChatItem(ChatItem)