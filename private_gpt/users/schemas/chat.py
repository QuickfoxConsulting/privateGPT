from datetime import datetime
from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field, Json
import uuid
from private_gpt.users.models.chat import Rating, MessageStatus

class ChatItemBase(BaseModel):
    conversation_id: uuid.UUID
    sender: str
    content: Union[str, Dict]

class ChatItemCreate(ChatItemBase):
    pass

class ChatItemUpdate(ChatItemBase):
    rating: Optional[Rating]


class ChatItem(ChatItemBase):
    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    rating: Optional[Rating]
    
    class Config:
        orm_mode = True


class ChatHistoryBase(BaseModel):
    user_id: int
    title: Optional[str]
    

class ChatHistoryCreate(ChatHistoryBase):
    chat_items: Optional[List[ChatItemCreate]]

class ChatHistoryUpdate(ChatHistoryBase):
    updated_at: datetime
    chat_items: Optional[List[ChatItemCreate]]

class Chat(BaseModel):
    conversation_id: uuid.UUID
    title: Optional[str]
    created_at: datetime

class ChatHistory(ChatHistoryBase):
    conversation_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    chat_items: List[ChatItem]
    
    class Config:
        orm_mode = True

class ChatDelete(BaseModel):
    conversation_id: uuid.UUID

class CreateChatHistory(BaseModel):
    user_id: int


class ChatItemRating(BaseModel):
    rating: Rating

class ChatSearch(BaseModel):
    query: Optional[str] = Field(None, description="Search term for chat title or content")
    include_deleted: Optional[bool] = Field(False, description="Include deleted chat histories")
    include_archived: Optional[bool] = Field(False, description="Include archived chat histories")
    start_date: Optional[datetime] = Field(None, description="Start date for filtering chat histories")
    end_date: Optional[datetime] = Field(None, description="End date for filtering chat histories")
    sort_by: str = Field("updated_at", description="Field to sort by (title or updated_at)")
    sort_desc: bool = Field(True, description="Sort in descending order if True")
    skip: int = Field(0, ge=0, description="Number of records to skip for pagination")
    limit: int = Field(10, ge=1, le=100, description="Number of records to return for pagination")
    updated_at: Optional[datetime]