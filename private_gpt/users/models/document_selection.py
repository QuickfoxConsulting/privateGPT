from private_gpt.users.db.base_class import Base
from sqlalchemy import Column, Integer, Table, ForeignKey, DateTime, String
from datetime import datetime

document_selection_association = Table(
    "document_selection_association",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE", onupdate="CASCADE")),
    Column("document_id", Integer, ForeignKey("document.id", ondelete="CASCADE", onupdate="CASCADE")),
    Column("created_at", DateTime, default=datetime.now),
    Column("notes", String(512), nullable=True),  
    Column("selection_group", String(128), nullable=True),
    extend_existing=True
)