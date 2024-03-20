from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class AuditBase(BaseModel):
    id: int
    model: str
    user_id: int
    action: str
    details: dict
    ip_address: str
    timestamp: Optional[datetime]


class AuditCreate(AuditBase):
    pass


class AuditUpdate(AuditBase):
    id: int


class AuditInDB(AuditBase):
    id: int

    class Config:
        orm_mode = True

class Audit(BaseModel):
    id: int
    model: Optional[str]
    username: Optional[str]
    action: Optional[str]
    details: dict
    timestamp: Optional[datetime]
    ip_address: Optional[str]


class GetAudit(BaseModel):
    id: int