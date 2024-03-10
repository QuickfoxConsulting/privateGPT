from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class MakerCheckerBase(BaseModel):
    table_name: str
    record_id: int


class MakerCheckerCreate(MakerCheckerBase):
    action_type: str
    status: str
    created_by: int

class CheckerUpdate(MakerCheckerBase):
    id: int
    action_type: str
    status: str

class MakerCheckerUpdate(BaseModel):
    action_type: str
    status: str
    verified_at: Optional[datetime]
    verified_by: int

class MakerCheckerList(MakerCheckerBase):
    id: int

class MakerChecker(MakerCheckerBase):
    id: int

    class Config:
        orm_mode = True

