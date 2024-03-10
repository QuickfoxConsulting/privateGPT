from typing import Optional

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.makerchecker import MakerChecker
from private_gpt.users.schemas.makerchecker import MakerCheckerCreate, MakerCheckerUpdate
from sqlalchemy.orm import Session


class CRUDRole(CRUDBase[MakerChecker, MakerCheckerCreate, MakerCheckerUpdate]):
    def get_by_id(self, db: Session, *, id: int) -> Optional[MakerChecker]:
        return db.query(self.model).filter(MakerChecker.id == id).first()
    
makerchecker = CRUDRole(MakerChecker)