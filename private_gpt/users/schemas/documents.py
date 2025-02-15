from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from typing import List
from fastapi import Form, UploadFile, File

from .category import CategoryList
from private_gpt.users.models.enums import *

class DocumentsBase(BaseModel):
    filename: str

class DepartmentList(BaseModel):
    id: int
    name: str

class DocumentCreate(DocumentsBase):
    uploaded_by: int

class DocumentUpdate(BaseModel):
    id: int
    status: str

class DocumentEnable(BaseModel):
    id: int
    is_enabled: bool

class DocumentDepartmentUpdate(DocumentsBase):
    departments: List[int] = []

class Document(BaseModel):
    id: int
    is_enabled: bool
    filename: str
    tags: str
    uploaded_by: int
    uploaded_at: datetime
    departments: List[DepartmentList] = []

    class Config:
        orm_mode = True

class DocumentMakerChecker(DocumentCreate):
    tags: Optional[str] = None

class DocumentMakerCreate(DocumentMakerChecker):
    pass

class UrlMakerChecker(BaseModel):
    filename: str
    uploaded_by: int
    action_type: str
    status: str

class DocumentCheckerUpdate(BaseModel):
    is_enabled: bool
    verified_at: datetime
    verified_by: int
    verified: bool

class DocumentVerify(BaseModel):
    id: int
    filename: str
    uploaded_by: str
    uploaded_at: datetime
    departments: List[DepartmentList] = []
    status: str
    categories: List[CategoryList] = []

    class Config:
        orm_mode = True

class DocumentFilter(BaseModel):
    filename: Optional[str] = None
    tags: Optional[str] = None
    uploaded_by: Optional[str] = None
    action_type: Optional[str] = None
    status: Optional[str] = None
    order_by: Optional[str] = None
    category_id: Optional[str] = None

class DocumentVersionBase(BaseModel):
    """Base schema for document version information."""
    status: Optional[MakerCheckerStatus] = None
    action_type: Optional[MakerCheckerActionType] = None 
    changes: Optional[str] = None
    file_path: str

    class Config:
        use_enum_values = True

class DocumentVersionCreate(DocumentVersionBase):
    """Schema for creating a new document version."""
    document_id: int
    uploaded_by: int

class DocumentVersionUpdate(DocumentVersionBase):
    """Schema for updating an existing document version."""
    reviewed_by: Optional[int] = None
    reviewed_at: Optional[datetime] = None

class DocumentVersionOut(DocumentVersionBase):
    """Schema for document version output in API responses."""
    id: int
    version_number: int
    uploaded_at: datetime
    reviewed_at: Optional[datetime]
    uploaded_by: int
    reviewed_by: Optional[int] = None

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "version_number": 1,
                "status": "APPROVED",
                "action_type": "CREATE",
                "changes": "Initial version",
                "file_path": "/path/to/file",
                "uploaded_at": "2024-02-14T12:00:00",
                "uploaded_by": 1,
                "reviewed_at": "2024-02-14T13:00:00",
                "reviewed_by": 2
            }
        }

class DocumentView(BaseModel):
    """Schema for document list view with related information."""
    id: int
    is_enabled: bool
    filename: str
    tags: str = "" 
    uploaded_by: str
    uploaded_at: datetime
    departments: List[DepartmentList] = []
    categories: List[CategoryList] = []
    version: Optional[DocumentVersionOut] = None

    class Config:
        orm_mode = True
        schema_extra = {
            "example": {
                "id": 1,
                "is_enabled": True,
                "filename": "example.pdf",
                "tags": "important,confidential",
                "uploaded_by": "john.doe",
                "uploaded_at": "2024-02-14T12:00:00",
                "departments": [{"id": 1, "name": "HR"}],
                "categories": [{"id": 1, "name": "Policies"}],
                "version": {
                    "id": 1,
                    "version_number": 1,
                    "status": "APPROVED",
                    "action_type": "CREATE",
                    "changes": "Initial version",
                    "file_path": "/path/to/file",
                    "uploaded_at": "2024-02-14T12:00:00",
                    "uploaded_by": 1,
                    "reviewed_at": "2024-02-14T13:00:00",
                    "reviewed_by": 2
                }
            }
        }

class DocCatUpdate(BaseModel):
    filename: str
    departments: Optional[List[int]] = None
    categories: Optional[List[int]] = None

class DocumentList(DocumentsBase):
    id: int
    is_enabled: bool
    uploaded_by: int
    uploaded_at: datetime
    vesion: Optional[DocumentVersionOut]
    categories: List[CategoryList] = []
    departments: List[DepartmentList] = []

    class Config:
        orm_mode = True

# =================
# Form Model
# =================
class UrlUpload(BaseModel):
    departments: str = Form(...)
    tags: Optional[str] = Form(...)
    category: int = Form(...)
    url: str = Form(...)

class DocumentUpload(BaseModel):
    departments: str = Form(...)
    tags: Optional[str] = Form(...)
    category: int = Form(...)
    file: UploadFile = File(...)

class DocumentCategoryUpdate(BaseModel):
    filename: str
    categories: List[int]