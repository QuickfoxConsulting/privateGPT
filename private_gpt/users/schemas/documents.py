import json
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from fastapi import Form, UploadFile, File

from .category import CategoryList
from private_gpt.users.models.enums import *
from private_gpt.server.ingest.ingest_service import ChunkingStrategy

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

class MetadataModel(BaseModel):
    tags: Optional[List[str]] = []
    departments: Optional[List[int]] = []
    category: Optional[int] = None
    
    class Config:
        extra = "allow" 

class Document(BaseModel):
    id: int
    is_enabled: bool
    filename: str
    doc_metadata: Dict[str, Any] = {}  # JSONB field for all doc_metadata
    doc_status: str
    uploaded_by: int
    uploaded_at: datetime
    departments: List[DepartmentList] = []  # Keep for backward compatibility

    model_config = ConfigDict(from_attributes=True)

class DocumentMakerChecker(DocumentCreate):
    doc_metadata: Optional[Dict[str, Any]] = None
    doc_status: str

class DocumentMakerCreate(DocumentMakerChecker):
    pass

class UrlMakerChecker(BaseModel):
    filename: str
    uploaded_by: int
    action_type: str
    status: str

class DocumentCheckerUpdate(BaseModel):
    filename: Optional[str] = None
    doc_metadata: Optional[Dict[str, Any]] = None
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
    doc_metadata: Optional[Dict[str, Any]] = {}

    model_config = ConfigDict(from_attributes=True)

class DocumentFilter(BaseModel):
    filename: Optional[str] = None
    tags: Optional[str] = None  # Search within doc_metadata.tags
    uploaded_by: Optional[str] = None
    action_type: Optional[str] = None
    status: Optional[str] = None
    order_by: Optional[str] = None
    category_id: Optional[str] = None  # Search within doc_metadata.category
    department_id: Optional[str] = None  # Search within doc_metadata.departments

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

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
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
    )

class DocumentView(BaseModel):
    """Schema for document list view with related information."""
    id: int
    is_enabled: bool
    filename: str
    doc_status: str
    doc_metadata: Dict[str, Any] = {}  # Updated to use JSONB doc_metadata
    uploaded_by: str
    uploaded_at: datetime
    departments: List[DepartmentList] = []
    categories: List[CategoryList] = []
    version: Optional[DocumentVersionOut] = None

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": 1,
                "is_enabled": True,
                "filename": "example.pdf",
                "doc_metadata": {
                    "tags": ["policy", "HR"],
                    "departments": [1, 2],
                    "category": 1,
                    "custom_field": "custom value"
                },
                "doc_status": "INGESTING",
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
    )

class DocCatUpdate(BaseModel):
    filename: str
    doc_metadata: Dict[str, Any] = {}  # Contains departments and categories

class DocumentList(DocumentsBase):
    id: int
    is_enabled: bool
    uploaded_by: int
    uploaded_at: datetime
    doc_status: str
    version: Optional[DocumentVersionOut]
    doc_metadata: Dict[str, Any] = {}
    categories: List[CategoryList] = []
    departments: List[DepartmentList] = []

    model_config = ConfigDict(from_attributes=True)

# =================
# Form Model
# =================
class UrlUpload(BaseModel):
    doc_metadata: str = Form(...)  # JSON string containing departments, tags, category
    url: str = Form(...)

class MetadataSchema(BaseModel):
    tags: Optional[List[str]] = []
    category: Optional[str] = None
    departments: Optional[List[Union[int, str]]] = []
    custom_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        extra = "allow" 

class DocumentCategoryUpdate(BaseModel):
    filename: str
    doc_metadata: Dict[str, Any] 

class DocumentFilePath(BaseModel):
    filename: str
    file_path: str

class DocumentUpload:
    def __init__(
        self,
        doc_metadata: str = Form(None),  # Made this optional by changing from Form(...) to Form(None)
        file: UploadFile = File(...),
        strategy: ChunkingStrategy = Form(ChunkingStrategy.LATE_CHUNKING)
    ):
        self.file = file
        self.metadata_raw = doc_metadata
        self.strategy = strategy

        # Handle optional metadata
        if doc_metadata and doc_metadata.strip():  
            try:
                parsed = json.loads(doc_metadata)
                self.doc_metadata = MetadataSchema(**parsed)
            except Exception as e:
                raise ValueError(f"Invalid doc_metadata JSON: {str(e)}")
        else:
            self.doc_metadata = MetadataSchema()
        

class DocumentSelection(BaseModel):
    document_ids: Optional[List[int]] = Field(..., description="List of document IDs to select")
    
    class Config:
        schema_extra = {
            "example": {
                "document_ids": [1, 2, 3, 4, 5]
            }
        }

class StatusUpdate(BaseModel):
    doc_status: str