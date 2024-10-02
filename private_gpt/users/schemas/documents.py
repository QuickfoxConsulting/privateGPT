from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from typing import List
from fastapi import Form, UploadFile, File

from fastapi_filter.contrib.sqlalchemy import Filter
from .category import CategoryList

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

class DocumentList(DocumentsBase):
    id: int
    is_enabled: bool
    uploaded_by: int
    uploaded_at: datetime
    departments: List[DepartmentList] = []

class Document(BaseModel):
    id: int
    is_enabled: bool
    filename: str
    uploaded_by: int
    uploaded_at: datetime
    departments: List[DepartmentList] = []

    class Config:
        orm_mode = True

class DocumentMakerChecker(DocumentCreate):
    action_type: str
    status: str
    doc_type_id: Optional[int]
    version_type: Optional[str]
    previous_document_id: Optional[int]

class DocumentMakerCreate(DocumentMakerChecker):
    pass

class UrlMakerChecker(BaseModel):
    filename: str
    uploaded_by: int
    action_type: str
    status: str

class DocumentCheckerUpdate(BaseModel):
    action_type: str
    status: str
    is_enabled: bool
    verified_at: datetime
    verified_by: int
    verified: bool

class DocumentDepartmentList(BaseModel):
    departments_ids: str = Form(...)
    doc_type_id: int = Form(...)
    category: int = Form(...)
    file: UploadFile = File(...)
    version_type: str = Form(...)  
    previous_document_id: Optional[int] = Form(None) 

class UrlUpload(BaseModel):
    departments_ids: str = Form(...)
    category: int = Form(...)
    url: str = Form(...)

class DocumentView(BaseModel):
    id: int
    is_enabled: bool
    filename: str
    uploaded_by: str
    uploaded_at: datetime
    departments: List[DepartmentList] = []
    action_type: str
    status: str
    categories: List[CategoryList] = []

    class Config:
        orm_mode = True


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
    uploaded_by: Optional[str] = None
    action_type: Optional[str] = None
    status: Optional[str] = None
    order_by: Optional[str] = None
    category_id: Optional[str] = None


class DocumentCategoryUpdate(BaseModel):
    filename: str
    categories: List[int]
    
class DocCatUpdate(BaseModel):
    filename: str
    departments: Optional[List[int]] = None
    categories: Optional[List[int]] = None