from enum import Enum as PythonEnum

class MakerCheckerStatus(PythonEnum):
    PENDING = 'PENDING'
    APPROVED = 'APPROVED'
    REJECTED = 'REJECTED'
class MakerCheckerActionType(PythonEnum):
    INSERT = 'INSERT'
    UPDATE = 'UPDATE'
    DELETE = 'DELETE'

class DocumentStatus(PythonEnum):
    INGESTING = 'INGESTING'
    EMBEDDING = 'EMBEDDING'
    READY = 'READY'