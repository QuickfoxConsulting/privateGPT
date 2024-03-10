from enum import Enum as PythonEnum

class MakerCheckerStatus(PythonEnum):
    PENDING = 'pending'
    APPROVED = 'approved'
    REJECTED = 'rejected'


class MakerCheckerActionType(PythonEnum):
    INSERT = 'insert'
    UPDATE = 'update'
    DELETE = 'delete'