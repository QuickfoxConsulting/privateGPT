from sqlalchemy import Enum

class UploadType(Enum):
    REGULAR = 'regular'
    SCANNED = 'scanned'
    BOTH = 'both'