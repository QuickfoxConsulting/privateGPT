from typing import Optional

from private_gpt.users.models.role import Role
from pydantic import BaseModel

# Shared properties
class UserRoleBase(BaseModel):
    user_id: Optional[int]
    role_id: Optional[int]

    class Config:
        arbitrary_types_allowed = True


# Properties to receive via API on creation
class UserRoleCreate(UserRoleBase):
    pass

# Properties to receive via API on update
class UserRoleUpdate(BaseModel):
    role_id: int

    class Config:
        arbitrary_types_allowed = True
          
class UserRoleInDBBase(UserRoleBase):
    role: Role

    class Config:
        orm_mode = True
        arbitrary_types_allowed = True



# Additional properties to return via API
class UserRole(UserRoleInDBBase):
    pass


class UserRoleInDB(UserRoleInDBBase):
    pass