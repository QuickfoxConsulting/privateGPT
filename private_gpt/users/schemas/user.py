
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, EmailStr
from private_gpt.users.models.user_role import UserRole


class UserBaseSchema(BaseModel):
	email: EmailStr
	fullname: str

	class Config:
		arbitrary_types_allowed = True

class UserCreate(UserBaseSchema):
	password: str = Field(alias="password")


class UserUpdate(UserBaseSchema):
	last_login: Optional[datetime] = None


class UserLoginSchema(BaseModel):
	email: EmailStr = Field(alias="email")
	password: str 
	
	class Config:
		arbitrary_types_allowed = True


class UserSchema(UserBaseSchema):
	id: int
	user_role: Optional[UserRole]
	last_login: Optional[datetime]
	created_at: datetime
	updated_at: datetime
	is_active: bool = Field(default=False)

	class Config:
		orm_mode = True
		json_exclude = {'user_role'}

# Additional properties to return via API
class User(UserSchema):
    pass

# Additional properties stored in DB
class UserInDB(UserSchema):
    hashed_password: str