from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, EmailStr, root_validator
from private_gpt.users.schemas.user_role import UserRole
from private_gpt.users.schemas.company import Company


class UserBaseSchema(BaseModel):
	email: EmailStr
	username: str
	company_id: int
	department_id: int
	checker: bool

	class Config:
		arbitrary_types_allowed = True


class UserCreate(UserBaseSchema):
	password: str = Field(alias="password")


class UsernameUpdate(BaseModel):
	username: str


class UserUpdate(BaseModel):
	last_login: Optional[datetime] = None

class LoginAttempt(BaseModel):
	failed_login_attempts: int
	last_failed_login: Optional[datetime] = None

class UserLoginSchema(BaseModel):
	email: EmailStr = Field(alias="email")
	password: str

	class Config:
		arbitrary_types_allowed = True

class UserSchema(UserBaseSchema):
	id: int
	user_role: Optional[UserRole] = None
	last_login: Optional[datetime] = None
	created_at: datetime
	updated_at: datetime
	is_active: bool = Field(default=False)
	department_name: Optional[str] = None  # Add department name

	@root_validator(pre=True)
	def extract_department_name(cls, values):
		"""Extract department name from ORM model if available."""
		# If values is an ORM model instance (not a dict)
		if hasattr(values, 'department') and values.department:
			values.department_name = values.department.name
		return values

	class Config:
		orm_mode = True

class User(UserSchema):
    pass


class UserInDB(UserSchema):
    hashed_password: str


class Profile(UserBaseSchema):
	role: str


class DeleteUser(BaseModel):
	id: int


class UserAdminUpdate(BaseModel):
	id: int
	username: Optional[str] = None
	email: Optional[EmailStr] = None
	role: Optional[str] = None
	department_id: Optional[int] = None
	company_id: Optional[int] = None
	checker: Optional[bool] = None

class UserDepartmentUpdate(BaseModel):
	username: Optional[str] = None
	email: Optional[EmailStr] = None
	department_id: Optional[int] = None
	company_id: Optional[int] = None
	checker: Optional[bool] = None

	class Config:
		orm_mode = True

class UserAdmin(BaseModel):
	username: str
	department_id: int


class PasswordUpdate(BaseModel):
	password_created: Optional[datetime] = None
