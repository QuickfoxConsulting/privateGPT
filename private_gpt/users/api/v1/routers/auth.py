import logging
import traceback
from typing import Any, Optional
from datetime import timedelta, datetime

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import APIRouter, Body, Depends, HTTPException, Security, status

from private_gpt.users.api import deps
from private_gpt.users.core import security
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.users.utils import send_registration_email, Ldap

logger = logging.getLogger(__name__)

LDAP_SERVER = settings.LDAP_SERVER
LDAP_ENABLE = settings.LDAP_ENABLE

# Login attempt settings
MAX_FAILED_ATTEMPTS = 5  # Maximum number of failed attempts before lockout
LOCKOUT_DURATION = 30  # Lockout duration in minutes
RESET_AFTER = 24  # Reset failed attempts after this many hours

router = APIRouter(prefix="/auth", tags=["auth"])

def check_account_locked(user: models.User) -> bool:
    """Check if the account is locked due to too many failed attempts."""
    if not user.failed_login_attempts or user.failed_login_attempts < MAX_FAILED_ATTEMPTS:
        return False
        
    if not user.last_failed_login:
        return False
        
    lockout_until = user.last_failed_login + timedelta(minutes=LOCKOUT_DURATION)
    if datetime.now() < lockout_until:
        return True        
    return False

def reset_failed_attempts(db: Session, user: models.User) -> None:
    """Reset failed login attempts after successful login."""
    user_in = schemas.LoginAttempt(
        failed_login_attempts=0,
        last_failed_login=None
    )
    crud.user.update(db, db_obj=user, obj_in=user_in)

def increment_failed_attempts(db: Session, user: models.User) -> None:
    """Increment failed login attempts counter."""
    attempts = (user.failed_login_attempts or 0) + 1
    user_in = schemas.LoginAttempt(
        failed_login_attempts=attempts,
        last_failed_login=datetime.now()
    )
    crud.user.update(db, db_obj=user, obj_in=user_in)

def register_user(
    db: Session,
    email: str,
    fullname: str,
    password: str,
    company: Optional[models.Company] = None,
    department: Optional[models.Department] = None,
    role: Optional[str] = None,
) -> models.User:
    """
    Register a new user in the database.
    """
    logging.info(f"User : {email} Password: {password} company_id: {company.id} deparment_id: {department.id}")
    user_in = schemas.UserCreate(
            email=email,
            password=password,
            username=fullname,
            company_id=company.id,
            department_id=department.id,
            checker= True if role == 'OPERATOR' else False
        )    
    try:
        send_registration_email(fullname, email, password)
    except Exception as e:
        logging.info(f"Failed to send registration email: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to send registration email.")
    return crud.user.create(db, obj_in=user_in)


def ldap_login(db, username, password):
    ldap = Ldap(LDAP_SERVER, username, password)
    username = ldap.who_am_i()
    department = ldap.get_department(username)
    if not ldap:
        raise HTTPException(
            status_code=400, detail="Incorrect email or password"
        )
    return username, department[0]

def create_user_role(
    db: Session,
    user: models.User,
    role_name: str,
    company: Optional[models.Company] = None,
) -> models.UserRole:
    """
    Create a user role in the database.
    """
    role = crud.role.get_by_name(db, name=role_name)
    user_role_in = schemas.UserRoleCreate(user_id=user.id, role_id=role.id, company_id=company.id if company else None)
    return crud.user_role.create(db, obj_in=user_role_in)


def create_token_payload(user: models.User, user_role: models.UserRole) -> dict:
    """
    Create a token payload for authentication.
    """
    return {
        "id": str(user.id),
        "email": str(user.email),
        "role": user_role.role.name,
        "username": str(user.username),
        "company_id": user_role.company.id if user_role.company else None,
        "department_id": user.department_id
    }

def ad_user_register(
    db: Session,
    email: str,
    fullname: str,
    password: str,
    department_id: int,
) -> models.User:
    """
    Register a new user in the database. Company id is directly given here.
    """
    user_in = schemas.UserCreate(email=email, password=password, username=fullname, company_id=1, department_id=department_id, checker=False)
    user = crud.user.create(db, obj_in=user_in)
    user_role_name = Role.GUEST["name"]
    company = crud.company.get(db, 1)

    user_role = create_user_role(db, user, user_role_name, company)
    return user


@router.post("/login", response_model=schemas.TokenSchema)
def login_access_token(
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    def ad_auth(LDAP_ENABLE):
        if LDAP_ENABLE:
            existing_user = crud.user.get_by_email(db, email=form_data.username)
            
            if existing_user:
                if existing_user.user_role.role.name == "SUPER_ADMIN":
                    return existing_user
                else:
                    try:
                        username, department = ldap_login(db=db, username=form_data.username, password=form_data.password)
                        return crud.user.get_by_name(db, name=username)
                    except HTTPException:
                        if existing_user:
                            increment_failed_attempts(db, existing_user)
                        raise
            else:
                try:
                    username, department = ldap_login(db=db, username=form_data.username, password=form_data.password)
                    depart = crud.department.get_by_department_name(db, name=department)

                    if depart:
                        user = ad_user_register(db=db, email=form_data.username, fullname=username, password=form_data.password, department_id=depart.id)
                    else:
                        department_in = schemas.DepartmentCreate(name=department)
                        new_department = crud.department.create(db, obj_in=department_in)
                        user = ad_user_register(db=db, email=form_data.username, fullname=username, password=form_data.password, department_id=new_department.id)
                    return user
                except HTTPException:
                    raise
        return None
    
    # Log login attempt
    log_audit(
        model='User', 
        action='login_attempt',
        details={
            "email": form_data.username,
            "ip_address": "captured_in_logger"
        },
        severity="INFO"
    )
    
    existing_user = crud.user.get_by_email(db, email=form_data.username)
    if existing_user and check_account_locked(existing_user):
        lockout_until = existing_user.last_failed_login + timedelta(minutes=LOCKOUT_DURATION)
        remaining_time = (lockout_until - datetime.now()).total_seconds() / 60
        
        # Log account lockout
        log_audit(
            model='User', 
            action='login_locked',
            details={
                "email": form_data.username,
                "remaining_time_minutes": int(remaining_time),
                "failed_attempts": existing_user.failed_login_attempts
            },
            severity="WARNING"
        )
        
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=f"Account is locked due to too many failed attempts. Please try again in {int(remaining_time)} minutes."
        )
    
    if LDAP_ENABLE:
        user = ad_auth(LDAP_ENABLE)
        if not user:
            # Log failed LDAP authentication
            log_audit(
                model='User', 
                action='login_failed_ldap',
                details={
                    "email": form_data.username,
                },
                severity="WARNING"
            )
            raise HTTPException(
                status_code=403,
                detail="Invalid Credentials!!!",
            )
    else:
        user = crud.user.authenticate(
            db, email=form_data.username, password=form_data.password
        )
        if not user:
            if existing_user:
                increment_failed_attempts(db, existing_user)
                remaining_attempts = MAX_FAILED_ATTEMPTS - (existing_user.failed_login_attempts or 0)
                
                # Log failed login with existing user
                log_audit(
                    model='User', 
                    action='login_failed',
                    details={
                        "email": form_data.username,
                        "remaining_attempts": remaining_attempts,
                        "failed_attempts": existing_user.failed_login_attempts
                    },
                    severity="WARNING"
                )
                
                raise HTTPException(
                    status_code=400, 
                    detail=f"Incorrect email or password. {remaining_attempts} attempts remaining."
                )
            else:
                # Log failed login with non-existing user
                log_audit(
                    model='User', 
                    action='login_failed_unknown_user',
                    details={
                        "email": form_data.username,
                    },
                    severity="WARNING"
                )
                
                raise HTTPException(
                    status_code=400,
                    detail="Invalid email or password."
                )

    # Reset failed attempts on successful login
    reset_failed_attempts(db, user)
    
    access_token_expires = timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    refresh_token_expires = timedelta(
        minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
    )
    user_in = schemas.UserUpdate(
        last_login=datetime.now()
    )
    user = crud.user.update(db, db_obj=user, obj_in=user_in)
    if user.user_role:
        role = user.user_role.role.name
        if user.user_role.company_id:
            company_id = user.user_role.company_id
        else: company_id = None
    
    token_payload = {
        "id": str(user.id),
        "email": str(user.email),
        "username": str(user.username),
        "role": role,
        "company_id": company_id,
        "department_id": str(user.department_id),
    }

    response_dict = {
        "access_token": security.create_access_token(
            token_payload, expires_delta=access_token_expires
        ),
        "refresh_token": security.create_refresh_token(
            token_payload, expires_delta=refresh_token_expires
        ),
        "user": token_payload,
        "token_type": "bearer",
    }
    
    # Log successful login
    log_audit(
        model='User', 
        action='login_success',
        details={
            "user_id": user.id,
            "username": user.username,
            "email": user.email,
            "role": role
        }, 
        user_id=user.id,
        username=user.username,
        severity="INFO"
    )
    
    return JSONResponse(content=response_dict)


@router.post("/login/refresh-token", response_model=schemas.TokenSchema)
def refresh_access_token(
    db: Session = Depends(deps.get_db),
    refresh_token: str = Body(..., embed=True),
) -> Any:
    token_payload = security.verify_refresh_token(refresh_token)

    if not token_payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    access_token_expires = timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(
        minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)

    response_dict = {
        "access_token": security.create_access_token(token_payload, expires_delta=access_token_expires),
        "refresh_token": security.create_refresh_token(token_payload, expires_delta=refresh_token_expires),
        "token_type": "bearer",
    }
    return JSONResponse(content=response_dict)


@router.post("/register", response_model=schemas.TokenSchema)
def register(
    *,
    log_audit: models.Audit = Depends(deps.get_audit_logger),

    db: Session = Depends(deps.get_db),
    email: str = Body(...),
    fullname: str = Body(...),
    # password: str = Body(...),
    department_id: int = Body(None, title="Department ID",
                                description="Department name for the user (if applicable)"),
    role_name: str = Body(None, title="Role Name",
                          description="User role name (if applicable)"),
    current_user: models.User = Security(
        deps.get_current_active_user,
        scopes=[Role.ADMIN["name"], 
                Role.SUPER_ADMIN["name"],
                Role.OPERATOR["name"]],
    ),
) -> Any:
    """
    Register new user with optional company assignment and role selection.
    """
    
    # Log registration attempt
    log_audit(
        model='User', 
        action='registration_attempt',
        details={
            "requester_id": current_user.id,
            "requester_username": current_user.username,
            "new_user_email": email,
            "new_user_fullname": fullname,
            "department_id": department_id,
            "role_name": role_name
        },
        user_id=current_user.id,
        username=current_user.username,
        severity="INFO"
    )

    existing_user = crud.user.get_by_email(db, email=email)
    if existing_user:
        # Log duplicate user attempt
        log_audit(
            model='User', 
            action='registration_duplicate',
            details={
                "status": '409', 
                'detail': "The user with this email already exists!",
                "requester_id": current_user.id,
                "requester_username": current_user.username,
                "existing_user_id": existing_user.id,
                "existing_user_email": email
            },
            user_id=current_user.id,
            username=current_user.username,
            severity="WARNING"
        )
        raise HTTPException(
            status_code=409,
            detail="The user with this email already exists!",
        )
    random_password = security.generate_random_password()
    # random_password = password
    try:
        company_id = current_user.company_id
        if company_id:
            company = crud.company.get(db, company_id)
            if not company:
                # Log company not found
                log_audit(
                    model='User', 
                    action='registration_company_not_found',
                    details={
                        "requester_id": current_user.id,
                        "requester_username": current_user.username,
                        "company_id": company_id
                    },
                    user_id=current_user.id,
                    username=current_user.username,
                    severity="ERROR"
                )
                raise HTTPException(
                    status_code=404,
                    detail="Company not found.",
                )
            if department_id:
                department = crud.department.get_by_id(
                    db=db, id=department_id)
                if not department:
                    # Log department not found
                    log_audit(
                        model='User', 
                        action='registration_department_not_found',
                        details={
                            "requester_id": current_user.id,
                            "requester_username": current_user.username,
                            "department_id": department_id
                        },
                        user_id=current_user.id,
                        username=current_user.username,
                        severity="ERROR"
                    )
                    raise HTTPException(
                        status_code=404,
                        detail="Department not found.",
                    )
                logging.info(f"Department is {department}")
            user = register_user(
                db, email, fullname, random_password, company, department, role_name
            )
            user_role_name = role_name or Role.GUEST["name"]
            user_role = create_user_role(db, user, user_role_name, company)
            log_audit(model='user_roles', action='create',
                      details={'detail': "User role created successfully.", }, user_id=current_user.id)
            
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Unable to create account.",
        )
    
    token_payload = create_token_payload(user, user_role)
    response_dict = {
        "access_token": security.create_access_token(token_payload, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)),
        "refresh_token": security.create_refresh_token(token_payload, expires_delta=timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)),
        "token_type": "bearer",
    }
    log_audit(model='User', action='creation',
              details={'detail': "User created successfully.",'username':fullname}, user_id=current_user.id)

    return JSONResponse(content=response_dict, status_code=status.HTTP_201_CREATED)

@router.post("/unlock-account")
def unlock_account(
    *,
    db: Session = Depends(deps.get_db),
    email: str = Body(...),
    current_user: models.User = Security(
        deps.get_current_active_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Unlock a user account that has been locked due to too many failed login attempts.
    Only accessible by admins and super admins.
    """
    user = crud.user.get_by_email(db, email=email)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
        
    reset_failed_attempts(db, user)
    
    return {"message": f"Account for {email} has been unlocked successfully."}
