from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from fastapi_pagination import Page, paginate

from private_gpt.users.api import deps
from private_gpt.users.models import User
from private_gpt.users.schemas.prompt import PromptCreate, PromptUpdate, PromptSchema
from private_gpt.users.services.prompt_service import prompt_service

router = APIRouter(prefix="/prompts", tags=["prompts"])

@router.post("", response_model=PromptSchema)
def create_prompt(
    prompt_in: PromptCreate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Create a new prompt.
    """
    return prompt_service.create_prompt(db, prompt_in, current_user.id)

@router.get("", response_model=Page[PromptSchema])
def read_prompts(
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Retrieve user prompts.
    """
    return paginate(prompt_service.get_user_prompts(db, current_user.id))

@router.get("/active", response_model=dict)
def read_active_prompt_text(
    mode: str,
    type: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Get the currently active prompt content for a given mode and type.
    This resolves the hierarchy (User > Default).
    Returns the resolved content wrapped in a schematic structure (partially filled).
    """
    content = prompt_service.get_active_prompt(db, current_user.id, mode, type)
    if not content:
        raise HTTPException(status_code=404, detail="No active prompt found")
    
    return {"content": content}

@router.get("/resolved", response_model=dict)
def get_resolved_prompt(
    mode: str,
    type: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Get the resolved prompt text (Default + Override).
    """
    content = prompt_service.get_resolved_prompt(db, current_user.id, mode, type)
    return {"content": content}

@router.get("/default", response_model=dict)
def get_default_prompt(
    mode: str,
    type: str,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Get the system default prompt text.
    """
    prompt = prompt_service.get_default_prompt(db, mode, type)
    if not prompt:
        raise HTTPException(status_code=404, detail="Default prompt not found")
    return {"content": prompt.content}

@router.put("/{prompt_id}", response_model=PromptSchema)
def update_prompt(
    prompt_id: int,
    prompt_in: PromptUpdate,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Update a prompt.
    """
    prompt = prompt_service.get_prompt_by_id(db, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="Prompt not found")
    if prompt.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    updated_prompt = prompt_service.update_prompt(db, prompt_id, prompt_in)
    return updated_prompt

@router.delete("/{prompt_id}")
def delete_prompt(
    prompt_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_user),
) -> Any:
    """
    Delete a prompt.
    """
    prompt = prompt_service.get_prompt_by_id(db, prompt_id)
    if not prompt:
         raise HTTPException(status_code=404, detail="Prompt not found")
    if prompt.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")

    prompt_service.delete_prompt(db, prompt_id)
    return {"message": "Prompt deleted successfully"}
