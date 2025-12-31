from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc

from private_gpt.users.models.prompt import Prompt
from private_gpt.users.schemas.prompt import PromptCreate, PromptUpdate

class PromptService:
    def create_prompt(self, db: Session, prompt_in: PromptCreate, user_id: int) -> Prompt:
        db_prompt = Prompt(
            name=prompt_in.name,
            content=prompt_in.content,
            description=prompt_in.description,
            type=prompt_in.type,
            mode=prompt_in.mode,
            is_active=prompt_in.is_active,
            is_default=prompt_in.is_default,
            user_id=user_id,
            version=1 # Initial version
        )
        
        # If this is set as active, deactivate other prompts of same type/mode for this user
        if prompt_in.is_active:
             self._deactivate_others(db, user_id, prompt_in.mode, prompt_in.type)

        db.add(db_prompt)
        db.commit()
        db.refresh(db_prompt)
        return db_prompt

    def get_prompt_by_id(self, db: Session, prompt_id: int) -> Optional[Prompt]:
        return db.query(Prompt).filter(Prompt.id == prompt_id).first()

    def get_user_prompts(self, db: Session, user_id: int) -> List[Prompt]:
        return db.query(Prompt).filter(Prompt.user_id == user_id).all()

    def update_prompt(self, db: Session, prompt_id: int, prompt_in: PromptUpdate) -> Optional[Prompt]:
        db_prompt = self.get_prompt_by_id(db, prompt_id)
        if not db_prompt:
            return None

        update_data = prompt_in.dict(exclude_unset=True)
        
        # If activating, deactivate others
        if update_data.get("is_active", False):
             self._deactivate_others(db, db_prompt.user_id, db_prompt.mode, db_prompt.type, exclude_id=prompt_id)

        for field, value in update_data.items():
            setattr(db_prompt, field, value)

        db.add(db_prompt)
        db.commit()
        db.refresh(db_prompt)
        return db_prompt

    def delete_prompt(self, db: Session, prompt_id: int) -> bool:
        db_prompt = self.get_prompt_by_id(db, prompt_id)
        if not db_prompt:
            return False
        db.delete(db_prompt)
        db.commit()
        return True

    def get_active_prompt(self, db: Session, user_id: Optional[int], mode: str, type: str) -> Optional[str]:
        """
        Get the prompt content for UI/API display.
        Priority: User's override > System Default.
        """
        # 1. Check user override
        if user_id:
            user_prompt = db.query(Prompt).filter(
                and_(
                    Prompt.user_id == user_id,
                    Prompt.mode == mode,
                    Prompt.type == type,
                    Prompt.is_active == True
                )
            ).order_by(desc(Prompt.version)).first()
            
            if user_prompt:
                return user_prompt.content

        # 2. Check system default
        system_prompt = self.get_default_prompt(db, mode, type)
        if system_prompt:
            return system_prompt.content
            
        return None

    def get_default_prompt(self, db: Session, mode: str, type: str) -> Optional[Prompt]:
        """Get the system default prompt."""
        return db.query(Prompt).filter(
            and_(
                Prompt.is_default == True,
                Prompt.mode == mode,
                Prompt.type == type,
                Prompt.is_active == True
            )
        ).order_by(desc(Prompt.version)).first()

    def get_resolved_prompt(self, db: Session, user_id: Optional[int], mode: str, type: str) -> Optional[str]:
        """
        Get the final resolved prompt for chat execution.
        If type is 'system', appends user override to default.
        For other types, strictly uses the default.
        """
        default_prompt = self.get_default_prompt(db, mode, type)
        default_content = default_prompt.content if default_prompt else None

        if type != "system":
             return default_content

        # If system prompt, try to append user override or fill placeholder
        if user_id:
            user_prompt = db.query(Prompt).filter(
                and_(
                     Prompt.user_id == user_id,
                     Prompt.mode == mode,
                     Prompt.type == type,
                     Prompt.is_active == True
                )
            ).order_by(desc(Prompt.version)).first()

            if user_prompt and user_prompt.content:
                if default_content:
                    # If default has a specific placeholder, use it
                    if "{user_override}" in default_content:
                        return default_content.replace("{user_override}", user_prompt.content)
                    return f"{default_content}\n\n{user_prompt.content}"
                return user_prompt.content
        
        # If no user override but placeholder exists, clear it
        if default_content and "{user_override}" in default_content:
             return default_content.replace("{user_override}", "")

        return default_content

    def _deactivate_others(self, db: Session, user_id: int, mode: str, type: str, exclude_id: int = None):
        """Deactivate all other prompts for this user/mode/type combination."""
        query = db.query(Prompt).filter(
            and_(
                Prompt.user_id == user_id,
                Prompt.mode == mode,
                Prompt.type == type,
                Prompt.is_active == True
            )
        )
        if exclude_id:
            query = query.filter(Prompt.id != exclude_id)
            

        prompts = query.all()
        for p in prompts:
            p.is_active = False
            db.add(p)
        # We don't commit here, expecting the caller to commit

prompt_service = PromptService()
