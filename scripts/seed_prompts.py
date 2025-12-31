
import logging
from sqlalchemy.orm import Session
from private_gpt.users.db.session import SessionLocal
from private_gpt.users.models.prompt import Prompt
from private_gpt.server.chat.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    RETRIEVAL_SYSTEM_PROMPT,
    AGENTIC_SYSTEM_PROMPT,
    ENHANCED_QA_TEMPLATE,
    ENHANCED_CONDENSE_PROMPT,
    ENHANCED_DECOMPOSE_PROMPT,
    ENHANCED_CONTEXT_PROMPT
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_prompts():
    db = SessionLocal()
    try:
        # Default prompts to seed
        # (mode, type, name, content)
        default_prompts = [
            ("chat", "system", "Default Chat System Prompt", DEFAULT_SYSTEM_PROMPT),
            ("search", "system", "Default Retrieval System Prompt", RETRIEVAL_SYSTEM_PROMPT),
            ("agentic", "system", "Default Agentic System Prompt", AGENTIC_SYSTEM_PROMPT),
            
            ("chat", "qa", "Default Chat QA Template", ENHANCED_QA_TEMPLATE),
            ("search", "qa", "Default Retrieval QA Template", ENHANCED_QA_TEMPLATE),
            ("agentic", "qa", "Default Agentic QA Template", ENHANCED_QA_TEMPLATE),
            
            ("chat", "condense", "Default Condense Prompt", ENHANCED_CONDENSE_PROMPT),
            ("search", "condense", "Default Condense Prompt", ENHANCED_CONDENSE_PROMPT),
            ("agentic", "condense", "Default Condense Prompt", ENHANCED_CONDENSE_PROMPT),
            
            ("chat", "decompose", "Default Decompose Prompt", ENHANCED_DECOMPOSE_PROMPT),
            ("search", "decompose", "Default Decompose Prompt", ENHANCED_DECOMPOSE_PROMPT),
            ("agentic", "decompose", "Default Decompose Prompt", ENHANCED_DECOMPOSE_PROMPT),

            ("chat", "context", "Default Context Prompt", ENHANCED_CONTEXT_PROMPT),
            ("search", "context", "Default Context Prompt", ENHANCED_CONTEXT_PROMPT),
            ("agentic", "context", "Default Context Prompt", ENHANCED_CONTEXT_PROMPT),
        ]

        for mode, type, name, content in default_prompts:
            # Check if this default prompt already exists
            existing = db.query(Prompt).filter(
                Prompt.mode == mode,
                Prompt.type == type,
                Prompt.is_default == True
            ).first()

            if existing:
                if existing.content != content:
                    logger.info(f"Updating default prompt: {mode} / {type}")
                    existing.content = content
                    existing.version += 1
                else:
                    logger.info(f"Default prompt {mode} / {type} is up to date")
            else:
                logger.info(f"Creating new default prompt: {mode} / {type}")
                new_prompt = Prompt(
                    mode=mode,
                    type=type,
                    name=name,
                    content=content,
                    is_default=True,
                    is_active=True,
                    version=1
                )
                db.add(new_prompt)

        db.commit()
        logger.info("Successfully seeded prompts.")
    except Exception as e:
        logger.error(f"Error seeding prompts: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_prompts()
