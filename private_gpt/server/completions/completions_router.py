import os
from pathlib import Path
from private_gpt.users import crud, models, schemas
import itertools
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from fastapi import APIRouter, Depends, Request, Security, HTTPException, status
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.users.models.document import Document
from private_gpt.users.models.enums import MakerCheckerStatus
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from sqlalchemy.orm import Session
import traceback
import logging
import json
logger = logging.getLogger(__name__)

from starlette.responses import StreamingResponse

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
)
from private_gpt.server.chat.chat_router import ChatBody, chat_completion
from private_gpt.server.utils.auth import authenticated
from private_gpt.users.api import deps
from private_gpt.users import crud, models, schemas
import uuid
completions_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class CompletionsBody(BaseModel):
    conversation_id: uuid.UUID
    history: Optional[list[OpenAIMessage]]
    prompt: str
    system_prompt: str | None = None
    use_context: bool = False
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    stream: bool = False
    category_id: Optional[int] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "conversation_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "history": [
                        {
                            "role": "user",
                            "content": "Hello!"
                        },
                        {
                            "role": "assistant",
                            "content": "Hello, how can I help you?"
                        }
                    ],
                    "prompt": "How do you fry an egg?",
                    "system_prompt": "You are a rapper. Always answer with a rap.",
                    "stream": False,
                    "use_context": False,
                    "include_sources": False,
                    "category_id": 1,
                }
            ]
        }
    }

class ChatContentCreate(BaseModel):
    content: Dict[str, Any]


class ChatResponse(BaseModel):
    id: uuid.UUID
    response: Union[OpenAICompletion, str]


async def get_latest_version_ids(
    service: IngestService,
    documents: List[Document]
) -> List[str]:
    """Get document IDs for the latest version of each filename.
    
    This function:
    1. Iterates through all documents
    2. For each document, finds all approved versions
    3. For each approved version, gets the Qdrant document IDs by filename
    """
    latest_doc_ids = []
    
    for doc in documents:
        approved_versions = [v for v in doc.versions if v.status == MakerCheckerStatus.APPROVED]
        
        if not approved_versions:
            continue
        latest_version = max(approved_versions, key=lambda v: v.uploaded_at)
        versioned_filename = os.path.basename(latest_version.file_path)
        logger.info(f"Extracting docs ids for file: {versioned_filename}")      
        exact_docs = service.get_doc_ids_by_filename(versioned_filename)
        latest_doc_ids.extend(exact_docs)
    
    return latest_doc_ids

def create_chat_item(db, sender, content, conversation_id):
    chat_item_create = schemas.ChatItemCreate(
            sender=sender,
            content=content,
            conversation_id=conversation_id
        )
    chat_history = crud.chat.get_conversation(db, conversation_id=conversation_id)
    if not chat_history.title or chat_history.title == "New Chat":
        chat_history.generate_title()
    return crud.chat_item.create(db, obj_in=chat_item_create)

@completions_router.post(
    "/chat",
    response_model=None,
    summary="Completion",
    responses={200: {"model": OpenAICompletion}},
    tags=["Contextual Completions"],
    openapi_extra={
        "x-fern-streaming": {
            "stream-condition": "stream",
            "response": {"$ref": "#/components/schemas/OpenAICompletion"},
            "response-stream": {"$ref": "#/components/schemas/OpenAICompletion"},
        }
    },
)

async def prompt_completion(
    request: Request,
    body: CompletionsBody,
    db: Session = Depends(deps.get_db),
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    current_user: models.User = Security(deps.get_current_user),
) -> ChatResponse | StreamingResponse:
    """Handle chat completion with proper document version filtering."""
    service = request.state.injector.get(IngestService)
    
    try:
        department = crud.department.get_by_id(db, id=current_user.department_id)
        if not department:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No department assigned to you"
            )

        documents = crud.documents.get_enabled_documents_by_departments(
            db,
            department_id=department.id,
            category_ids=body.category_id
        )
        if not documents:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No documents uploaded for your department. Please upload documents to chat."
            )
        latest_doc_ids = await get_latest_version_ids(service, documents)
        
        if not latest_doc_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Could not find any valid document versions"
            )

        body.context_filter = {"docs_ids": latest_doc_ids}

        chat_history = crud.chat.get_by_id(db, id=body.conversation_id)
        if (chat_history is None) or (chat_history.user_id != current_user.id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )

        def build_history() -> List[OpenAIMessage]:
            history_messages: List[OpenAIMessage] = []
            for interaction in (body.history or []):
                role = "user" if interaction.role == "user" else "assistant"
                history_messages.append(
                    OpenAIMessage(
                        content=interaction.content,
                        role=role
                    )
                )
            return history_messages

        user_message = OpenAIMessage(content=body.prompt, role="user")
        user_message_json = {"text": body.prompt}
        
        user_chat = create_chat_item(
            db,
            "user",
            user_message_json,
            body.conversation_id
        )
        print("USER MESSAGE: ",user_chat.id, user_chat.index)

        messages = [user_message]
        if body.system_prompt:
            messages.insert(
                0,
                OpenAIMessage(content=body.system_prompt, role="system")
            )

        chat_body = ChatBody(
            messages=[*build_history(), user_message],
            use_context=body.use_context,
            stream=body.stream,
            include_sources=body.include_sources,
            context_filter=body.context_filter,
        )
        log_audit(
            model='Chat',
            action='Chat',
            details={
                "query": body.prompt,
                'user': current_user.username,
                'document_versions': latest_doc_ids
            },
            user_id=current_user.id
        )
        chat_response = await chat_completion(request, chat_body)  
        if isinstance(chat_response, StreamingResponse):
            return chat_response 
        
        ai_response = chat_response.model_dump(mode="json")
        chat = create_chat_item(
            db,
            "assistant",
            ai_response,
            body.conversation_id
        )
        print("AI MESSAGE: ",chat.id, chat.index)
        response = ChatResponse(id=chat.id, response=chat_response)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in prompt completion: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat completion"
        )