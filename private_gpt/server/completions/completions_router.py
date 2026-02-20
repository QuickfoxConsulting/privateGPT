import os
import json
import uuid
import logging
import traceback
import itertools
import time
import re
from pathlib import Path


from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Union

from starlette.responses import StreamingResponse

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
)

from private_gpt.server.cache.faq_service import FAQService
from private_gpt.server.cache.cache_service import CacheService
from private_gpt.server.integrations.website_crawl_service import WebsiteCrawlService
from private_gpt.server.integrations.google_drive_service import GoogleDriveService
from private_gpt.users.schemas.faq import FAQCreate
from private_gpt.users.db.session import SessionLocal
from private_gpt.users.utils.audit import log_audit_entry
from private_gpt.utils.chat_enums import ChatMode
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.users import crud, models, schemas
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from fastapi import APIRouter, Depends, Request, Security, HTTPException, status
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.users.services import DocumentSelectionService
from private_gpt.settings.settings import settings
from private_gpt.users.models.document import Document
from private_gpt.users.models.enums import MakerCheckerStatus



from private_gpt.users.api import deps
from private_gpt.users import crud, models, schemas
from private_gpt.server.utils.auth import authenticated
from private_gpt.server.chat.chat_router import ChatBody, chat_completion


logger = logging.getLogger(__name__)
completions_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class CompletionsBody(BaseModel):
    conversation_id: uuid.UUID
    history: Optional[list[OpenAIMessage]]
    prompt: str
    system_prompt: str | None = None
    use_context: str = ChatMode.CHAT.value
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
                    "use_context": "chat",
                    "include_sources": False,
                    "category_id": 1,
                }
            ]
        }
    }

REFUSAL_PATTERNS = [
    r"following question detail is not provided in docs",
    r"provided documents do not contain information",
    r"cannot find information about this in the provided documents",
    r"context does not contain",
    r"answer is not in the context provided"
]

def check_is_answered(response_text: str, source_count: int, mode: str) -> bool:
    """
    Smart logic to determine if a question was answered.
    Greetings (0 sources) are considered 'Answered'.
    RAG Refusals (0 sources + Refusal Pattern) are 'Unanswered'.
    """
    # If sources were found, it's always answered
    if source_count > 0:
        return True
    
    # If not in a retrieval mode, 0 sources is normal (it's a chat)
    # RAG, Search, and Agentic are retrieval-based
    if mode.upper() not in ["RAG", "SEARCH", "AGENTIC"]:
        return True
        
    # Check for refusal patterns
    is_refusal = any(re.search(p, response_text, re.IGNORECASE) for p in REFUSAL_PATTERNS)
    return not is_refusal

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

def create_chat_item(db: Session, sender: str, content: dict, conversation_id: uuid.UUID) -> models.ChatItem:
    chat_item_create = schemas.ChatItemCreate( 
            sender=sender,
            content=content,
            conversation_id=conversation_id
        )
    chat_history = crud.chat.get_by_id(db, id=conversation_id)

    if not chat_history:
        raise ValueError(f"Chat history with ID {conversation_id} not found.")
    if not chat_history.title_generated:
        chat_history.generate_title()
    new_chat_item = crud.chat_item.create(db, obj_in=chat_item_create)
    return new_chat_item

async def save_response_to_db(full_response, unique_sources, conversation_id, original_prompt, current_user):
    """Separate function to handle database persistence."""
    if not full_response and not unique_sources:
        logger.warning(f"No response content to save for conversation {conversation_id}")
        return
        
    db_session = SessionLocal()
    try:
        final_sources = list(unique_sources.values())
        
        ai_response_json = {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "private-gpt",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": full_response
                },
                "finish_reason": "stop",
                "sources": final_sources
            }]
        }
        
        create_chat_item(db_session, "assistant", ai_response_json, conversation_id)
        
        log_audit_entry(
            db_session,
            model="Chat",
            action="chat_completion_success" if len(final_sources) > 0 else "chat_completion_unanswered",
            details={
                "query": original_prompt,
                "user": current_user.username,
                "response_length": len(full_response),
                "source_count": len(final_sources),
                "is_answered": len(final_sources) > 0,
                "is_streamed": True
            },
            user_id=current_user.id,
            username=current_user.username,
            resource_id=str(conversation_id),
            severity="INFO"
        )
        
        db_session.commit()
        logger.info(f"Successfully saved streamed response to DB for conversation {conversation_id}")
        
    except Exception as e:
        logger.error(f"Failed to save streamed response to DB: {e}", exc_info=True)
        db_session.rollback()
    finally:
        db_session.close()

async def stream_with_persistence(response_gen, conversation_id, original_prompt, current_user):
    """Wrap the SSE stream to persist the full response after streaming.
    
    Args:
        response_gen: Iterator yielding SSE-formatted strings/bytes
        conversation_id: UUID of the conversation
        original_prompt: Original user prompt
        current_user: Current user object
    """
    full_response = ""
    unique_sources = {}
    
    async def process_and_yield():
        nonlocal full_response, unique_sources
        
        try:
            # Handle standard async iterator
            if hasattr(response_gen, "__aiter__"):
                async for chunk in response_gen:
                    yield chunk
                    try:
                        line = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                continue
                            data = json.loads(data_str)
                            
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    full_response += delta["content"]
                                
                                chunk_sources = data["choices"][0].get("sources")
                                if chunk_sources and isinstance(chunk_sources, list):
                                    for s in chunk_sources:
                                        doc_meta = s.get('document', {}).get('doc_metadata', {})
                                        file_name = doc_meta.get('file_name', 'unknown')
                                        page = doc_meta.get('page_label', '') or doc_meta.get('page', '')
                                        source_key = f"{file_name}_{page}_{hash(s.get('text', ''))}"
                                        
                                        if source_key not in unique_sources:
                                            unique_sources[source_key] = s
                    except Exception as e:
                        logger.warning(f"Failed to process SSE chunk for persistence: {e}")
                        
            # Handle sync iterator fallback (though FastAPI StreamingResponse usually wraps this)
            else:
                for chunk in response_gen:
                    yield chunk
                    try:
                        line = chunk.decode("utf-8") if isinstance(chunk, bytes) else str(chunk)
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                continue
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    full_response += delta["content"]
                                chunk_sources = data["choices"][0].get("sources")
                                if chunk_sources and isinstance(chunk_sources, list):
                                    for s in chunk_sources:
                                        doc_meta = s.get('document', {}).get('doc_metadata', {})
                                        file_name = doc_meta.get('file_name', 'unknown')
                                        source_key = f"{file_name}_{hash(s.get('text', ''))}"
                                        if source_key not in unique_sources:
                                            unique_sources[source_key] = s
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.error(f"Error during stream processing: {e}", exc_info=True)
            
            # Log failure for the stream
            log_audit_entry(
                SessionLocal(), # New session for immediate log
                model="Chat",
                action="chat_completion_failure",
                details={
                    "query": original_prompt,
                    "user": current_user.username,
                    "error": str(e),
                    "is_streamed": True
                },
                user_id=current_user.id,
                username=current_user.username,
                resource_id=str(conversation_id),
                severity="ERROR"
            )
            raise
        finally:
            await save_response_to_db(
                full_response, 
                unique_sources, 
                conversation_id, 
                original_prompt, 
                current_user
            )
            
    return process_and_yield()

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
    """Handle chat completion with intelligent context handling and fallbacks."""
    try:
        # doc_service = DocumentSelectionService(db)
        original_prompt = body.prompt
        original_use_context = body.use_context
        document_status = "not_requested"
        department = None
        latest_doc_ids = []
        
        # Log the chat completion attempt
        log_audit(
            model="Chat",
            action="chat_completion_attempt",
            details={
                "query": original_prompt,
                "user": current_user.username,
                "use_context": original_use_context,
                "stream": body.stream,
                "conversation_id": str(body.conversation_id),
                "category_id": body.category_id
            },
            user_id=current_user.id,
            username=current_user.username,
            resource_id=str(body.conversation_id),
            severity="INFO"
        )
        
        chat_history = crud.chat.get_by_id(db, id=body.conversation_id)
        if (chat_history is None) or (chat_history.user_id != current_user.id):
            # Log unauthorized access attempt
            log_audit(
                model="Chat",
                action="unauthorized_access",
                details={
                    "query": original_prompt,
                    "user": current_user.username,
                    "attempted_conversation_id": body.conversation_id,
                    "reason": "Chat not found or unauthorized access"
                },
                user_id=current_user.id,
                username=current_user.username,
                resource_id=str(body.conversation_id),
                severity="WARNING"
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat not found"
            )
        logger.info(f"Chat Mode: {body.use_context}")
        is_using_context = body.use_context != ChatMode.CHAT.value        
        if is_using_context:
            service = request.state.injector.get(IngestService)
            document_status = "requested"
            
            department = crud.department.get_by_id(db, id=current_user.department_id)
            if not department:
                # Log missing department
                log_audit(
                    model="Chat",
                    action="missing_department",
                    details={
                        "query": original_prompt,
                        "user": current_user.username,
                        "user_id": current_user.id,
                        "reason": "No department assigned to user"
                    },
                    user_id=current_user.id,
                    username=current_user.username,
                    severity="WARNING"
                )
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No department assigned to you"
                )
                
            documents = crud.documents.get_enabled_documents_by_departments(
                db,
                department_id=department.id,
                category_ids=body.category_id
            )
            # documents = doc_service.get_enabled_documents(user_id=current_user.id)
            file_list = [doc.filename for doc in documents]
            if not documents:
                document_status = "no_documents"
                is_using_context = False
                body.use_context = ChatMode.SEARCH.value
                body.system_prompt = (body.system_prompt or "") + "\n\nIMPORTANT: No documents are available for this user's department. "            
                body.context_filter = None
                logger.warning(f"No documents found for department {department.id}")
                
                # Log no documents available
                log_audit(
                    model="Chat",
                    action="no_documents_available",
                    details={
                        "query": original_prompt,
                        "user": current_user.username,
                        "department_id": department.id,
                        "category_id": body.category_id
                    },
                    user_id=current_user.id,
                    username=current_user.username,
                    resource_id=str(body.conversation_id),
                    severity="INFO"
                )
            else:
                latest_doc_ids = await get_latest_version_ids(service, documents)
            
            # Add website content
            try:
                website_service = WebsiteCrawlService(db)
                # Pass department_id to filter by department
                ingested_pages = website_service.get_ingested_pages(department.id)
                for page in ingested_pages:
                     # Use URL as filename to get doc IDs
                    page_doc_ids = service.get_doc_ids_by_filename(page.url)
                    latest_doc_ids.extend(page_doc_ids)
                
                if ingested_pages:
                    logger.info(f"Added nodes from {len(ingested_pages)} website pages")
            except Exception as e:
                logger.error(f"Failed to add website content: {e}")

            # Add Google Drive content
            try:
                drive_service = GoogleDriveService(db)
                ingested_drive_files = drive_service.get_ingested_files(department.id)
                for file in ingested_drive_files:
                    # Use unique doc_id stored in metadata to avoid filename collisions
                    unique_doc_id = f"google_drive_{file.file_id}"
                    drive_doc_ids = service.get_doc_ids_by_metadata("doc_id", unique_doc_id)
                    latest_doc_ids.extend(drive_doc_ids)
                
                if ingested_drive_files:
                    logger.info(f"Added nodes from {len(ingested_drive_files)} Google Drive files")
            except Exception as e:
                logger.error(f"Failed to add Google Drive content: {e}")

            if not latest_doc_ids:
                document_status = "no_valid_versions"
                # Do NOT force fallback to CHAT mode.
                # Instead, set empty context filter so strict RAG returns "I don't know"
                # body.use_context = ChatMode.CHAT.value  <-- REMOVED
                
                body.context_filter = ContextFilter(docs_ids=[])
                
                logger.warning(f"No valid document/website versions found")
                
                # Log no valid versions
                log_audit(
                    model="Chat",
                    action="no_valid_docs",
                    details={
                        "query": original_prompt,
                        "user": current_user.username,
                        "department_id": department.id,
                        "reason": "no_valid_document_versions",
                        "source_count": 0,
                        "is_answered": False
                    },
                    user_id=current_user.id,
                    username=current_user.username,
                    resource_id=str(body.conversation_id),
                    severity="WARNING"
                )
            else:
                document_status = "available"
                body.context_filter = ContextFilter(docs_ids=latest_doc_ids)
                logger.info(f"Found {len(latest_doc_ids)} valid document nodes/versions")

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

        user_message = OpenAIMessage(content=original_prompt, role="user")
        user_message_json = {"text": original_prompt}
        
        user_chat = create_chat_item(
            db,
            "user",
            user_message_json,
            body.conversation_id
        )
        
        messages = [user_message]
        if body.system_prompt:
            messages.insert(0, OpenAIMessage(content=body.system_prompt, role="system"))

        chat_body = ChatBody(
            messages=[*build_history(), user_message],
            use_context=body.use_context,
            stream=body.stream,
            include_sources=body.include_sources if is_using_context else False,
            context_filter=body.context_filter,
        )        
        
        audit_details = {
            "query": original_prompt,
            "user": current_user.username,
            "context_requested": original_use_context,
            "context_used": body.use_context,
            "document_status": document_status,
        }
        
        if department:
            audit_details["department_id"] = department.id
        
        # if body.context_filter and "docs_ids" in body.context_filter:
        #     audit_details["document_versions"] = body.context_filter["docs_ids"]
        
        log_audit(
            model="Chat",
            action="Chat",
            details=audit_details,
            user_id=current_user.id,
            username=current_user.username,
            resource_id=str(body.conversation_id),
            severity="INFO"
        )        
        chat_response = await chat_completion(
            request=request,
            body=chat_body,
            file_list=file_list,
            db=db,
            token_user=current_user
        )
        
        if isinstance(chat_response, StreamingResponse):
            # Wrap the iterator to persist it when done
            # We await stream_with_persistence to get the generator
            stream_gen = await stream_with_persistence(
                chat_response.body_iterator,
                body.conversation_id,
                original_prompt,
                current_user
            )
            return StreamingResponse(
                stream_gen,
                media_type="text/event-stream"
            )
        
        ai_response = chat_response.model_dump(mode="json")
        chat = create_chat_item(
            db,
            "assistant",
            ai_response,
            body.conversation_id
        )
        # Prepare data for storage and audit
        choice = ai_response["choices"][0] if ai_response.get("choices") else None
        message = choice.get("message") if choice else None
        response_content = message.get("content") if message else ""
        response_sources = (choice.get("sources") or []) if choice else []

        cache_response = {
            "content": response_content,
            "sources": response_sources
        }
            
        if settings().faq.enabled:
            try:
                # Get FAQ service once
                faq_service = request.state.injector.get(FAQService)            
                cache_id = ai_response["choices"][0].get("cache_id")
                if cache_id:
                    # Convert cache_id string back to integer for database operations
                    try:
                        cache_id_int = int(cache_id)
                        success = faq_service.increment_cache_hit_frequency(db, cache_id_int)
                        if success:
                            logger.info(f"Successfully incremented frequency for cached FAQ {cache_id}")
                        else:
                            logger.warning(f"Failed to increment frequency for cached FAQ {cache_id}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Invalid cache_id format '{cache_id}': {e}")
                else:
                    # This is a new interaction - optionally create FAQ for future caching
                    # Only create FAQ if it meets certain criteria (e.g., not already exists)
                    cache_in = FAQCreate(
                        question=original_prompt,
                        answer=cache_response,
                        category='cache'
                    )
                    new_faq = faq_service.create_faq(db, cache_in, current_user.id)
                    logger.info(f"Created new FAQ entry {new_faq.id} for potential future caching")
            except Exception as e:
                logger.error(f"Error handling FAQ cache operations: {e}", exc_info=True)
    
        response = ChatResponse(id=chat.id, response=chat_response)
        
        # Determine if it was actually answered or a refusal
        is_actually_answered = check_is_answered(
            response_content,
            len(response_sources),
            body.use_context
        )
        
        # Log outcome
        log_audit(
            model="Chat",
            action="chat_completion_success" if is_actually_answered else "chat_completion_unanswered",
            details={
                "query": original_prompt,
                "user": current_user.username,
                "response_length": len(response_content),
                "document_status": document_status,
                "source_count": len(response_sources),
                "is_answered": is_actually_answered
            },
            user_id=current_user.id,
            username=current_user.username,
            resource_id=str(body.conversation_id),
            severity="INFO"
        )
        
        return response

    except HTTPException as e:
        # Log completion failure for known HTTP errors (like 404 Chat Not Found)
        log_audit(
            model="Chat",
            action="chat_completion_failure",
            details={
                "query": body.prompt if 'body' in locals() and hasattr(body, 'prompt') else "Unknown",
                "user": current_user.username if 'current_user' in locals() else "Unknown",
                "status_code": e.status_code,
                "error": e.detail
            },
            user_id=current_user.id if 'current_user' in locals() else None,
            username=current_user.username if 'current_user' in locals() else None,
            resource_id=str(body.conversation_id) if 'body' in locals() and hasattr(body, 'conversation_id') else None,
            severity="ERROR"
        )
        raise
    except Exception as e:
        logger.error(f"Error in prompt completion: {str(e)}\n{traceback.format_exc()}")
        
        # Log completion failure
        log_audit(
            model="Chat",
            action="chat_completion_failure",
            details={
                "query": body.prompt if 'body' in locals() and hasattr(body, 'prompt') else "Unknown",
                "user": current_user.username if 'current_user' in locals() else "Unknown",
                "error": str(e)
            },
            user_id=current_user.id if 'current_user' in locals() else None,
            username=current_user.username if 'current_user' in locals() else None,
            resource_id=str(body.conversation_id) if 'body' in locals() and hasattr(body, 'conversation_id') else None,
            severity="ERROR"
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat completion"
        )
