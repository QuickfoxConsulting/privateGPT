from functools import cache
from typing import Dict, List, Any
from fastapi import APIRouter, Depends, Request, HTTPException, Security
from sqlalchemy.orm import Session
from private_gpt.users.api import deps
from private_gpt.users import models
from llama_index.core.llms import ChatMessage, MessageRole
from pydantic import BaseModel, field_validator
from starlette.responses import StreamingResponse

from private_gpt.server.cache.cache_service import CacheService
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.utils.chat_enums import ChatMode
from private_gpt.open_ai.openai_models import (
    OpenAICompletion,
    OpenAIMessage,
    to_openai_response,
    to_openai_sse_stream,
)
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.cache.faq_service import FAQService

from private_gpt.server.utils.auth import authenticated

chat_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class ChatBody(BaseModel):
    messages: list[OpenAIMessage]
    use_context: str = ChatMode.CHAT.value
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    stream: bool = False

    @field_validator('messages')
    @classmethod
    def validate_messages(cls, v):
        """Validate messages list for proper format and constraints"""
        if not v:
            raise ValueError("Messages list cannot be empty")
        if len(v) > 100:
            raise ValueError("Too many messages (maximum 100 allowed)")
        
        # Check for at least one user message
        has_user_message = any(msg.role == "user" for msg in v)
        if not has_user_message:
            raise ValueError("At least one user message is required")
        
        return v
    
    @field_validator('use_context')
    @classmethod
    def validate_use_context(cls, v):
        """Validate chat mode is one of the supported modes"""
        valid_modes = [ChatMode.CHAT.value, ChatMode.SEARCH.value, ChatMode.AGENTIC.value]
        if v not in valid_modes:
            raise ValueError(
                f"Invalid chat mode: '{v}'. Must be one of: {', '.join(valid_modes)}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a rapper. Always answer with a rap.",
                            "image": None
                        },
                        {
                            "role": "user",
                            "content": "How do you fry an egg?",
                            "image": "<base64-image-string>"
                        },
                    ],
                    "stream": False,
                    "use_context": "search",
                    "include_sources": True,
                    "context_filter": {
                        "docs_ids": ["c202d5e6-7b69-4869-81cc-dd574ee8ee11"]
                    },
                }
            ]
        }
    }


@chat_router.post(
    "/chat/completions",
    response_model=None,
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
async def chat_completion(
    request: Request, body: ChatBody, file_list: List[str],
    current_user: Any = Depends(authenticated), # Keep existing auth check too?
    # Actually, let's try to get the user using deps. But we need to handle if auth is disabled?
    # The authenticated dependency in this file return True or raises 401.
    # We should add the deps.get_current_user. 
    # But wait, `authenticated` is a dependency on the router.
    # Let's add db and try to get user.
    db: Session = Depends(deps.get_db),
    # We use a looser dependency or try-except block if we want to support both.
    # However, existing codebase has `users` module, so let's enforce user presence for now or make it optional.
    # To be safe and stick to the plan: "Prompts should be associated with specific users."
    # We will try to get the user.
    token_user: models.User = Security(deps.get_current_user, scopes=[]),
) -> OpenAICompletion | StreamingResponse:
    """Given a list of messages comprising a conversation, return a response.

    Optionally include an initial `role: system` message to influence the way
    the LLM answers.

    If `use_context` is set to `true`, the model will use context coming
    from the ingested documents to create the response. The documents being used can
    be filtered using the `context_filter` and passing the document IDs to be used.
    Ingested documents IDs can be found using `/ingest/list` endpoint. If you want
    all ingested documents to be used, remove `context_filter` altogether.

    When using `'include_sources': true`, the API will return the source Chunks used
    to create the response, which come from the context provided.

    When using `'stream': true`, the API will return data chunks following [OpenAI's
    streaming model](https://platform.openai.com/docs/api-reference/chat/streaming):
    ```
    {"id":"12345","object":"completion.chunk","created":1694268190,
    "model":"private-gpt","choices":[{"index":0,"delta":{"content":"Hello"},
    "finish_reason":null}]}
    ```
    """
    service = request.state.injector.get(ChatService)
    cache_service = request.state.injector.get(CacheService)

    all_messages = [
        ChatMessage(content=m.content, role=MessageRole(m.role)) for m in body.messages
    ]
    if body.stream:
        completion_gen = await service.stream_chat(
            messages=all_messages,
            use_context=body.use_context,
            context_filter=body.context_filter,
            file_list=file_list,
            cache_service=cache_service,
            user_id=token_user.id,
            db=db,
        )
        return StreamingResponse(
            to_openai_sse_stream(
                completion_gen.response,
                completion_gen.sources if body.include_sources else None,
            ),
            media_type="text/event-stream",
        )
    else:
        completion = await service.chat(
            messages=all_messages,
            use_context=body.use_context,
            context_filter=body.context_filter,
            file_list=file_list,
            cache_service=cache_service,
            user_id=token_user.id,
            db=db,
        )
        return to_openai_response(
            completion.response, completion.sources if body.include_sources else None, completion.cache_id
        )