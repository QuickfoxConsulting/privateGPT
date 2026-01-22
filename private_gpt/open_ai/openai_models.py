import time
import uuid
from collections.abc import Iterator
from typing import Literal, Optional, Any, Union, AsyncIterator

from llama_index.core.llms import ChatResponse, CompletionResponse
from pydantic import BaseModel, Field

from private_gpt.server.chunks.chunks_service import Chunk


class OpenAIDelta(BaseModel):
    """A piece of completion that needs to be concatenated to get the full message."""

    content: str | None = None
    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None


class OpenAIMessage(BaseModel):
    """Inference result, with the source of the message.

    Role could be the assistant or system
    (providing a default response, not AI generated).
    Optionally supports an image (base64 string, URL, or file path) for multimodal chat.
    """

    role: Literal["assistant", "system", "user"] = Field(default="user")
    content: str | None
    image: str | None = None  # base64 string, URL, or file path


class OpenAIChoice(BaseModel):
    """Response from AI.

    Either the delta or the message will be present, but never both.
    Sources used will be returned in case context retrieval was enabled.
    """

    finish_reason: str | None = Field(examples=["stop"])
    delta: OpenAIDelta | None = None
    message: OpenAIMessage | None = None
    sources: list[Chunk] | None = None
    cache_id: Optional[str] = None


class OpenAICompletion(BaseModel):
    """Clone of OpenAI Completion model.

    For more information see: https://platform.openai.com/docs/api-reference/chat/object
    """

    id: str
    object: Literal["completion", "completion.chunk"] = Field(default="completion")
    created: int = Field(..., examples=[1623340000])
    model: Literal["private-gpt"]
    choices: list[OpenAIChoice]

    @classmethod
    def from_text(
        cls,
        text: str | None,
        finish_reason: str | None = None,
        sources: list[Chunk] | None = None,
        cache_id: Optional[str] | None = None,
    ) -> "OpenAICompletion":
        return OpenAICompletion(
            id=str(uuid.uuid4()),
            object="completion",
            created=int(time.time()),
            model="private-gpt",
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                    sources=sources,
                    cache_id=cache_id,
                )
            ],
        )

    @classmethod
    def json_from_delta(
        cls,
        *,
        text: str | None = None,
        thought: str | None = None,
        action: str | None = None,
        action_input: str | None = None,
        observation: str | None = None,
        finish_reason: str | None = None,
        sources: list[Chunk] | None = None,
    ) -> str:
        chunk = OpenAICompletion(
            id=str(uuid.uuid4()),
            object="completion.chunk",
            created=int(time.time()),
            model="private-gpt",
            choices=[
                OpenAIChoice(
                    delta=OpenAIDelta(
                        content=text,
                        thought=thought,
                        action=action,
                        action_input=action_input,
                        observation=observation,
                    ),
                    finish_reason=finish_reason,
                    sources=sources,
                )
            ],
        )

        return chunk.model_dump_json()


def to_openai_response(
    response: str | ChatResponse, sources: list[Chunk] | None = None, cache_id: Optional[str] | None = None
) -> OpenAICompletion:
    if isinstance(response, ChatResponse):
        return OpenAICompletion.from_text(response.delta, finish_reason="stop", cache_id=cache_id)
    else:
        return OpenAICompletion.from_text(
            response, finish_reason="stop", sources=sources, cache_id=cache_id
        )


async def to_openai_sse_stream(
    response_generator: Union[AsyncIterator[Any], Iterator[Any]],
    sources: list[Chunk] | None = None,
) -> AsyncIterator[str]:
    """Convert response generator to OpenAI SSE stream format with error handling.
    
    Args:
        response_generator: Async or sync iterator yielding response chunks
        sources: Optional list of source chunks to include in the stream
        
    Yields:
        SSE-formatted strings in the format: "data: <json>\\n\\n"
    """
    import logging
    logger = logging.getLogger(__name__)
    
    first_chunk = True
    chunk_count = 0
    
    try:
        if hasattr(response_generator, "__aiter__"):
            logger.info("Starting async SSE stream")
            async for response in response_generator:
                chunk_count += 1
                
                if first_chunk:
                    logger.info(f"First chunk type: {type(response)}, content preview: {str(response)[:100]}")
                    first_chunk = False
                
                if isinstance(response, dict):
                    yield f"data: {OpenAICompletion.json_from_delta(**response)}\n\n"
                elif isinstance(response, (CompletionResponse, ChatResponse)):
                    yield f"data: {OpenAICompletion.json_from_delta(text=response.delta)}\n\n"
                else:
                    yield f"data: {OpenAICompletion.json_from_delta(text=response, sources=sources)}\n\n"
        else:
            logger.info("Starting sync SSE stream")
            for response in response_generator:
                chunk_count += 1
                
                if first_chunk:
                    logger.info(f"First chunk type: {type(response)}, content preview: {str(response)[:100]}")
                    first_chunk = False
                
                if isinstance(response, dict):
                    yield f"data: {OpenAICompletion.json_from_delta(**response)}\n\n"
                elif isinstance(response, (CompletionResponse, ChatResponse)):
                    yield f"data: {OpenAICompletion.json_from_delta(text=response.delta)}\n\n"
                else:
                    yield f"data: {OpenAICompletion.json_from_delta(text=response, sources=sources)}\n\n"
    except Exception as e:
        logger.error(f"Error in SSE stream after {chunk_count} chunks: {e}", exc_info=True)
        # Send error event to client
        yield f"data: {OpenAICompletion.json_from_delta(text='', finish_reason='error')}\n\n"
    finally:
        # Always send completion events
        logger.info(f"SSE stream completed with {chunk_count} chunks")
        yield f"data: {OpenAICompletion.json_from_delta(text='', finish_reason='stop')}\n\n"
        yield "data: [DONE]\n\n"
