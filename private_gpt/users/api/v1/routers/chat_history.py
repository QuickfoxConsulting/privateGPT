# import logging
# import traceback
# import uuid
# from private_gpt.server.chat.chat_service import ChatService
# from sqlalchemy.orm import Session
# from fastapi.responses import JSONResponse
# from fastapi import APIRouter, Depends, HTTPException, Request, status, Security
# from fastapi_pagination import Page, paginate

# from private_gpt.users.api import deps
# from private_gpt.users import crud, models, schemas

# logger = logging.getLogger(__name__)
# router = APIRouter(prefix="/c", tags=["Chat Histories"])


# @router.get("", response_model=Page[schemas.Chat])
# def list_chat_histories(
#     db: Session = Depends(deps.get_db),
#     current_user: models.User = Security(
#         deps.get_current_user,
#     ),
# ) -> Page[schemas.Chat]:
#     """
#     Retrieve a list of chat histories with pagination support.
#     """
#     try:
#         chat_histories = crud.chat.get_chat_history(
#             db, user_id=current_user.id)
#         return paginate(chat_histories)
#     except Exception as e:
#         print(traceback.format_exc())
#         logger.error(f"Error listing chat histories: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Internal Server Error",
#         )


# @router.post("/create", response_model=schemas.ChatHistory)
# def create_chat_history(
#     db: Session = Depends(deps.get_db),
#     current_user: models.User = Security(
#         deps.get_current_user,
#     ),
# ) -> schemas.ChatHistory:
#     """
#     Create a new chat history
#     """
#     try:
#         chat_history_in = schemas.CreateChatHistory(
#             user_id= current_user.id
#         )
#         chat_history = crud.chat.create(
#             db=db, obj_in=chat_history_in)
#         return chat_history
#     except Exception as e:
#         print(traceback.format_exc())
#         logger.error(f"Error creating chat history: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Internal Server Error",
#         )


# @router.get("/{conversation_id}", response_model=schemas.ChatHistory)
# def read_chat_history(
#     conversation_id: uuid.UUID,
#     skip: int = 0,
#     limit: int = 20,
#     db: Session = Depends(deps.get_db),
#     current_user: models.User = Security(
#         deps.get_current_user,
#     ),
# ) -> schemas.ChatHistory:
#     """
#     Read a chat history by ID
#     """
#     try:
#         chat_history = crud.chat.get_by_id(db, id=conversation_id, skip=skip, limit=limit)
#         if chat_history is None or chat_history.user_id != current_user.id:
#             raise HTTPException(
#                 status_code=404, detail="Chat history not found")
#         return chat_history
#     except Exception as e:
#         print(traceback.format_exc())
#         logger.error(f"Error reading chat history: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Internal Server Error",
#         )


# @router.post("/delete")
# def delete_chat_history(
#     chat_history_in: schemas.ChatDelete,
#     db: Session = Depends(deps.get_db),
#     current_user: models.User = Security(
#         deps.get_current_user,
#     ),
# ):
#     """
#     Delete a chat history by ID
#     """
#     try:
#         chat_history_id = chat_history_in.conversation_id
#         chat_history = crud.chat.get_by_id(db, id=chat_history_id)
#         if chat_history is None or chat_history.user_id != current_user.id:
#             raise HTTPException(
#                 status_code=404, detail="Chat history not found")

#         crud.chat.remove(db=db, id=chat_history_id)
#         return JSONResponse(
#             status_code=status.HTTP_200_OK,
#             content={
#                 "message": "Chat history deleted successfully",
#             },
#         )
#     except Exception as e:
#         print(traceback.format_exc())
#         logger.error(f"Error deleting chat history: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Internal Server Error",
#         )


# @router.get("/{conversation_id}/title")
# async def create_chat_history_title(
#     request: Request,
#     conversation_id: uuid.UUID,
#     db: Session = Depends(deps.get_db),
#     current_user: models.User = Security(
#         deps.get_current_user,
#     ),
# ) -> schemas.ChatHistory:
#     """
#     Create a title for a chat history by ID
#     """
#     service = request.state.injector.get(ChatService)
#     try:
#         chat_history = crud.chat.get_by_id(db, id=conversation_id)
#         if chat_history is None or chat_history.user_id != current_user.id:
#             raise HTTPException(
#                 status_code=404, detail="Chat history not found")
        
#         first_user_chat_item = [item for item in chat_history.chat_items if item.sender == "user"][0]

#         logger.info(f"Chat items: {first_user_chat_item.content}")
#         title = await service.generate_title([first_user_chat_item])
#         logger.info(f"Title: {title}")
#         chat_history.title = title.title
#         db.commit()
#         db.refresh(chat_history)
#         return chat_history
#     except Exception as e:
#         print(traceback.format_exc())
#         logger.error(f"Error getting chat history title: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail="Internal Server Error",
#         )



import logging
import traceback
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status, Security, Query
from fastapi.responses import JSONResponse
from fastapi_pagination import Page, paginate
from sqlalchemy.orm import Session

from private_gpt.server.chat.chat_service import ChatService
from private_gpt.users.api import deps
from private_gpt.users import crud, models, schemas
from private_gpt.users.models.chat import Rating, MessageStatus

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/c", tags=["Chat Histories"])


@router.get("", response_model=Page[schemas.Chat])
def list_chat_histories(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    include_archived: bool = Query(False, description="Include archived chats"),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> Page[schemas.Chat]:
    """
    Retrieve a list of chat histories with pagination support.
    """
    try:
        chat_histories = crud.chat.get_chat_history(
            db, 
            user_id=current_user.id,
            skip=skip,
            limit=limit,
            include_archived=include_archived
        )
        return paginate(chat_histories)
    except Exception as e:
        logger.error(f"Error listing chat histories: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post("/create", response_model=schemas.ChatHistory)
def create_chat_history(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> schemas.ChatHistory:
    """
    Create a new chat history with optional initial title
    """
    try:
        chat_history_in = schemas.CreateChatHistory(
            user_id= current_user.id
        )
        chat_history = crud.chat.create(
            db=db, obj_in=chat_history_in)
        return chat_history
    except Exception as e:
        logger.error(f"Error creating chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get("/{conversation_id}", response_model=schemas.ChatHistory)
def read_chat_history(
    conversation_id: uuid.UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> schemas.ChatHistory:
    """
    Read a chat history by ID with paginated chat items
    """
    try:
        chat_history = crud.chat.get_by_id(db, id=conversation_id, skip=skip, limit=limit)
        if chat_history is None or chat_history.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Chat history not found"
            )
        return chat_history
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post("/delete")
def delete_chat_history(
    chat_history_in: schemas.ChatDelete,
    permanent: bool = Query(False, description="Permanently delete chat history"),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
):
    """
    Delete a chat history by ID (soft delete by default)
    """
    try:
        chat_history_id = chat_history_in.conversation_id
        chat_history = crud.chat.get_by_id(db, id=chat_history_id)
        
        if chat_history is None or chat_history.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Chat history not found"
            )

        if permanent:
            crud.chat.remove(db=db, id=chat_history_id)
            message = "Chat history permanently deleted"
        else:
            # Soft delete
            success = crud.chat.soft_delete(db=db, id=chat_history_id, user_id=current_user.id)
            if not success:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, 
                    detail="Chat history not found"
                )
            message = "Chat history deleted"
            
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": message,
                "conversation_id": str(chat_history_id)
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post("/archive/{conversation_id}")
def archive_chat_history(
    conversation_id: uuid.UUID,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
):
    """
    Archive a chat history by ID
    """
    try:
        chat_history = crud.chat.archive_chat(
            db=db, 
            id=conversation_id, 
            user_id=current_user.id
        )
        
        if chat_history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Chat history not found"
            )
            
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Chat history archived successfully",
                "conversation_id": str(conversation_id)
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error archiving chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post("/unarchive/{conversation_id}")
def unarchive_chat_history(
    conversation_id: uuid.UUID,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
):
    """
    Unarchive a chat history by ID
    """
    try:
        chat_history = crud.chat.unarchive_chat(
            db=db, 
            id=conversation_id, 
            user_id=current_user.id
        )
        
        if chat_history is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Chat history not found"
            )
            
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Chat history unarchived successfully",
                "conversation_id": str(conversation_id)
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unarchiving chat history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.get("/{conversation_id}/title")
async def create_chat_history_title(
    request: Request,
    conversation_id: uuid.UUID,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> schemas.ChatHistory:
    """
    Create a title for a chat history by ID using AI
    """
    service = request.state.injector.get(ChatService)
    try:
        chat_history = crud.chat.get_by_id(db, id=conversation_id)
        if chat_history is None or chat_history.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Chat history not found"
            )
        
        user_chat_items = [item for item in chat_history.chat_items if item.sender == "user"]
        
        if not user_chat_items:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No user messages found to generate title"
            )
        
        title = await service.generate_title([user_chat_items[0]])        
        chat_history.title = title.title
        chat_history.title_generated = True
        db.commit()
        db.refresh(chat_history)
        return chat_history
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chat history title: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post("/{conversation_id}/messages", response_model=schemas.ChatItem)
def add_chat_message(
    conversation_id: uuid.UUID,
    message: schemas.ChatItemCreate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> schemas.ChatItem:
    """
    Add a new message to a chat history
    """
    try:
        chat_history = crud.chat.get_conversation(db, conversation_id=conversation_id)
        if chat_history is None or chat_history.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Chat history not found"
            )        
        chat_item = crud.chat_item.create_with_chat(
            db=db,
            obj_in=message,
            conversation_id=conversation_id
        )
        return chat_item
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Error adding chat message: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error adding chat message: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )


@router.post("/messages/{message_id}/rate")
def rate_chat_message(
    message_id: uuid.UUID,
    rating_data: schemas.ChatItemRating,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
):
    """
    Rate a chat message
    """
    try:
        chat_item = crud.chat_item.set_rating(
            db=db,
            id=message_id,
            rating=rating_data.rating,
            user_id=current_user.id
        )
        
        if chat_item is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="Chat message not found"
            )
            
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Chat message rated successfully",
                "message_id": str(message_id),
                "rating": rating_data.rating.value
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rating chat message: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal Server Error",
        )

