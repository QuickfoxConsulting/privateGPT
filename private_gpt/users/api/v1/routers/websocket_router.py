import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect, Depends, Security
from fastapi.security import HTTPBearer
from jose import jwt
from sqlalchemy.orm import Session

from private_gpt.manager.websocket_manager import manager
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.users.core import security
from private_gpt.users.api import deps

logger = logging.getLogger(__name__)

websocket_router = APIRouter()
security = HTTPBearer()

async def get_current_user(token: Optional[str], db: Session):
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=["HS256"],
            options={"verify_signature": True}  
        )        
        user_id: str = payload.get("id")
        if user_id is None:
            return None
        
        user = crud.user.get(db, id=user_id)
        if not user:
            return None

        # Update last login
        user_in = schemas.UserUpdate(last_login=datetime.now())
        crud.user.update(db, db_obj=user, obj_in=user_in)
        
        return user

    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError as e:
        logger.error(f"JWT validation error: {str(e)}")
        return None

@websocket_router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    db: Session = Depends(deps.get_db),
):
    print("GETTING Params: {websocket.query_params}")
    token = websocket.query_params.get("token")
    if not token: 
        await websocket.close(code=4001, reason="Missing authentication token")
        return

    try:
        await websocket.accept()
        current_user = await get_current_user(token, db)
        print(f"CURRENTUSER: {current_user}")
        if not current_user:
            await websocket.close(code=4001, reason="Invalid authentication")
            return

        await manager.connect(websocket, str(current_user.id), db)  

        try:
            while True:
                try:
                    data = await websocket.receive_json()
                    
                    match data.get("type"):
                        case "heartbeat":
                            await manager.handle_heartbeat(str(current_user.id), db)  
                        
                        case "update_status":
                            if "payload" in data and "status" in data["payload"]:
                                await manager.update_user_status(
                                    str(current_user.id), 
                                    data["payload"]["status"],
                                    db  
                                )
                            else:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "Invalid status update format"
                                })
                        
                        case "get_user_status":
                            await websocket.send_json({
                                "type": "user_status",
                                "payload": manager.user_status
                            })
                        
                        case "user_logout":
                            await manager.disconnect(str(current_user.id), db)  
                            await websocket.close(code=1000)
                            return
                            
                        case _:
                            await websocket.send_json({
                                "type": "error",
                                "message": "Unknown message type"
                            })

                except ValueError as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid JSON format"
                    })

        except WebSocketDisconnect:
            logger.info(f"Client disconnected: {current_user.id}")
        finally:
            await manager.disconnect(str(current_user.id), db)  # Pass db session

    except Exception as e:
        logger.error(f"WebSocket error for user {current_user.id if current_user else 'unknown'}: {str(e)}")
        await websocket.close(code=1011)  # Internal error