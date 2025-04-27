from typing import Dict, Set
from fastapi import WebSocket
import json
import asyncio
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session

from private_gpt.users import crud, models, schemas

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_status: Dict[str, Dict] = {}
        self.heartbeat_tasks: Dict[str, asyncio.Task] = {}
        self.user_info: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, user_id: str, db: Session):
        self.active_connections[user_id] = websocket
        
        user = crud.user.get(db, id=user_id)
        if user:
            self.user_info[user_id] = {
                "id": str(user.id),
                "email": user.email,
                "username": user.username,
                "role": user.user_role.role.name if user.user_role else None,
                "company_id": user.user_role.company_id if user.user_role else None,
                "department_id": user.department_id
            }
            
            self.user_status[user_id] = {
                "status": "online",
                "last_seen": datetime.now().isoformat(),
                "user_info": self.user_info[user_id]
            }            
            self.heartbeat_tasks[user_id] = asyncio.create_task(self._heartbeat_check(user_id, db))
            await self.broadcast_user_status()

    async def disconnect(self, user_id: str, db: Session = None):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
            if user_id in self.heartbeat_tasks:
                self.heartbeat_tasks[user_id].cancel()
                del self.heartbeat_tasks[user_id]
            
            if db:
                user = crud.user.get(db, id=user_id)
                if user:
                    user_in = schemas.UserUpdate(last_login=datetime.now())
                    crud.user.update(db, db_obj=user, obj_in=user_in)
            
            self.user_status[user_id] = {
                "status": "offline",
                "last_seen": datetime.now().isoformat(),
                "user_info": self.user_info.get(user_id, {})
            }
            await self.broadcast_user_status()

    async def update_user_status(self, user_id: str, status: str, db: Session = None):
        if user_id in self.user_status:
            self.user_status[user_id]["status"] = status
            self.user_status[user_id]["last_seen"] = datetime.now().isoformat()
            
            if db:
                user = crud.user.get(db, id=user_id)
                if user:
                    user_in = schemas.UserUpdate(last_login=datetime.now())
                    crud.user.update(db, db_obj=user, obj_in=user_in)
            
            await self.broadcast_user_status()

    async def broadcast_user_status(self):
        message = {
            "type": "user_status_update",
            "payload": self.user_status
        }
        for connection in self.active_connections.values():
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

    async def _heartbeat_check(self, user_id: str, db: Session):
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                if user_id in self.user_status:
                    last_seen = datetime.fromisoformat(self.user_status[user_id]["last_seen"])
                    if datetime.now() - last_seen > timedelta(minutes=5):
                        # User is inactive for more than 5 minutes
                        await self.update_user_status(user_id, "away", db)
        except asyncio.CancelledError:
            pass

    async def handle_heartbeat(self, user_id: str, db: Session = None):
        if user_id in self.user_status:
            self.user_status[user_id]["last_seen"] = datetime.now().isoformat()
            if self.user_status[user_id]["status"] == "away":
                await self.update_user_status(user_id, "online", db)

manager = ConnectionManager()