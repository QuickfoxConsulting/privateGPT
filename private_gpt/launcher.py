import logging
from typing import Annotated
from injector import Injector
from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.global_handlers import create_global_handler
from llama_index.core.settings import Settings as LlamaIndexSettings
from private_gpt.settings.settings import Settings
from private_gpt.users.api.v1.api import api_router
from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.chunks.chunks_router import chunks_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.recipes.summarize.summarize_router import summarize_router
from private_gpt.users.api.v1.routers.websocket_router import websocket_router

logger = logging.getLogger(__name__)

def create_app(root_injector: Injector) -> FastAPI:
    """Create the FastAPI application with dependency injection."""
    
    app = FastAPI()
    
    # Use middleware for regular HTTP requests
    @app.middleware("http")
    async def injector_middleware(request: Request, call_next):
        request.state.injector = root_injector
        response = await call_next(request)
        return response
    
    # Include all routers
    app.include_router(completions_router)
    app.include_router(chat_router)
    app.include_router(chunks_router)
    app.include_router(ingest_router)
    app.include_router(embeddings_router)
    app.include_router(summarize_router)
    app.include_router(health_router)
    app.include_router(websocket_router)
    app.include_router(api_router)
    
    # Define a function to get injector from request
    def get_injector(request: Request) -> Injector:
        return request.state.injector
    
    # Create a dependency
    InjectorDep = Annotated[Injector, Depends(get_injector)]
    
    # Store the dependency on the app
    app.state.InjectorDep = InjectorDep
    
    # Store the root injector for WebSocket endpoints to use
    app.state.root_injector = root_injector
    
    # Add LlamaIndex simple observability
    global_handler = create_global_handler("simple")
    LlamaIndexSettings.callback_manager = CallbackManager([global_handler])
    
    # Set up CORS if enabled
    settings = root_injector.get(Settings)
    if settings.server.cors.enabled:
        logger.debug("Setting up CORS middleware")
        app.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_origins=[
                "http://localhost:3000/",
                "http://10.1.101.125:80",
                "http://quickgpt.gibl.com.np:80",
                "http://127.0.0.1",
                "http://10.1.101.125",
                "http://quickgpt.gibl.com.np",
                "http://localhost:8000",
                "http://192.168.1.93",
                "http://192.168.1.93:88",
                "http://192.168.1.98",
                "http://192.168.1.98:3000",
                "http://localhost:3000",
                "https://globaldocquery.gibl.com.np/",
                "http://127.0.0.1/",
                "http://localhost/",
                "http://localhost:80",
                "http://192.168.1.131",
                'http://192.168.1.131:3000',
                "http://192.168.1.65",
                'http://192.168.1.65:3001',
                "http://192.168.1.109",
                'http://192.168.1.109:3001',
                "http://192.168.1.135",
                'http://192.168.1.135:3000',
                "http://192.168.101.80",
                'http://192.168.101.80:3000',
                "ws://localhost:3000",
            ],
            allow_methods=["DELETE", "GET", "POST", "PUT", "OPTIONS", "PATCH"],
            allow_headers=["*"],
        )
    
    
    return app