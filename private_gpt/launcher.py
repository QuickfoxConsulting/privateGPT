import logging
from typing import Annotated
import nest_asyncio

# Patch asyncio to allow nested event loops (required for LlamaIndex sync tools in async context)
nest_asyncio.apply()
from injector import Injector
from pathlib import Path
from fastapi import Depends, FastAPI, Request, WebSocket
from fastapi.staticfiles import StaticFiles
from private_gpt.constants import UPLOAD_DIR, UNCHECKED_DIR
from fastapi.middleware.cors import CORSMiddleware
from llama_index.core.callbacks import CallbackManager
from llama_index.core.callbacks.global_handlers import create_global_handler
from llama_index.core.settings import Settings as LlamaIndexSettings
from private_gpt.settings.settings import Settings
from private_gpt.users.api.v1.api import api_router
from private_gpt.server.chat.chat_router import chat_router
from private_gpt.server.health.health_router import health_router
from private_gpt.server.ingest.ingest_router import ingest_router
from private_gpt.server.completions.completions_router import completions_router
from private_gpt.server.embeddings.embeddings_router import embeddings_router
from private_gpt.server.recipes.summarize.summarize_router import summarize_router
from private_gpt.server.cache.cache_router import router as cache_router
from private_gpt.users.api.v1.routers.websocket_router import websocket_router
from private_gpt.server.integrations.integrations_router import integrations_router


from private_gpt.server.integrations.website_crawl_service import WebsiteCrawlService
logger = logging.getLogger(__name__)

def create_app(root_injector: Injector) -> FastAPI:
    """Create the FastAPI application with dependency injection."""
    
    # Initialize cache management
    try:
        _initialize_cache_management(root_injector)
    except Exception as e:
        logger.error(f"Failed to initialize cache management: {str(e)}")
    
    try:
        WebsiteCrawlService.start_scheduler()
    except Exception as e:
        logger.error(f"Failed to start website crawl scheduler: {str(e)}")
        
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
    app.include_router(ingest_router)
    app.include_router(embeddings_router)
    app.include_router(summarize_router)
    app.include_router(health_router)
    app.include_router(websocket_router)
    app.include_router(cache_router)
    app.include_router(api_router)
    app.include_router(integrations_router)

    # Mount static and media files
    app.mount("/static", StaticFiles(directory=str(UNCHECKED_DIR)), name="static")
    app.mount("/media", StaticFiles(directory=str(Path(UPLOAD_DIR) / "documents")), name="media")
    
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
                "http://localhost:3000",
                "http://10.1.101.125:80",
                "http://quickgpt.gibl.com.np",
                "http://127.0.0.1",
                "http://10.1.101.125",
                "http://quickgpt.gibl.com.np",
                "http://localhost:8000",
                "http://192.168.1.93",
                "http://192.168.1.93:88",
                "http://192.168.1.98",
                "http://192.168.1.98:3000",
                "http://192.168.1.68:3000",
                "http://localhost:3000",
                "https://globaldocquery.gibl.com.np",
                "http://127.0.0.1",
                "http://localhost",
                "http://localhost",
                "http://192.168.1.135",
                'http://192.168.1.135:3000',
                "http://192.168.101.6",
                'http://192.168.101.6:3000',
                "http://192.168.1.167",
                'http://192.168.101.167:3000',
                "http://18.224.145.213",
                "http://18.224.145.213:3000",
                "ws://localhost:3000",
                "ws://localhost",
                "http://192.168.1.85",
                "http://192.168.1.85:3000",
                "http://192.168.1.104",
                "http://192.168.1.104:3000",
            ],
            allow_methods=["DELETE", "GET", "POST", "PUT", "OPTIONS", "PATCH"],
            allow_headers=["*"],
        )
    
    return app

def _initialize_cache_management(injector: Injector) -> None:
    """Initialize cache management with warming and eviction policies."""
    try:
        # Get required services
        from private_gpt.server.cache.cache_manager import CacheManager
        cache_manager = injector.get(CacheManager)
        
        # Warm cache on startup
        from private_gpt.users.db.session import SessionLocal
        db = SessionLocal()
        try:
            cache_manager.warm_cache_on_startup(db, limit=50)
        finally:
            db.close()
            
        # Start background cache management
        cache_manager.start_background_cache_management(
            db_session_factory=SessionLocal,
            warm_interval=3600,  # Every hour
            eviction_interval=1800  # Every 30 minutes
        )
        
        logger.info("Cache management initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize cache management: {str(e)}")