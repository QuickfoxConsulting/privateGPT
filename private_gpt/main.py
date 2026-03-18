"""FastAPI app creation, logger configuration and main API routes."""

from private_gpt.di import global_injector
from private_gpt.launcher import create_app
from fastapi_pagination import add_pagination
from fastapi_pagination.utils import disable_installed_extensions_check

disable_installed_extensions_check()

app = create_app(global_injector)
add_pagination(app)