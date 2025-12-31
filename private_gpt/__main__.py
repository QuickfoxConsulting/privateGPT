# start a fastapi server with uvicorn

import os
import uvicorn
from pathlib import Path
from datetime import datetime

from private_gpt.main import app
from private_gpt.settings.settings import settings
from private_gpt.utils.logging_config import setup_logging
# import nest_asyncio
# nest_asyncio.apply()
# Set log_config=None to do not use the uvicorn logging configuration, and
# use ours instead. For reference, see below:
# https://github.com/tiangolo/fastapi/discussions/7457#discussioncomment-5141108

log_dir = Path("logs")
log_file = log_dir / f"private_gpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
setup_logging(
    log_level="INFO",
    rich_tracebacks=True,
    rich_markup=True
)
# Configure uvicorn logging
log_config = uvicorn.config.LOGGING_CONFIG
log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"

uvicorn.run(
    app,
    host="0.0.0.0",
    port=settings().server.port,
    log_config=log_config
)