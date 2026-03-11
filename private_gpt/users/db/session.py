from private_gpt.users.core.db_config import (
    SQLALCHEMY_DATABASE_URI,
    POOL_SIZE,
    MAX_OVERFLOW,
    POOL_TIMEOUT,
    POOL_RECYCLE,
    POOL_PRE_PING,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging

# logging.basicConfig()
# logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
# logging.getLogger("sqlalchemy.pool").setLevel(logging.DEBUG)

engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    echo=False,  # Set to True for SQL debugging
    future=True,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_recycle=POOL_RECYCLE,
    pool_pre_ping=POOL_PRE_PING,
    logging_name="myengine",
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
