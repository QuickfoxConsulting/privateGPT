from private_gpt.users.core.config import settings

SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}".format(
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    db_name=settings.DB_NAME,
    username=settings.DB_USER,
    password=settings.DB_PASSWORD,
)

# Connection pool configuration
POOL_SIZE = 10  # Number of connections to keep open in the pool
MAX_OVERFLOW = 20  # Maximum number of connections to create beyond pool_size
POOL_TIMEOUT = 30  # Seconds to wait before giving up on getting a connection from the pool
POOL_RECYCLE = 3600  # Recycle connections after 1 hour to prevent stale connections
POOL_PRE_PING = True  # Enable connection health checks before using