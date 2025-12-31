"""add_error_status_to_documentstatus

Revision ID: 74c21a6df740
Revises: b27038ed57a8
Create Date: 2025-12-30 11:35:58.895299

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '74c21a6df740'
down_revision: Union[str, None] = 'b27038ed57a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Use execute to add the value to the PostgreSQL enum
    # We wrap it in try/except or just use a raw execute. 
    # In PG, ALTER TYPE ... ADD VALUE cannot be executed in a transaction block
    # so we might need to handle that, but typically simple execute works if it's the only thing.
    op.execute("ALTER TYPE documentstatus ADD VALUE 'ERROR'")


def downgrade() -> None:
    # PostgreSQL doesn't easily support removing enum values
    pass
