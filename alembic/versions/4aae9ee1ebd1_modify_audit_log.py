"""modify audit log

Revision ID: 4aae9ee1ebd1
Revises: b3579174f52d
Create Date: 2025-09-22 10:32:53.521835
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '4aae9ee1ebd1'
down_revision: Union[str, None] = 'b3579174f52d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # add new audit columns
    op.add_column('audit', sa.Column('username', sa.String(length=100), nullable=True))
    op.add_column('audit', sa.Column('user_agent', sa.Text(), nullable=True))
    op.add_column('audit', sa.Column('session_id', sa.String(length=100), nullable=True))
    op.add_column('audit', sa.Column('request_id', sa.String(length=100), nullable=True))
    op.add_column('audit', sa.Column('severity', sa.String(length=20), nullable=True))
    op.add_column('audit', sa.Column('resource_id', sa.String(length=100), nullable=True))

    # alter faqs.answer from TEXT -> JSONB with explicit cast
    op.execute(
        """
        ALTER TABLE faqs
        ALTER COLUMN answer
        TYPE JSONB
        USING answer::jsonb
        """
    )

    # if you want the unique constraint uncomment and adjust:
    # op.create_unique_constraint('unique_user_role', 'user_roles', ['user_id', 'role_id', 'company_id'])


def downgrade() -> None:
    # reverse faqs.answer back to TEXT
    op.execute(
        """
        ALTER TABLE faqs
        ALTER COLUMN answer
        TYPE TEXT
        USING answer::text
        """
    )

    # drop audit columns
    op.drop_column('audit', 'resource_id')
    op.drop_column('audit', 'severity')
    op.drop_column('audit', 'request_id')
    op.drop_column('audit', 'session_id')
    op.drop_column('audit', 'user_agent')
    op.drop_column('audit', 'username')

    # drop constraint if created in upgrade
    # op.drop_constraint('unique_user_role', 'user_roles', type_='unique')
