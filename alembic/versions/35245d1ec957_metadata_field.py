"""Metadata_field

Revision ID: 35245d1ec957
Revises: fe4501b3fbe3
Create Date: 2025-05-11 08:43:20.147449

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '35245d1ec957'
down_revision: Union[str, None] = 'fe4501b3fbe3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Define enum separately
document_status_enum = sa.Enum('INGESTING', 'EMBEDDING', 'READY', name='documentstatus')

def upgrade() -> None:
    # ✅ Explicitly create the enum type
    document_status_enum.create(op.get_bind(), checkfirst=True)

    # Add new fields
    op.add_column('document', sa.Column('doc_status', document_status_enum, nullable=True))
    op.add_column('document', sa.Column('doc_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True))
    op.drop_column('document', 'tags')

def downgrade() -> None:
    # Restore 'tags' column
    op.add_column('document', sa.Column('tags', sa.VARCHAR(length=512), autoincrement=False, nullable=True))
    op.drop_column('document', 'doc_metadata')
    op.drop_column('document', 'doc_status')

    # ✅ Drop the enum type
    document_status_enum.drop(op.get_bind(), checkfirst=True)
