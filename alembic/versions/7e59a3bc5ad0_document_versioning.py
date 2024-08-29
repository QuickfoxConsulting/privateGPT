from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '7e59a3bc5ad0'
down_revision: Union[str, None] = 'a065ce619603'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

version_type_enum = sa.Enum('NEW', 'AMENDMENT', name='versiontype')

def upgrade() -> None:
    version_type_enum.create(op.get_bind())
    
    # Step 1: Add the column with a default value
    op.add_column('document', sa.Column('version_type', version_type_enum, nullable=False, server_default='NEW'))
    op.add_column('document', sa.Column('previous_document_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_previous_document', 'document', 'document', ['previous_document_id'], ['id'])

    # Step 2: Update existing rows with the default value
    op.execute("UPDATE document SET version_type = 'NEW' WHERE version_type IS NULL")

    # Step 3: Remove the server default if needed
    op.alter_column('document', 'version_type', server_default=None)

def downgrade() -> None:
    op.drop_constraint('fk_previous_document', 'document', type_='foreignkey')
    op.drop_column('document', 'previous_document_id')
    op.drop_column('document', 'version_type')
    version_type_enum.drop(op.get_bind())
