from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '89e9f96ee1cd'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Enable pg_vector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Optional: Create unique constraint on user_roles if needed
    # op.create_unique_constraint('unique_user_role', 'user_roles', ['user_id', 'role_id', 'company_id'])

def downgrade() -> None:
    # Remove the unique constraint
    # op.drop_constraint('unique_user_role', 'user_roles', type_='unique')

    # Remove pg_vector extension
    op.execute('DROP EXTENSION IF EXISTS vector')