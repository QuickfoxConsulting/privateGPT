"""Set null on delete audit

Revision ID: 186dc125c8c4
Revises: 9aa759c05b19
Create Date: 2024-02-26 15:43:49.759556

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '186dc125c8c4'
down_revision: Union[str, None] = '9aa759c05b19'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint('audit_user_id_fkey', 'audit', type_='foreignkey')
    op.create_foreign_key(None, 'audit', 'users', ['user_id'], ['id'], ondelete='SET NULL')
    # op.create_unique_constraint('unique_user_role', 'user_roles', ['user_id', 'role_id', 'company_id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    # op.drop_constraint('unique_user_role', 'user_roles', type_='unique')
    op.drop_constraint(None, 'audit', type_='foreignkey')
    op.create_foreign_key('audit_user_id_fkey', 'audit', 'users', ['user_id'], ['id'])
    # ### end Alembic commands ###