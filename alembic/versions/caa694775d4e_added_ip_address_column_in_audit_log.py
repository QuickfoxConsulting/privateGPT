"""Added ip_address column in audit log

Revision ID: caa694775d4e
Revises: 9ae2f4e97436
Create Date: 2024-03-05 10:37:38.955333

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'caa694775d4e'
down_revision: Union[str, None] = '9ae2f4e97436'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('audit', sa.Column('ip_address', sa.String(), nullable=True))
    op.alter_column('document', 'department_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    # op.create_unique_constraint('unique_user_role', 'user_roles', ['user_id', 'role_id', 'company_id'])
    op.alter_column('users', 'department_id',
               existing_type=sa.INTEGER(),
               nullable=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('users', 'department_id',
               existing_type=sa.INTEGER(),
               nullable=True)
    # op.drop_constraint('unique_user_role', 'user_roles', type_='unique')
    op.alter_column('document', 'department_id',
               existing_type=sa.INTEGER(),
               nullable=True)
    op.drop_column('audit', 'ip_address')
    # ### end Alembic commands ###