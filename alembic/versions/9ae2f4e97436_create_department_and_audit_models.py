"""Create department and audit models

Revision ID: 9ae2f4e97436
Revises: 
Create Date: 2024-02-28 14:53:19.144973

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9ae2f4e97436'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('departments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(), nullable=True),
    sa.Column('company_id', sa.Integer(), nullable=True),
    sa.Column('total_users', sa.Integer(), nullable=True),
    sa.Column('total_documents', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['company_id'], ['companies.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_departments_id'), 'departments', ['id'], unique=False)
    op.create_index(op.f('ix_departments_name'), 'departments', ['name'], unique=True)
    op.create_table('audit',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('timestamp', sa.DateTime(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('model', sa.String(), nullable=False),
    sa.Column('action', sa.String(), nullable=False),
    sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_id'), 'audit', ['id'], unique=False)
    op.add_column('document', sa.Column('department_id', sa.Integer(), nullable=True))
    op.create_foreign_key(None, 'document', 'departments', ['department_id'], ['id'])
    # op.create_unique_constraint('unique_user_role', 'user_roles', ['user_id', 'role_id', 'company_id'])
    op.add_column('users', sa.Column('password_created', sa.DateTime(), nullable=True))
    op.add_column('users', sa.Column('department_id', sa.Integer(), nullable=True))
    op.create_foreign_key(None, 'users', 'departments', ['department_id'], ['id'])
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'users', type_='foreignkey')
    op.drop_column('users', 'department_id')
    op.drop_column('users', 'password_created')
    # op.drop_constraint('unique_user_role', 'user_roles', type_='unique')
    op.drop_constraint(None, 'document', type_='foreignkey')
    op.drop_column('document', 'department_id')
    op.drop_index(op.f('ix_audit_id'), table_name='audit')
    op.drop_table('audit')
    op.drop_index(op.f('ix_departments_name'), table_name='departments')
    op.drop_index(op.f('ix_departments_id'), table_name='departments')
    op.drop_table('departments')
    # ### end Alembic commands ###