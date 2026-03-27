"""Add NER Entity models

Revision ID: 0aef2a09f209
Revises: 8cf44e862a10
Create Date: 2026-02-13 08:24:56.986547

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0aef2a09f209'
down_revision: Union[str, None] = '8cf44e862a10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create ner_entities table
    op.create_table(
        'ner_entities',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_ner_entities_id'),   'ner_entities', ['id'],   unique=False)
    op.create_index(op.f('ix_ner_entities_name'), 'ner_entities', ['name'], unique=False)
    op.create_index(op.f('ix_ner_entities_type'), 'ner_entities', ['type'], unique=False)

    # Create node_ner_entities junction table
    op.create_table(
        'node_ner_entities',
        sa.Column('id',        sa.Integer(), nullable=False),
        sa.Column('entity_id', sa.Integer(), nullable=False),
        sa.Column('node_id',   sa.String(),  nullable=False),
        sa.Column('doc_id',    sa.String(),  nullable=False),
        sa.ForeignKeyConstraint(
            ['entity_id'], ['ner_entities.id'],
            ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_node_ner_entities_id'),      'node_ner_entities', ['id'],      unique=False)
    op.create_index(op.f('ix_node_ner_entities_node_id'), 'node_ner_entities', ['node_id'], unique=False)
    op.create_index(op.f('ix_node_ner_entities_doc_id'),  'node_ner_entities', ['doc_id'],  unique=False)


def downgrade() -> None:
    # Drop node_ner_entities first (has FK dependency on ner_entities)
    op.drop_index(op.f('ix_node_ner_entities_doc_id'),  table_name='node_ner_entities')
    op.drop_index(op.f('ix_node_ner_entities_node_id'), table_name='node_ner_entities')
    op.drop_index(op.f('ix_node_ner_entities_id'),      table_name='node_ner_entities')
    op.drop_table('node_ner_entities')

    # Drop ner_entities
    op.drop_index(op.f('ix_ner_entities_type'), table_name='ner_entities')
    op.drop_index(op.f('ix_ner_entities_name'), table_name='ner_entities')
    op.drop_index(op.f('ix_ner_entities_id'),   table_name='ner_entities')
    op.drop_table('ner_entities')