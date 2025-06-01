"""Add last_login column to users table

Revision ID: 003_add_last_login_to_users
Revises: 002_seed_demo_user
Create Date: 2025-06-01 22:22:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003_add_last_login_to_users'
down_revision: Union[str, None] = '002_seed_demo_user'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add last_login column to users table."""
    # Add last_login column to users table
    op.add_column('users', sa.Column('last_login', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    """Remove last_login column from users table."""
    # Remove last_login column from users table
    op.drop_column('users', 'last_login')