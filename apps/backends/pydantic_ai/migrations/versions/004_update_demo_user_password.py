"""Update demo user with proper bcrypt password hash

Revision ID: 004_update_demo_user_password
Revises: 003_add_last_login_to_users
Create Date: 2025-06-01 22:27:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from passlib.context import CryptContext


# revision identifiers, used by Alembic.
revision: str = '004_update_demo_user_password'
down_revision: Union[str, None] = '003_add_last_login_to_users'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Update demo user with proper bcrypt password hash."""
    # Create password context for hashing
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    # Hash the password "demo" -> will work with bcrypt verification
    demo_password_hash = pwd_context.hash("demo")
    
    # Update the demo user with proper password hash
    op.execute(f"""
        UPDATE users 
        SET password_hash = '{demo_password_hash}'
        WHERE username = 'demo';
    """)


def downgrade() -> None:
    """Revert demo user password hash to placeholder."""
    op.execute("""
        UPDATE users 
        SET password_hash = 'demo_hash_placeholder'
        WHERE username = 'demo';
    """)