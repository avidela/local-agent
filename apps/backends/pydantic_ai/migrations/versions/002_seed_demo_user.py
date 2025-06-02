"""Seed demo user

Revision ID: 002_seed_demo_user
Revises:
Create Date: 2025-05-31 19:06:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.sql import table, column
from sqlalchemy import String, Boolean, DateTime
from datetime import datetime
from passlib.context import CryptContext
import os

# revision identifiers, used by Alembic.
revision: str = '002_seed_demo_user'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create a table reference for the users table
    users_table = table('users',
        column('id', sa.Integer),
        column('username', String),
        column('email', String),
        column('password_hash', String),
        column('full_name', String),
        column('role', String),
        column('is_active', Boolean),
        column('last_login', DateTime),
        column('created_at', DateTime),
        column('updated_at', DateTime),
    )
    
    # Create password context for hashing
    pwd_context = CryptContext(
        schemes=["argon2"],
        deprecated="auto",
        argon2__memory_cost=65536,  # 64 MB
        argon2__time_cost=3,        # 3 iterations
        argon2__parallelism=2       # 2 threads
    )
    
    # Get demo user password from environment variable
    demo_password = os.getenv("DEMO_USER_PASSWORD", "demo")
    demo_password_hash = pwd_context.hash(demo_password)
    
    # Insert demo user
    now = datetime.utcnow()
    op.bulk_insert(users_table, [
        {
            'id': 1,
            'username': 'demo',
            'email': 'demo@example.com',
            'password_hash': demo_password_hash,
            'full_name': 'Demo User',
            'role': 'admin',
            'is_active': True,
            'last_login': None,
            'created_at': now,
            'updated_at': now,
        }
    ])


def downgrade() -> None:
    # Remove demo user
    op.execute("DELETE FROM users WHERE username = 'demo'")