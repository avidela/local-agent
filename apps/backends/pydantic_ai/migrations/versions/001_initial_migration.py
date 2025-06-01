"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2025-06-01 19:41:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    # Create enums with DO blocks to handle IF NOT EXISTS
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'userrole') THEN
                CREATE TYPE userrole AS ENUM ('user', 'admin', 'moderator');
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'modelprovider') THEN
                CREATE TYPE modelprovider AS ENUM ('openai', 'anthropic', 'google');
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'sessionstatus') THEN
                CREATE TYPE sessionstatus AS ENUM ('active', 'completed', 'failed', 'archived');
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'messagerole') THEN
                CREATE TYPE messagerole AS ENUM ('system', 'user', 'assistant', 'tool');
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'filetype') THEN
                CREATE TYPE filetype AS ENUM ('image', 'audio', 'video', 'document', 'other');
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'evaluationstatus') THEN
                CREATE TYPE evaluationstatus AS ENUM ('pending', 'running', 'completed', 'failed');
            END IF;
        END $$;
    """)
    
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'workflowstatus') THEN
                CREATE TYPE workflowstatus AS ENUM ('pending', 'running', 'paused', 'completed', 'failed');
            END IF;
        END $$;
    """)

    # Create users table
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('username', sa.String(50), nullable=False),
    sa.Column('email', sa.String(255), nullable=False),
    sa.Column('password_hash', sa.String(255), nullable=False),
    sa.Column('full_name', sa.String(200), nullable=True),
    sa.Column('role', sa.Text(), nullable=False, server_default='user'),
    sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('email'),
    sa.UniqueConstraint('username')
    )
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)

    # Create agents table
    op.create_table('agents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(100), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('system_prompt', sa.Text(), nullable=False),
    sa.Column('model_provider', sa.Text(), nullable=False),
    sa.Column('model_name', sa.String(100), nullable=False),
    sa.Column('temperature', sa.Float(), nullable=False, server_default='0.7'),
    sa.Column('max_tokens', sa.Integer(), nullable=True),
    sa.Column('tools', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
    sa.Column('output_type', sa.String(100), nullable=True),
    sa.Column('retries', sa.Integer(), nullable=False, server_default='2'),
    sa.Column('is_public', sa.Boolean(), nullable=False, server_default='false'),
    sa.Column('is_active', sa.Boolean(), nullable=False, server_default='true'),
    sa.Column('usage_count', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('owner_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_agents_id'), 'agents', ['id'], unique=False)

    # Create sessions table
    op.create_table('sessions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.String(36), nullable=False),
    sa.Column('title', sa.String(200), nullable=True),
    sa.Column('status', sa.Text(), nullable=False, server_default='active'),
    sa.Column('total_cost', sa.Float(), nullable=False, server_default='0.0'),
    sa.Column('total_tokens', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('request_tokens', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('response_tokens', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('meta_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('agent_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('session_id')
    )
    op.create_index(op.f('ix_sessions_id'), 'sessions', ['id'], unique=False)

    # Create messages table
    op.create_table('messages',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('content', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('role', sa.Text(), nullable=False),
    sa.Column('attachments', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
    sa.Column('cost', sa.Float(), nullable=False, server_default='0.0'),
    sa.Column('tokens', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('tool_calls', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('tool_response', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('meta_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    sa.Column('session_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_messages_id'), 'messages', ['id'], unique=False)

    # Create files table
    op.create_table('files',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('filename', sa.String(255), nullable=False),
    sa.Column('original_filename', sa.String(255), nullable=False),
    sa.Column('file_path', sa.String(500), nullable=False),
    sa.Column('file_type', sa.Text(), nullable=False),
    sa.Column('file_size', sa.Integer(), nullable=False),
    sa.Column('mime_type', sa.String(100), nullable=True),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_files_id'), 'files', ['id'], unique=False)

    # Create evaluations table
    op.create_table('evaluations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(100), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('status', sa.Text(), nullable=False, server_default='pending'),
    sa.Column('agent_id', sa.Integer(), nullable=False),
    sa.Column('dataset_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('evaluator_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('results', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    sa.Column('score', sa.Float(), nullable=True),
    sa.Column('total_cases', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('passed_cases', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('failed_cases', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('duration_seconds', sa.Float(), nullable=True),
    sa.Column('created_by', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['agent_id'], ['agents.id'], ),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_evaluations_id'), 'evaluations', ['id'], unique=False)

    # Create workflows table
    op.create_table('workflows',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(100), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('workflow_type', sa.String(50), nullable=False),
    sa.Column('status', sa.Text(), nullable=False, server_default='pending'),
    sa.Column('graph_config', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('initial_state', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    sa.Column('current_state', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    sa.Column('current_node', sa.String(100), nullable=True),
    sa.Column('execution_history', postgresql.ARRAY(postgresql.JSONB(astext_type=sa.Text())), nullable=False, server_default='{}'),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('paused_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
    sa.Column('max_retries', sa.Integer(), nullable=False, server_default='3'),
    sa.Column('created_by', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['created_by'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflows_id'), 'workflows', ['id'], unique=False)

    # Create workflow_executions table
    op.create_table('workflow_executions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('workflow_id', sa.Integer(), nullable=False),
    sa.Column('status', sa.Text(), nullable=False, server_default='pending'),
    sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    sa.Column('output_data', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
    sa.Column('execution_log', postgresql.ARRAY(postgresql.JSONB(astext_type=sa.Text())), nullable=False, server_default='{}'),
    sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['workflow_id'], ['workflows.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_workflow_executions_id'), 'workflow_executions', ['id'], unique=False)

def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('workflow_executions')
    op.drop_table('workflows')
    op.drop_table('evaluations')
    op.drop_table('files')
    op.drop_table('messages')
    op.drop_table('sessions')
    op.drop_table('agents')
    op.drop_table('users')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS workflowstatus')
    op.execute('DROP TYPE IF EXISTS evaluationstatus')
    op.execute('DROP TYPE IF EXISTS filetype')
    op.execute('DROP TYPE IF EXISTS messagerole')
    op.execute('DROP TYPE IF EXISTS sessionstatus')
    op.execute('DROP TYPE IF EXISTS modelprovider')
    op.execute('DROP TYPE IF EXISTS userrole')