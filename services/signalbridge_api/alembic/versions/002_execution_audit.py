"""
Execution Audit Table

Create table for detailed execution audit trail.
Tracks every event in the signal-to-execution pipeline.

Revision ID: 002_execution_audit
Revises: 001_user_risk_limits
Create Date: 2026-01-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "002_execution_audit"
down_revision = "001_user_risk_limits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create execution_audit table."""

    # Create enum for event types
    event_type_enum = postgresql.ENUM(
        "SIGNAL_RECEIVED",
        "RISK_CHECK_STARTED",
        "RISK_CHECK_PASSED",
        "RISK_CHECK_FAILED",
        "EXECUTION_STARTED",
        "EXECUTION_SUBMITTED",
        "EXECUTION_FILLED",
        "EXECUTION_FAILED",
        "EXECUTION_CANCELLED",
        "KILL_SWITCH_ACTIVATED",
        "KILL_SWITCH_DEACTIVATED",
        name="bridge_event_type",
        create_type=True,
    )

    op.create_table(
        "execution_audit",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("execution_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "event_type",
            event_type_enum,
            nullable=False,
        ),
        sa.Column("event_data", postgresql.JSONB, server_default="{}", nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["execution_id"], ["executions.id"], ondelete="CASCADE"),
    )

    # Indexes for efficient querying
    op.create_index(
        "ix_execution_audit_execution_id",
        "execution_audit",
        ["execution_id"],
    )

    op.create_index(
        "ix_execution_audit_event_type",
        "execution_audit",
        ["event_type"],
    )

    op.create_index(
        "ix_execution_audit_created_at",
        "execution_audit",
        ["created_at"],
    )

    # Composite index for common queries
    op.create_index(
        "ix_execution_audit_execution_created",
        "execution_audit",
        ["execution_id", "created_at"],
    )

    # BRIN index for time-series queries (efficient for append-only tables)
    op.execute("""
        CREATE INDEX ix_execution_audit_created_at_brin
        ON execution_audit USING BRIN (created_at);
    """)


def downgrade() -> None:
    """Drop execution_audit table."""
    op.drop_index("ix_execution_audit_created_at_brin", table_name="execution_audit")
    op.drop_index("ix_execution_audit_execution_created", table_name="execution_audit")
    op.drop_index("ix_execution_audit_created_at", table_name="execution_audit")
    op.drop_index("ix_execution_audit_event_type", table_name="execution_audit")
    op.drop_index("ix_execution_audit_execution_id", table_name="execution_audit")
    op.drop_table("execution_audit")

    # Drop the enum type
    op.execute("DROP TYPE IF EXISTS bridge_event_type;")
