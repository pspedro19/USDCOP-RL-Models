"""
Signal-Execution Link Columns

Add columns to executions table to link with inference signals.
Enables tracing execution back to the original model prediction.

Revision ID: 003_signal_execution_link
Revises: 002_execution_audit
Create Date: 2026-01-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "003_signal_execution_link"
down_revision = "002_execution_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add signal linkage columns to executions table."""

    # Add inference signal ID - links to the original signal from inference API
    op.add_column(
        "executions",
        sa.Column(
            "inference_signal_id",
            sa.String(100),
            nullable=True,
            comment="UUID of the signal from inference API",
        ),
    )

    # Add model ID - identifies which model generated the signal
    op.add_column(
        "executions",
        sa.Column(
            "model_id",
            sa.String(100),
            nullable=True,
            comment="Model that generated the signal",
        ),
    )

    # Add confidence score - model's confidence in the prediction
    op.add_column(
        "executions",
        sa.Column(
            "confidence",
            sa.Numeric(5, 4),
            nullable=True,
            comment="Model confidence score (0-1)",
        ),
    )

    # Add processing time tracking
    op.add_column(
        "executions",
        sa.Column(
            "processing_time_ms",
            sa.Float(),
            nullable=True,
            comment="Time from signal received to execution submitted (ms)",
        ),
    )

    # Add risk check result storage
    op.add_column(
        "executions",
        sa.Column(
            "risk_check_result",
            postgresql.JSONB,
            nullable=True,
            comment="Risk check decision and metadata",
        ),
    )

    # Indexes for common queries
    op.create_index(
        "ix_executions_inference_signal_id",
        "executions",
        ["inference_signal_id"],
    )

    op.create_index(
        "ix_executions_model_id",
        "executions",
        ["model_id"],
    )

    # Composite index for model performance analysis
    op.create_index(
        "ix_executions_model_created",
        "executions",
        ["model_id", "created_at"],
    )

    # Partial index for high-confidence executions
    op.execute("""
        CREATE INDEX ix_executions_high_confidence
        ON executions (confidence)
        WHERE confidence >= 0.7;
    """)


def downgrade() -> None:
    """Remove signal linkage columns from executions table."""
    op.execute("DROP INDEX IF EXISTS ix_executions_high_confidence;")
    op.drop_index("ix_executions_model_created", table_name="executions")
    op.drop_index("ix_executions_model_id", table_name="executions")
    op.drop_index("ix_executions_inference_signal_id", table_name="executions")

    op.drop_column("executions", "risk_check_result")
    op.drop_column("executions", "processing_time_ms")
    op.drop_column("executions", "confidence")
    op.drop_column("executions", "model_id")
    op.drop_column("executions", "inference_signal_id")
