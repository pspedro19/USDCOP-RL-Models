"""
User Risk Limits Table

Create table for user-specific risk limits that override
the default RiskEnforcer limits.

Revision ID: 001_user_risk_limits
Revises:
Create Date: 2026-01-22
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "001_user_risk_limits"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create user_risk_limits table."""
    op.create_table(
        "user_risk_limits",
        sa.Column("id", postgresql.UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("max_daily_loss_pct", sa.Numeric(5, 2), server_default="2.00", nullable=False),
        sa.Column("max_trades_per_day", sa.Integer(), server_default="10", nullable=False),
        sa.Column("max_position_size_usd", sa.Numeric(20, 8), server_default="1000.00", nullable=False),
        sa.Column("cooldown_minutes", sa.Integer(), server_default="15", nullable=False),
        sa.Column("enable_short", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", name="uq_user_risk_limits_user_id"),
    )

    # Index for fast user lookup
    op.create_index(
        "ix_user_risk_limits_user_id",
        "user_risk_limits",
        ["user_id"],
    )

    # Trigger to auto-update updated_at
    op.execute("""
        CREATE OR REPLACE FUNCTION update_user_risk_limits_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    op.execute("""
        CREATE TRIGGER trigger_user_risk_limits_updated_at
        BEFORE UPDATE ON user_risk_limits
        FOR EACH ROW
        EXECUTE FUNCTION update_user_risk_limits_updated_at();
    """)


def downgrade() -> None:
    """Drop user_risk_limits table."""
    op.execute("DROP TRIGGER IF EXISTS trigger_user_risk_limits_updated_at ON user_risk_limits;")
    op.execute("DROP FUNCTION IF EXISTS update_user_risk_limits_updated_at();")
    op.drop_index("ix_user_risk_limits_user_id", table_name="user_risk_limits")
    op.drop_table("user_risk_limits")
