"""Add release_date and ffilled_from_date columns to macro_indicators_daily.

Revision ID: 021_add_release_date
Revises: None
Create Date: 2026-01-17

This migration adds two new columns to the macro_indicators_daily table:
- release_date: The actual release date of the macro indicator data
- ffilled_from_date: The date from which forward-filled data was sourced

These columns support tracking data provenance and identifying stale/forward-filled
values in the macro indicators table.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic
revision: str = "021_add_release_date"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add release_date and ffilled_from_date columns to macro_indicators_daily.

    The release_date column tracks when the macro indicator data was officially
    released by the source (e.g., when FRED published the unemployment rate).

    The ffilled_from_date column tracks the original date of data that was
    forward-filled to populate missing values. This is critical for:
    - Tracking data quality and staleness
    - Identifying which records contain actual vs. interpolated data
    - Auditing and compliance requirements
    """
    # Add release_date column (nullable Date)
    op.add_column(
        "macro_indicators_daily",
        sa.Column(
            "release_date",
            sa.Date(),
            nullable=True,
            comment="Official release date of the macro indicator data from source",
        ),
    )

    # Add ffilled_from_date column (nullable Date)
    op.add_column(
        "macro_indicators_daily",
        sa.Column(
            "ffilled_from_date",
            sa.Date(),
            nullable=True,
            comment="Original date of forward-filled data (NULL if data is original)",
        ),
    )

    # Create index on release_date for efficient querying
    # This index supports queries like:
    # - Finding records by release date range
    # - Identifying the most recent releases
    # - Joining with economic calendar data
    op.create_index(
        "idx_macro_release_date",
        "macro_indicators_daily",
        ["release_date"],
        unique=False,
    )

    # Create composite index for staleness analysis queries
    # This supports queries that identify forward-filled vs original data
    op.create_index(
        "idx_macro_ffill_tracking",
        "macro_indicators_daily",
        ["date", "ffilled_from_date"],
        unique=False,
    )


def downgrade() -> None:
    """
    Remove release_date and ffilled_from_date columns from macro_indicators_daily.

    This rollback will:
    - Drop the indexes created for these columns
    - Remove the columns themselves

    WARNING: This will permanently delete any data stored in these columns.
    Ensure you have a backup before running this downgrade.
    """
    # Drop composite index for ffill tracking
    op.drop_index(
        "idx_macro_ffill_tracking",
        table_name="macro_indicators_daily",
    )

    # Drop index on release_date
    op.drop_index(
        "idx_macro_release_date",
        table_name="macro_indicators_daily",
    )

    # Remove ffilled_from_date column
    op.drop_column("macro_indicators_daily", "ffilled_from_date")

    # Remove release_date column
    op.drop_column("macro_indicators_daily", "release_date")
