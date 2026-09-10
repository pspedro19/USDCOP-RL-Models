from __future__ import annotations

import itertools
import os

import asyncpg
import pytest

from src.governance.declaration import (
    DAG_ELIGIBLE_STATES,
    CapitalTier,
    DeclarationError,
    GovernanceDeclaration,
    OperationalState,
    ResearchState,
)


@pytest.mark.asyncio
async def test_python_declaration_matrix_matches_live_postgres_checks_for_action_surface() -> None:
    """Compare the Python model with the live CHECKs for ``surface='action'``.

    ``GovernanceDeclaration`` does not model ``surface``.  Diagnostic-surface
    parity is therefore deliberately outside this test's claim and remains a
    contract gap tracked by BL-16.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not provided; PostgreSQL integration not executed")

    connection = await asyncpg.connect(database_url)
    try:
        constraints = await connection.fetch(
            """
            SELECT c.conname, pg_get_expr(c.conbin, c.conrelid) AS expression
            FROM pg_constraint c
            WHERE c.conrelid = 'control.strategy_declaration'::regclass
              AND c.contype = 'c'
            ORDER BY c.conname
            """
        )
        assert constraints, "strategy_declaration must expose live CHECK constraints"

        dag_states = {state.value for state in DAG_ELIGIBLE_STATES}
        mismatches: list[tuple[str, str, str, bool, list[str]]] = []
        combinations = itertools.product(ResearchState, CapitalTier, OperationalState)
        for research_state, capital_tier, operational_state in combinations:
            exit_checklist = (
                "PASS" if research_state is ResearchState.WITHDRAWN else None
            )
            dag_declared = research_state.value in dag_states
            try:
                GovernanceDeclaration.from_mapping(
                    {
                        "research_state": research_state.value,
                        "capital_tier": capital_tier.value,
                        "operational_state": operational_state.value,
                        "exit_checklist": exit_checklist,
                        "dag_declared": dag_declared,
                    }
                )
                python_legal = True
            except DeclarationError:
                python_legal = False

            failed_constraints: list[str] = []
            for constraint in constraints:
                query = f"""
                    SELECT COALESCE(({constraint['expression']}), TRUE)
                    FROM (
                        SELECT $1::text AS research_state,
                               $2::text AS capital_tier,
                               $3::text AS operational_state,
                               $4::text AS exit_checklist,
                               $5::boolean AS dag_declared,
                               'action'::text AS surface,
                               ('sha256:' || repeat('a', 64))::text
                                   AS spec_fingerprint
                    ) AS candidate
                """
                sql_legal = await connection.fetchval(
                    query,
                    research_state.value,
                    capital_tier.value,
                    operational_state.value,
                    exit_checklist,
                    dag_declared,
                )
                if not sql_legal:
                    failed_constraints.append(constraint["conname"])

            postgres_legal = not failed_constraints
            if python_legal != postgres_legal:
                mismatches.append(
                    (
                        research_state.value,
                        capital_tier.value,
                        operational_state.value,
                        python_legal,
                        failed_constraints,
                    )
                )

        assert mismatches == []
    finally:
        await connection.close()
