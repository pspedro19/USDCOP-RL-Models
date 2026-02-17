"""
PromotionContract - Contrato de propuesta de promoción L4→Dashboard.
====================================================================
Contract ID: PROP-{model_id}
Version: 1.0.0

Generado por L4 después del backtest out-of-sample.
Requiere aprobación humana en Dashboard (segundo voto).

Usage:
    from src.core.contracts.promotion_contract import (
        PromotionProposal,
        BacktestMetrics,
        CriteriaResult,
        PromotionRecommendation,
    )

    # Create from backtest results
    proposal = PromotionProposal(
        proposal_id="PROP-exp1_20260131",
        model_id="exp1_curriculum_v1_20260131",
        experiment_name="exp1_curriculum_v1",
        recommendation=PromotionRecommendation.PROMOTE,
        confidence=0.85,
        reason="All criteria passed",
        metrics=backtest_metrics,
        criteria_results=criteria_results,
        lineage=lineage_dict,
    )

    # Save to database
    proposal.save_to_db(conn)
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)

PROMOTION_CONTRACT_VERSION = "1.0.0"


class PromotionRecommendation(Enum):
    """L4 recommendation for model promotion."""
    PROMOTE = "PROMOTE"    # All criteria passed, recommend promotion
    REJECT = "REJECT"      # Criteria failed, recommend rejection
    REVIEW = "REVIEW"      # Criteria passed but needs human review


class PromotionStatus(Enum):
    """Status of a promotion proposal."""
    PENDING_APPROVAL = "PENDING_APPROVAL"  # Awaiting human approval
    APPROVED = "APPROVED"                   # Approved by human (second vote)
    REJECTED = "REJECTED"                   # Rejected by human
    EXPIRED = "EXPIRED"                     # Expired without decision


class PromotionContractError(ValueError):
    """Raised when promotion contract validation fails."""
    pass


@dataclass
class BacktestMetrics:
    """Metrics from out-of-sample backtest."""

    sharpe_ratio: float
    max_drawdown: float       # As decimal (0.12 = 12%)
    win_rate: float           # As decimal (0.55 = 55%)
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float      # Average PnL per trade as decimal
    test_period_start: str    # ISO date string
    test_period_end: str      # ISO date string

    # Optional extended metrics
    calmar_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    max_consecutive_losses: Optional[int] = None
    recovery_factor: Optional[float] = None
    final_equity: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "avg_trade_pnl": self.avg_trade_pnl,
            "test_period_start": self.test_period_start,
            "test_period_end": self.test_period_end,
            "calmar_ratio": self.calmar_ratio,
            "sortino_ratio": self.sortino_ratio,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_consecutive_losses": self.max_consecutive_losses,
            "recovery_factor": self.recovery_factor,
            "final_equity": self.final_equity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestMetrics":
        """Create from dictionary."""
        return cls(
            sharpe_ratio=data.get("sharpe_ratio", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            win_rate=data.get("win_rate", 0.0),
            profit_factor=data.get("profit_factor", 0.0),
            total_trades=data.get("total_trades", 0),
            avg_trade_pnl=data.get("avg_trade_pnl", 0.0),
            test_period_start=data.get("test_period_start", ""),
            test_period_end=data.get("test_period_end", ""),
            calmar_ratio=data.get("calmar_ratio"),
            sortino_ratio=data.get("sortino_ratio"),
            avg_win=data.get("avg_win"),
            avg_loss=data.get("avg_loss"),
            max_consecutive_losses=data.get("max_consecutive_losses"),
            recovery_factor=data.get("recovery_factor"),
            final_equity=data.get("final_equity"),
        )


@dataclass
class CriteriaResult:
    """Result of evaluating a single success criterion."""

    name: str           # Criterion name (e.g., "min_sharpe")
    threshold: float    # Required threshold
    actual: float       # Actual value
    passed: bool        # Whether criterion was met

    @property
    def status_str(self) -> str:
        """Human-readable status string."""
        return "PASS" if self.passed else "FAIL"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "threshold": self.threshold,
            "actual": self.actual,
            "passed": self.passed,
            "status": self.status_str,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CriteriaResult":
        """Create from dictionary."""
        return cls(
            name=data.get("name", ""),
            threshold=data.get("threshold", 0.0),
            actual=data.get("actual", 0.0),
            passed=data.get("passed", False),
        )


@dataclass
class PromotionProposal:
    """
    Propuesta de promoción generada por L4 (primer voto).

    Esta propuesta requiere aprobación humana en el Dashboard (segundo voto)
    antes de que el modelo pueda ser promovido a producción.
    """

    # Identity
    proposal_id: str              # PROP-{model_id}
    model_id: str                 # Unique model identifier
    experiment_name: str          # From experiment YAML

    # Recommendation (primer voto - L4)
    recommendation: PromotionRecommendation
    confidence: float             # 0.0 to 1.0
    reason: str                   # Human-readable reason

    # Metrics del backtest
    metrics: BacktestMetrics

    # Resultados de criterios
    criteria_results: List[CriteriaResult] = field(default_factory=list)

    # Comparación vs baseline (si existe)
    vs_baseline: Optional[Dict[str, float]] = None
    baseline_model_id: Optional[str] = None

    # Lineage completo
    lineage: Dict[str, str] = field(default_factory=dict)

    # Status
    status: PromotionStatus = PromotionStatus.PENDING_APPROVAL
    requires_human_approval: bool = True

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    reviewer: Optional[str] = None
    reviewer_email: Optional[str] = None
    reviewer_notes: Optional[str] = None

    def __post_init__(self):
        """Set default expiration if not provided."""
        if self.expires_at is None:
            from datetime import timedelta
            self.expires_at = self.created_at + timedelta(days=7)

    def all_criteria_passed(self) -> bool:
        """Check if all criteria passed."""
        return all(cr.passed for cr in self.criteria_results)

    def get_failed_criteria(self) -> List[CriteriaResult]:
        """Get list of failed criteria."""
        return [cr for cr in self.criteria_results if not cr.passed]

    def approve(self, reviewer: str, reviewer_email: str, notes: Optional[str] = None) -> None:
        """
        Approve this proposal (segundo voto).

        Args:
            reviewer: Name of the reviewer
            reviewer_email: Email of the reviewer
            notes: Optional notes about the approval
        """
        self.status = PromotionStatus.APPROVED
        self.reviewed_at = datetime.utcnow()
        self.reviewer = reviewer
        self.reviewer_email = reviewer_email
        self.reviewer_notes = notes

    def reject(self, reviewer: str, reviewer_email: str, notes: Optional[str] = None) -> None:
        """
        Reject this proposal.

        Args:
            reviewer: Name of the reviewer
            reviewer_email: Email of the reviewer
            notes: Optional notes about the rejection
        """
        self.status = PromotionStatus.REJECTED
        self.reviewed_at = datetime.utcnow()
        self.reviewer = reviewer
        self.reviewer_email = reviewer_email
        self.reviewer_notes = notes

    def is_expired(self) -> bool:
        """Check if this proposal has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API/DB."""
        return {
            "proposal_id": self.proposal_id,
            "model_id": self.model_id,
            "experiment_name": self.experiment_name,
            "recommendation": self.recommendation.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "metrics": self.metrics.to_dict(),
            "criteria_results": [cr.to_dict() for cr in self.criteria_results],
            "vs_baseline": self.vs_baseline,
            "baseline_model_id": self.baseline_model_id,
            "lineage": self.lineage,
            "status": self.status.value,
            "requires_human_approval": self.requires_human_approval,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer": self.reviewer,
            "reviewer_email": self.reviewer_email,
            "reviewer_notes": self.reviewer_notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromotionProposal":
        """Create from dictionary."""
        metrics = BacktestMetrics.from_dict(data.get("metrics", {}))
        criteria_results = [
            CriteriaResult.from_dict(cr) for cr in data.get("criteria_results", [])
        ]

        return cls(
            proposal_id=data.get("proposal_id", ""),
            model_id=data.get("model_id", ""),
            experiment_name=data.get("experiment_name", ""),
            recommendation=PromotionRecommendation(data.get("recommendation", "REVIEW")),
            confidence=data.get("confidence", 0.0),
            reason=data.get("reason", ""),
            metrics=metrics,
            criteria_results=criteria_results,
            vs_baseline=data.get("vs_baseline"),
            baseline_model_id=data.get("baseline_model_id"),
            lineage=data.get("lineage", {}),
            status=PromotionStatus(data.get("status", "PENDING_APPROVAL")),
            requires_human_approval=data.get("requires_human_approval", True),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            reviewed_at=datetime.fromisoformat(data["reviewed_at"]) if data.get("reviewed_at") else None,
            reviewer=data.get("reviewer"),
            reviewer_email=data.get("reviewer_email"),
            reviewer_notes=data.get("reviewer_notes"),
        )

    def save_to_db(self, conn) -> int:
        """
        Guardar proposal en base de datos.

        Args:
            conn: Database connection

        Returns:
            ID of inserted row
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO promotion_proposals (
                    proposal_id, model_id, experiment_name,
                    recommendation, confidence, reason,
                    metrics, vs_baseline, criteria_results,
                    lineage, status, expires_at, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (proposal_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    reviewer = EXCLUDED.reviewer,
                    reviewer_notes = EXCLUDED.reviewer_notes,
                    reviewed_at = EXCLUDED.reviewed_at
                RETURNING id
            """, (
                self.proposal_id,
                self.model_id,
                self.experiment_name,
                self.recommendation.value,
                self.confidence,
                self.reason,
                json.dumps(self.metrics.to_dict()),
                json.dumps(self.vs_baseline) if self.vs_baseline else None,
                json.dumps([cr.to_dict() for cr in self.criteria_results]),
                json.dumps(self.lineage),
                self.status.value,
                self.expires_at,
                self.created_at,
            ))
            result = cur.fetchone()
            conn.commit()
            logger.info(f"Saved PromotionProposal: {self.proposal_id}")
            return result[0]

        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving PromotionProposal: {e}")
            raise
        finally:
            cur.close()

    @classmethod
    def from_db(cls, conn, proposal_id: str) -> Optional["PromotionProposal"]:
        """
        Load proposal from database.

        Args:
            conn: Database connection
            proposal_id: Proposal ID to load

        Returns:
            PromotionProposal or None if not found
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT
                    proposal_id, model_id, experiment_name,
                    recommendation, confidence, reason,
                    metrics, vs_baseline, criteria_results,
                    lineage, status, expires_at, created_at,
                    reviewed_at, reviewer, reviewer_notes
                FROM promotion_proposals
                WHERE proposal_id = %s
            """, (proposal_id,))
            row = cur.fetchone()

            if not row:
                return None

            metrics_data = row[6] if isinstance(row[6], dict) else json.loads(row[6])
            vs_baseline = row[7] if isinstance(row[7], dict) else (json.loads(row[7]) if row[7] else None)
            criteria_data = row[8] if isinstance(row[8], list) else json.loads(row[8])
            lineage = row[9] if isinstance(row[9], dict) else json.loads(row[9])

            return cls(
                proposal_id=row[0],
                model_id=row[1],
                experiment_name=row[2],
                recommendation=PromotionRecommendation(row[3]),
                confidence=float(row[4]),
                reason=row[5],
                metrics=BacktestMetrics.from_dict(metrics_data),
                vs_baseline=vs_baseline,
                criteria_results=[CriteriaResult.from_dict(cr) for cr in criteria_data],
                lineage=lineage,
                status=PromotionStatus(row[10]),
                expires_at=row[11],
                created_at=row[12],
                reviewed_at=row[13],
                reviewer=row[14],
                reviewer_notes=row[15],
            )
        finally:
            cur.close()

    @classmethod
    def get_pending_proposals(cls, conn) -> List["PromotionProposal"]:
        """
        Get all pending proposals.

        Args:
            conn: Database connection

        Returns:
            List of pending PromotionProposal
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT proposal_id
                FROM promotion_proposals
                WHERE status = 'PENDING_APPROVAL'
                  AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()

            proposals = []
            for row in rows:
                proposal = cls.from_db(conn, row[0])
                if proposal:
                    proposals.append(proposal)

            return proposals
        finally:
            cur.close()


def evaluate_criteria(
    metrics: BacktestMetrics,
    success_criteria: Dict[str, float],
    comparison: Optional[Dict[str, float]] = None,
) -> Tuple[List[CriteriaResult], bool]:
    """
    Evaluate backtest metrics against success criteria.

    Args:
        metrics: Backtest metrics
        success_criteria: Dict of criterion_name -> threshold
        comparison: Optional comparison vs baseline

    Returns:
        Tuple of (list of CriteriaResult, all_passed)
    """
    results = []

    # Min Sharpe
    if "min_sharpe" in success_criteria:
        results.append(CriteriaResult(
            name="min_sharpe",
            threshold=success_criteria["min_sharpe"],
            actual=metrics.sharpe_ratio,
            passed=metrics.sharpe_ratio >= success_criteria["min_sharpe"],
        ))

    # Max Drawdown
    if "max_drawdown" in success_criteria:
        results.append(CriteriaResult(
            name="max_drawdown",
            threshold=success_criteria["max_drawdown"],
            actual=metrics.max_drawdown,
            passed=metrics.max_drawdown <= success_criteria["max_drawdown"],
        ))

    # Min Win Rate
    if "min_win_rate" in success_criteria:
        results.append(CriteriaResult(
            name="min_win_rate",
            threshold=success_criteria["min_win_rate"],
            actual=metrics.win_rate,
            passed=metrics.win_rate >= success_criteria["min_win_rate"],
        ))

    # Min Trades
    if "min_trades" in success_criteria:
        results.append(CriteriaResult(
            name="min_trades",
            threshold=float(success_criteria["min_trades"]),
            actual=float(metrics.total_trades),
            passed=metrics.total_trades >= success_criteria["min_trades"],
        ))

    # Improvement threshold (if comparison available)
    if "improvement_threshold" in success_criteria and comparison:
        improvement = comparison.get("sharpe_improvement", 0.0)
        results.append(CriteriaResult(
            name="improvement_threshold",
            threshold=success_criteria["improvement_threshold"],
            actual=improvement,
            passed=improvement >= success_criteria["improvement_threshold"],
        ))

    all_passed = all(cr.passed for cr in results)
    return results, all_passed


def determine_recommendation(
    criteria_results: List[CriteriaResult],
    comparison: Optional[Dict[str, float]] = None,
) -> Tuple[PromotionRecommendation, float, str]:
    """
    Determine promotion recommendation based on criteria results.

    Args:
        criteria_results: List of evaluated criteria
        comparison: Optional comparison vs baseline

    Returns:
        Tuple of (recommendation, confidence, reason)
    """
    all_passed = all(cr.passed for cr in criteria_results)
    failed = [cr for cr in criteria_results if not cr.passed]

    # Check improvement if available
    improvement_met = True
    improvement_cr = next((cr for cr in criteria_results if cr.name == "improvement_threshold"), None)
    if improvement_cr:
        improvement_met = improvement_cr.passed

    if all_passed and improvement_met:
        return (
            PromotionRecommendation.PROMOTE,
            0.85,
            "All criteria passed with sufficient improvement"
        )
    elif all_passed and not improvement_met:
        return (
            PromotionRecommendation.REVIEW,
            0.60,
            "All criteria passed but improvement below threshold"
        )
    else:
        failed_names = ", ".join(cr.name for cr in failed)
        return (
            PromotionRecommendation.REJECT,
            0.90,
            f"Criteria failed: {failed_names}"
        )
