"""
ProductionContract - Contrato del modelo actualmente en producción.
===================================================================
Contract ID: PROD-{model_id}
Version: 1.0.0

L1 y L5 leen este contrato para saber qué modelo usar y con qué
configuración (norm_stats, feature_order, etc.).

Este contrato representa el modelo que ha pasado ambos votos:
1. Primer voto: L4 Backtest + Promotion Proposal (recomendación automática)
2. Segundo voto: Dashboard Approval (aprobación humana)

Usage:
    from src.core.contracts.production_contract import ProductionContract

    # Load current production model
    contract = ProductionContract.get_active(conn)

    # Validate feature order matches
    if contract.validate_feature_order(current_hash):
        # Safe to run inference

    # Load specific model
    contract = ProductionContract.from_db(conn, model_id="exp1_v1_20260131")
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import logging
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

PRODUCTION_CONTRACT_VERSION = "1.0.0"


class ProductionContractError(ValueError):
    """Raised when production contract validation fails."""
    pass


@dataclass
class ProductionContract:
    """
    Contrato del modelo en producción (aprobado con ambos votos).

    Este contrato contiene toda la información necesaria para:
    1. L1: Validar que feature_order coincide con el modelo entrenado
    2. L5: Cargar el modelo correcto con su norm_stats
    3. Auditoría: Tracking completo del lineage y aprobación
    """

    # Model identity
    model_id: str
    experiment_name: str
    model_path: str

    # Hashes para validación
    model_hash: str
    config_hash: str
    feature_order_hash: str
    norm_stats_hash: str
    dataset_hash: str

    # Paths a artifacts
    norm_stats_path: str
    config_path: Optional[str] = None

    # Approval info (segundo voto)
    l4_proposal_id: str = ""
    l4_recommendation: str = ""
    l4_confidence: float = 0.0
    approved_by: str = ""
    approved_at: Optional[datetime] = None

    # Metrics from L4 backtest
    metrics: Dict[str, Any] = field(default_factory=dict)

    # Lineage completo
    lineage: Dict[str, Any] = field(default_factory=dict)

    # Status
    is_active: bool = True
    stage: str = "production"  # production, archived, canary

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    deployed_at: datetime = field(default_factory=datetime.utcnow)
    archived_at: Optional[datetime] = None

    @classmethod
    def get_active(cls, conn) -> Optional["ProductionContract"]:
        """
        Cargar el contrato de producción activo.

        Args:
            conn: Database connection

        Returns:
            ProductionContract or None if no active model
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT
                    model_id, experiment_name, model_path,
                    model_hash, config_hash, feature_order_hash,
                    norm_stats_hash, dataset_hash, norm_stats_path,
                    l4_proposal_id, l4_recommendation, l4_confidence,
                    approved_by, approved_at, metrics, lineage,
                    stage, created_at, promoted_at
                FROM model_registry
                WHERE stage = 'production' AND is_active = TRUE
                ORDER BY promoted_at DESC
                LIMIT 1
            """)
            row = cur.fetchone()

            if not row:
                logger.warning("No active production model found")
                return None

            return cls._from_row(row)

        except Exception as e:
            logger.error(f"Error loading production contract: {e}")
            return None
        finally:
            cur.close()

    @classmethod
    def from_db(cls, conn, model_id: str) -> Optional["ProductionContract"]:
        """
        Cargar contrato desde DB por model_id.

        Args:
            conn: Database connection
            model_id: Model ID to load

        Returns:
            ProductionContract or None if not found
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT
                    model_id, experiment_name, model_path,
                    model_hash, config_hash, feature_order_hash,
                    norm_stats_hash, dataset_hash, norm_stats_path,
                    l4_proposal_id, l4_recommendation, l4_confidence,
                    approved_by, approved_at, metrics, lineage,
                    stage, created_at, promoted_at
                FROM model_registry
                WHERE model_id = %s
            """, (model_id,))
            row = cur.fetchone()

            if not row:
                return None

            return cls._from_row(row)

        finally:
            cur.close()

    @classmethod
    def _from_row(cls, row) -> "ProductionContract":
        """Create contract from database row."""
        metrics = row[14]
        if isinstance(metrics, str):
            metrics = json.loads(metrics) if metrics else {}

        lineage = row[15]
        if isinstance(lineage, str):
            lineage = json.loads(lineage) if lineage else {}

        return cls(
            model_id=row[0],
            experiment_name=row[1],
            model_path=row[2],
            model_hash=row[3] or "",
            config_hash=row[4] or "",
            feature_order_hash=row[5] or "",
            norm_stats_hash=row[6] or "",
            dataset_hash=row[7] or "",
            norm_stats_path=row[8] or "",
            l4_proposal_id=row[9] or "",
            l4_recommendation=row[10] or "",
            l4_confidence=float(row[11]) if row[11] else 0.0,
            approved_by=row[12] or "",
            approved_at=row[13],
            metrics=metrics,
            lineage=lineage,
            stage=row[16] or "production",
            created_at=row[17] or datetime.utcnow(),
            deployed_at=row[18] or datetime.utcnow(),
            is_active=row[16] == "production",
        )

    def validate_feature_order(self, current_hash: str) -> bool:
        """
        Validar que feature_order_hash coincide con el actual.

        Args:
            current_hash: Hash from current FEATURE_ORDER

        Returns:
            True if hashes match
        """
        if not self.feature_order_hash:
            logger.warning("Production contract has no feature_order_hash")
            return True  # Allow if not set

        matches = self.feature_order_hash == current_hash
        if not matches:
            logger.error(
                f"FEATURE_ORDER_HASH mismatch: "
                f"production={self.feature_order_hash}, "
                f"current={current_hash}"
            )
        return matches

    def validate_norm_stats(self, norm_stats_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Validar que norm_stats.json coincide con el hash esperado.

        Args:
            norm_stats_path: Path to norm_stats.json (uses self.norm_stats_path if None)

        Returns:
            Tuple of (is_valid, message)
        """
        path = norm_stats_path or self.norm_stats_path
        if not path:
            return True, "No norm_stats_path configured"

        if not self.norm_stats_hash:
            return True, "No expected hash (validation skipped)"

        try:
            with open(path, 'rb') as f:
                content = f.read()
                current_hash = hashlib.sha256(content).hexdigest()[:16]

            if current_hash != self.norm_stats_hash:
                return False, (
                    f"Hash mismatch: expected={self.norm_stats_hash}, "
                    f"current={current_hash}"
                )

            return True, f"norm_stats validated: {current_hash}"

        except FileNotFoundError:
            return False, f"norm_stats.json not found: {path}"
        except Exception as e:
            return False, f"Error validating norm_stats: {e}"

    def get_model_path(self) -> Path:
        """Get Path object for model file."""
        return Path(self.model_path)

    def get_norm_stats_path(self) -> Path:
        """Get Path object for norm_stats file."""
        return Path(self.norm_stats_path)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for API."""
        return {
            "model_id": self.model_id,
            "experiment_name": self.experiment_name,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "config_hash": self.config_hash,
            "feature_order_hash": self.feature_order_hash,
            "norm_stats_hash": self.norm_stats_hash,
            "dataset_hash": self.dataset_hash,
            "norm_stats_path": self.norm_stats_path,
            "l4_proposal_id": self.l4_proposal_id,
            "l4_recommendation": self.l4_recommendation,
            "l4_confidence": self.l4_confidence,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "metrics": self.metrics,
            "lineage": self.lineage,
            "is_active": self.is_active,
            "stage": self.stage,
            "deployed_at": self.deployed_at.isoformat() if self.deployed_at else None,
        }

    def validate_all(self, current_feature_hash: str) -> Tuple[bool, List[str]]:
        """
        Run all validations.

        Args:
            current_feature_hash: Current FEATURE_ORDER_HASH

        Returns:
            Tuple of (all_valid, list of error messages)
        """
        errors = []

        # Validate feature order
        if not self.validate_feature_order(current_feature_hash):
            errors.append(
                f"Feature order mismatch: {self.feature_order_hash} != {current_feature_hash}"
            )

        # Validate norm_stats
        is_valid, msg = self.validate_norm_stats()
        if not is_valid:
            errors.append(f"Norm stats validation failed: {msg}")

        # Validate model file exists
        if self.model_path and not Path(self.model_path).exists():
            errors.append(f"Model file not found: {self.model_path}")

        return len(errors) == 0, errors


def get_production_contract(conn) -> Optional[ProductionContract]:
    """
    Convenience function to get the active production contract.

    Args:
        conn: Database connection

    Returns:
        ProductionContract or None
    """
    return ProductionContract.get_active(conn)


def promote_model_to_production(
    conn,
    model_id: str,
    proposal_id: str,
    approved_by: str,
) -> bool:
    """
    Promote a model to production after human approval.

    This function:
    1. Archives the current production model
    2. Updates the new model to production stage
    3. Logs the approval in audit_log

    Args:
        conn: Database connection
        model_id: Model to promote
        proposal_id: Promotion proposal ID
        approved_by: Email of approver

    Returns:
        True if successful
    """
    cur = conn.cursor()
    try:
        # Get current production model
        cur.execute("""
            SELECT model_id FROM model_registry
            WHERE stage = 'production' AND is_active = TRUE
        """)
        current_prod = cur.fetchone()
        previous_model_id = current_prod[0] if current_prod else None

        # Archive current production
        if previous_model_id:
            cur.execute("""
                UPDATE model_registry
                SET stage = 'archived', is_active = FALSE, archived_at = NOW()
                WHERE model_id = %s
            """, (previous_model_id,))

        # Promote new model
        cur.execute("""
            UPDATE model_registry
            SET stage = 'production',
                is_active = TRUE,
                approved_by = %s,
                approved_at = NOW(),
                promoted_at = NOW()
            WHERE model_id = %s
        """, (approved_by, model_id))

        # Update proposal status
        cur.execute("""
            UPDATE promotion_proposals
            SET status = 'APPROVED',
                reviewer = %s,
                reviewed_at = NOW()
            WHERE proposal_id = %s
        """, (approved_by, proposal_id))

        # Log to audit
        cur.execute("""
            INSERT INTO approval_audit_log
            (action, model_id, proposal_id, reviewer, previous_production_model)
            VALUES ('APPROVE', %s, %s, %s, %s)
        """, (model_id, proposal_id, approved_by, previous_model_id))

        conn.commit()
        logger.info(f"Promoted model {model_id} to production by {approved_by}")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Error promoting model: {e}")
        raise
    finally:
        cur.close()


def archive_production_model(conn, model_id: str, reason: str = "manual") -> bool:
    """
    Archive a production model (rollback).

    Args:
        conn: Database connection
        model_id: Model to archive
        reason: Reason for archiving

    Returns:
        True if successful
    """
    cur = conn.cursor()
    try:
        cur.execute("""
            UPDATE model_registry
            SET stage = 'archived',
                is_active = FALSE,
                archived_at = NOW()
            WHERE model_id = %s AND stage = 'production'
        """, (model_id,))

        if cur.rowcount == 0:
            logger.warning(f"Model {model_id} not in production, cannot archive")
            return False

        # Log to audit
        cur.execute("""
            INSERT INTO approval_audit_log
            (action, model_id, reviewer, notes)
            VALUES ('ARCHIVE', %s, 'system', %s)
        """, (model_id, reason))

        conn.commit()
        logger.info(f"Archived production model {model_id}: {reason}")
        return True

    except Exception as e:
        conn.rollback()
        logger.error(f"Error archiving model: {e}")
        raise
    finally:
        cur.close()
