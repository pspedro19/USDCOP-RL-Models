"""
ExperimentContract - Contrato generado desde experiment YAML.
=============================================================
Contract ID: CTR-EXP-{experiment_name}
Version: 1.0.0

Este contrato es INMUTABLE una vez creado y contiene todos los hashes
necesarios para tracking de lineage completo desde YAML SSOT.

Usage:
    from src.core.contracts.experiment_contract import ExperimentContract

    # Create from YAML
    contract = ExperimentContract.from_yaml(Path("config/experiments/exp1.yaml"))

    # Save to database
    contract.save_to_db(conn)

    # Load from database
    contract = ExperimentContract.from_db(conn, contract_id="CTR-EXP-exp1")
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import hashlib
import json
import logging
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import SSOT feature order hash
from .feature_contract import FEATURE_ORDER_HASH, FEATURE_CONTRACT_VERSION

logger = logging.getLogger(__name__)

EXPERIMENT_CONTRACT_VERSION = "1.0.0"


class ExperimentContractError(ValueError):
    """Raised when experiment contract validation fails."""
    pass


@dataclass
class ExperimentContract:
    """
    Contrato inmutable generado desde un experiment YAML.

    Este contrato captura todos los hashes necesarios para reproducibilidad
    y tracking de lineage completo.
    """

    # Identity
    contract_id: str                    # CTR-EXP-{experiment_name}
    experiment_name: str
    experiment_version: str

    # Hashes para lineage
    config_hash: str                    # sha256(yaml_content)[:16]
    feature_order_hash: str             # from feature_contract.py
    reward_config_hash: str             # sha256(reward section)[:16]

    # Referencias a otros contratos
    feature_contract_version: str       # e.g., "v2.1.0"
    date_ranges_version: str            # e.g., "1.0.0"

    # Config congelada (inmutable)
    frozen_config: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    yaml_path: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Derived config sections (convenience)
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    reward_config: Dict[str, Any] = field(default_factory=dict)
    success_criteria: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExperimentContract":
        """
        Crear contrato desde archivo YAML.

        Args:
            yaml_path: Path al archivo experiment YAML

        Returns:
            ExperimentContract inmutable

        Raises:
            ExperimentContractError: Si YAML no es valido o faltan campos
        """
        if not YAML_AVAILABLE:
            raise ExperimentContractError("PyYAML is required for experiment contracts")

        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise ExperimentContractError(f"YAML file not found: {yaml_path}")

        # Read raw bytes for hash computation
        with open(yaml_path, 'rb') as f:
            yaml_bytes = f.read()

        # Parse YAML
        try:
            config = yaml.safe_load(yaml_bytes.decode('utf-8'))
        except yaml.YAMLError as e:
            raise ExperimentContractError(f"Invalid YAML: {e}")

        if config is None:
            raise ExperimentContractError("YAML file is empty")

        # Compute config hash (full file)
        config_hash = hashlib.sha256(yaml_bytes).hexdigest()[:16]

        # Extract experiment metadata
        exp_meta = config.get("experiment", {})
        exp_name = exp_meta.get("name")
        if not exp_name:
            # Try to infer from filename
            exp_name = yaml_path.stem
            logger.warning(f"No experiment.name in YAML, using filename: {exp_name}")

        exp_version = exp_meta.get("version", "1.0.0")

        # Extract and hash reward config
        reward_section = config.get("reward", config.get("training", {}).get("reward", {}))
        reward_json = json.dumps(reward_section, sort_keys=True, default=str)
        reward_hash = hashlib.sha256(reward_json.encode()).hexdigest()[:16]

        # Extract config sections
        model_config = config.get("model", config.get("agent", {}))
        training_config = config.get("training", {})
        success_criteria = config.get("evaluation", {}).get("success_criteria", {})

        # Get feature contract reference
        env_config = config.get("environment", {})
        feature_contract_id = env_config.get("feature_contract_id", FEATURE_CONTRACT_VERSION)

        return cls(
            contract_id=f"CTR-EXP-{exp_name}",
            experiment_name=exp_name,
            experiment_version=exp_version,
            config_hash=config_hash,
            feature_order_hash=FEATURE_ORDER_HASH,
            reward_config_hash=reward_hash,
            feature_contract_version=feature_contract_id,
            date_ranges_version="1.0.0",
            frozen_config=config,
            yaml_path=str(yaml_path),
            model_config=model_config,
            training_config=training_config,
            reward_config=reward_section,
            success_criteria=success_criteria,
        )

    @classmethod
    def from_db(cls, conn, contract_id: str) -> Optional["ExperimentContract"]:
        """
        Cargar contrato desde base de datos.

        Args:
            conn: Database connection
            contract_id: Contract ID (e.g., "CTR-EXP-exp1")

        Returns:
            ExperimentContract or None if not found
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT
                    contract_id, experiment_name, experiment_version,
                    config_hash, feature_order_hash, reward_config_hash,
                    frozen_config, created_at
                FROM experiment_contracts
                WHERE contract_id = %s
            """, (contract_id,))
            row = cur.fetchone()

            if not row:
                return None

            frozen_config = row[6]
            if isinstance(frozen_config, str):
                frozen_config = json.loads(frozen_config)

            return cls(
                contract_id=row[0],
                experiment_name=row[1],
                experiment_version=row[2],
                config_hash=row[3],
                feature_order_hash=row[4],
                reward_config_hash=row[5],
                feature_contract_version=frozen_config.get("environment", {}).get(
                    "feature_contract_id", FEATURE_CONTRACT_VERSION
                ),
                date_ranges_version="1.0.0",
                frozen_config=frozen_config,
                created_at=row[7],
                model_config=frozen_config.get("model", frozen_config.get("agent", {})),
                training_config=frozen_config.get("training", {}),
                reward_config=frozen_config.get("reward", {}),
                success_criteria=frozen_config.get("evaluation", {}).get("success_criteria", {}),
            )
        finally:
            cur.close()

    def save_to_db(self, conn) -> Optional[int]:
        """
        Guardar contrato en base de datos (inmutable - no actualiza si existe).

        Args:
            conn: Database connection

        Returns:
            ID of inserted row, or None if already exists
        """
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO experiment_contracts (
                    contract_id, experiment_name, experiment_version,
                    config_hash, feature_order_hash, reward_config_hash,
                    frozen_config, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (contract_id) DO NOTHING
                RETURNING id
            """, (
                self.contract_id,
                self.experiment_name,
                self.experiment_version,
                self.config_hash,
                self.feature_order_hash,
                self.reward_config_hash,
                json.dumps(self.frozen_config),
                self.created_at,
            ))
            result = cur.fetchone()
            conn.commit()

            if result:
                logger.info(f"Saved ExperimentContract: {self.contract_id}")
                return result[0]
            else:
                logger.info(f"ExperimentContract already exists: {self.contract_id}")
                return None

        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving ExperimentContract: {e}")
            raise
        finally:
            cur.close()

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get model hyperparameters from frozen config."""
        return {
            "learning_rate": self.model_config.get("learning_rate", 3e-4),
            "batch_size": self.training_config.get("batch_size", 64),
            "n_steps": self.training_config.get("n_steps", 2048),
            "gamma": self.model_config.get("gamma", 0.95),
            "gae_lambda": self.model_config.get("gae_lambda", 0.95),
            "clip_range": self.model_config.get("clip_range", 0.2),
            "ent_coef": self.model_config.get("ent_coef", 0.01),
            "vf_coef": self.model_config.get("vf_coef", 0.5),
            "max_grad_norm": self.model_config.get("max_grad_norm", 0.5),
        }

    def get_reward_weights(self) -> Dict[str, float]:
        """Get reward weights from frozen config."""
        weights = self.reward_config.get("weights", {})
        return {
            "pnl": weights.get("pnl", 1.0),
            "drawdown": weights.get("drawdown", -0.5),
            "holding": weights.get("holding", -0.001),
            "transaction": weights.get("transaction", -0.0005),
        }

    def get_success_criteria(self) -> Dict[str, float]:
        """Get success criteria for L4 evaluation."""
        return {
            "min_sharpe": self.success_criteria.get("min_sharpe", 0.5),
            "max_drawdown": self.success_criteria.get("max_drawdown", 0.15),
            "min_win_rate": self.success_criteria.get("min_win_rate", 0.45),
            "min_trades": self.success_criteria.get("min_trades", 50),
            "improvement_threshold": self.success_criteria.get("improvement_threshold", 0.05),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización."""
        return {
            "contract_id": self.contract_id,
            "experiment_name": self.experiment_name,
            "experiment_version": self.experiment_version,
            "config_hash": self.config_hash,
            "feature_order_hash": self.feature_order_hash,
            "reward_config_hash": self.reward_config_hash,
            "feature_contract_version": self.feature_contract_version,
            "date_ranges_version": self.date_ranges_version,
            "frozen_config": self.frozen_config,
            "yaml_path": self.yaml_path,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validar integridad del contrato.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if not self.contract_id:
            errors.append("contract_id is required")

        if not self.experiment_name:
            errors.append("experiment_name is required")

        if not self.config_hash or len(self.config_hash) != 16:
            errors.append("config_hash must be 16 characters")

        if not self.feature_order_hash:
            errors.append("feature_order_hash is required")

        if not self.frozen_config:
            errors.append("frozen_config cannot be empty")

        return len(errors) == 0, errors

    def __hash__(self):
        """Hash based on config_hash for deduplication."""
        return hash(self.config_hash)

    def __eq__(self, other):
        """Two contracts are equal if they have the same config_hash."""
        if not isinstance(other, ExperimentContract):
            return False
        return self.config_hash == other.config_hash


def load_experiment_contract(
    yaml_path: Optional[Path] = None,
    contract_id: Optional[str] = None,
    conn=None,
) -> Optional[ExperimentContract]:
    """
    Convenience function to load an experiment contract.

    Args:
        yaml_path: Path to YAML file (creates new contract)
        contract_id: Contract ID to load from DB
        conn: Database connection (required if loading from DB)

    Returns:
        ExperimentContract or None
    """
    if yaml_path:
        return ExperimentContract.from_yaml(yaml_path)
    elif contract_id and conn:
        return ExperimentContract.from_db(conn, contract_id)
    else:
        raise ValueError("Either yaml_path or (contract_id + conn) is required")
