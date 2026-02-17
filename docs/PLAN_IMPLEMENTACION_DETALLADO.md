# Plan de Implementación Detallado: L0-L5 Pipeline Completo

**Contract: CTR-IMPL-001**
**Version: 1.0.0**
**Date: 2026-01-31**

---

## Resumen de Cambios

Este documento detalla exactamente qué archivos crear, modificar y eliminar para implementar la arquitectura completa L0-L5 con:

1. **Contratos versionados** desde YAMLs SSOT
2. **L4 Backtest + Promotion** (primer voto)
3. **Dashboard Approval** (segundo voto)
4. **L1 Production Inference** que lee el contrato aprobado
5. **Lineage completo** de datos

---

## 1. ARCHIVOS A CREAR (NEW)

### 1.1 Contratos y Servicios Core

```
src/contracts/
├── experiment_contract.py          # NEW: ExperimentContract desde YAML
├── promotion_contract.py           # NEW: Contrato de promoción L4→Dashboard
└── production_contract.py          # NEW: Contrato del modelo en producción

src/services/
├── promotion_service.py            # NEW: Lógica de promoción y evaluación
├── contract_generator.py           # NEW: Genera contracts desde YAMLs
└── lineage_service.py              # NEW: Tracking de lineage completo
```

#### 1.1.1 `src/contracts/experiment_contract.py`

```python
"""
ExperimentContract - Contrato generado desde experiment YAML.

Este contrato es INMUTABLE una vez creado y contiene todos los hashes
necesarios para tracking de lineage completo.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import hashlib
import json
import yaml
from datetime import datetime

# Import SSOT
from src.core.contracts.feature_contract import FEATURE_ORDER_HASH

@dataclass
class ExperimentContract:
    """Contrato inmutable generado desde un experiment YAML."""

    # Identity
    contract_id: str                    # CTR-EXP-{experiment_name}
    experiment_name: str
    experiment_version: str

    # Hashes para lineage
    config_hash: str                    # sha256(yaml_content)
    feature_order_hash: str             # from feature_contract.py
    reward_config_hash: str             # sha256(reward section)

    # Referencias a otros contratos
    feature_contract_version: str       # e.g., "v2.1.0"
    date_ranges_version: str            # e.g., "1.0.0"

    # Config congelada
    frozen_config: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "ExperimentContract":
        """Crear contrato desde archivo YAML."""
        with open(yaml_path, 'rb') as f:
            yaml_bytes = f.read()

        config = yaml.safe_load(yaml_bytes.decode('utf-8'))

        # Compute hashes
        config_hash = hashlib.sha256(yaml_bytes).hexdigest()[:16]

        reward_section = json.dumps(
            config.get("reward", config.get("training", {}).get("reward", {})),
            sort_keys=True
        )
        reward_hash = hashlib.sha256(reward_section.encode()).hexdigest()[:16]

        exp_meta = config.get("experiment", {})

        return cls(
            contract_id=f"CTR-EXP-{exp_meta.get('name', 'unknown')}",
            experiment_name=exp_meta.get("name", "unknown"),
            experiment_version=exp_meta.get("version", "1.0.0"),
            config_hash=config_hash,
            feature_order_hash=FEATURE_ORDER_HASH,
            reward_config_hash=reward_hash,
            feature_contract_version=config.get("environment", {}).get(
                "feature_contract_id", "v1.0.0"
            ),
            date_ranges_version="1.0.0",
            frozen_config=config,
        )

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
            "created_at": self.created_at.isoformat(),
        }

    def save_to_db(self, conn) -> int:
        """Guardar contrato en base de datos (inmutable)."""
        cur = conn.cursor()
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
        return result[0] if result else None
```

#### 1.1.2 `src/contracts/promotion_contract.py`

```python
"""
PromotionContract - Contrato de propuesta de promoción L4→Dashboard.

Generado por L4 después del backtest out-of-sample.
Requiere aprobación humana en Dashboard (segundo voto).
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import json

class PromotionRecommendation(Enum):
    PROMOTE = "PROMOTE"
    REJECT = "REJECT"
    REVIEW = "REVIEW"

class PromotionStatus(Enum):
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

@dataclass
class BacktestMetrics:
    """Métricas del backtest out-of-sample."""
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    test_period_start: str
    test_period_end: str

@dataclass
class CriteriaResult:
    """Resultado de evaluación de un criterio."""
    name: str
    threshold: float
    actual: float
    passed: bool

    @property
    def status_str(self) -> str:
        return "PASS" if self.passed else "FAIL"

@dataclass
class PromotionProposal:
    """Propuesta de promoción generada por L4."""

    # Identity
    proposal_id: str
    model_id: str
    experiment_name: str

    # Recommendation (primer voto - L4)
    recommendation: PromotionRecommendation
    confidence: float
    reason: str

    # Métricas del backtest
    metrics: BacktestMetrics

    # Comparación vs baseline
    vs_baseline: Optional[Dict[str, float]] = None
    baseline_model_id: Optional[str] = None

    # Resultados de criterios
    criteria_results: List[CriteriaResult] = field(default_factory=list)

    # Lineage
    lineage: Dict[str, str] = field(default_factory=dict)

    # Status
    status: PromotionStatus = PromotionStatus.PENDING_APPROVAL
    requires_human_approval: bool = True

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewer: Optional[str] = None
    reviewer_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a dict para API/DB."""
        return {
            "proposal_id": self.proposal_id,
            "model_id": self.model_id,
            "experiment_name": self.experiment_name,
            "recommendation": self.recommendation.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "metrics": {
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "max_drawdown": self.metrics.max_drawdown,
                "win_rate": self.metrics.win_rate,
                "profit_factor": self.metrics.profit_factor,
                "total_trades": self.metrics.total_trades,
                "avg_trade_pnl": self.metrics.avg_trade_pnl,
                "test_period": f"{self.metrics.test_period_start} to {self.metrics.test_period_end}",
            },
            "vs_baseline": self.vs_baseline,
            "baseline_model_id": self.baseline_model_id,
            "criteria_results": {
                cr.name: f"{cr.status_str} ({cr.actual:.4f} vs {cr.threshold:.4f})"
                for cr in self.criteria_results
            },
            "lineage": self.lineage,
            "status": self.status.value,
            "requires_human_approval": self.requires_human_approval,
            "created_at": self.created_at.isoformat(),
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "reviewer": self.reviewer,
            "reviewer_notes": self.reviewer_notes,
        }

    def save_to_db(self, conn) -> int:
        """Guardar proposal en base de datos."""
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO promotion_proposals (
                proposal_id, model_id, experiment_name,
                recommendation, confidence, reason,
                metrics, vs_baseline, criteria_results,
                lineage, status, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (
            self.proposal_id,
            self.model_id,
            self.experiment_name,
            self.recommendation.value,
            self.confidence,
            self.reason,
            json.dumps(self.to_dict()["metrics"]),
            json.dumps(self.vs_baseline) if self.vs_baseline else None,
            json.dumps(self.to_dict()["criteria_results"]),
            json.dumps(self.lineage),
            self.status.value,
            self.created_at,
        ))
        result = cur.fetchone()
        conn.commit()
        return result[0]
```

#### 1.1.3 `src/contracts/production_contract.py`

```python
"""
ProductionContract - Contrato del modelo actualmente en producción.

L1 y L5 leen este contrato para saber qué modelo usar y con qué
configuración (norm_stats, feature_order, etc.).
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import json

@dataclass
class ProductionContract:
    """Contrato del modelo en producción (aprobado con 2 votos)."""

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
    config_path: str

    # Approval info
    l4_proposal_id: str
    l4_recommendation: str
    l4_confidence: float
    approved_by: str
    approved_at: datetime

    # Lineage completo
    lineage: Dict[str, Any] = field(default_factory=dict)

    # Status
    is_active: bool = True
    deployed_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_db(cls, conn, model_id: Optional[str] = None) -> Optional["ProductionContract"]:
        """Cargar contrato de producción desde DB.

        Si model_id es None, carga el modelo activo actual.
        """
        cur = conn.cursor()

        if model_id:
            query = """
                SELECT * FROM model_registry
                WHERE model_id = %s AND stage = 'production'
            """
            cur.execute(query, (model_id,))
        else:
            query = """
                SELECT * FROM model_registry
                WHERE stage = 'production' AND is_active = TRUE
                ORDER BY promoted_at DESC LIMIT 1
            """
            cur.execute(query)

        row = cur.fetchone()
        if not row:
            return None

        # Construir contrato desde row
        return cls(
            model_id=row[1],  # model_id
            experiment_name=row[2],  # experiment_name
            model_path=row[3],  # model_path
            model_hash=row[4],  # model_hash
            config_hash=row[6],  # config_hash
            feature_order_hash=row[7],  # feature_order_hash
            norm_stats_hash=row[5],  # norm_stats_hash
            dataset_hash=row[8],  # dataset_hash
            norm_stats_path=row[9] if len(row) > 9 else "",
            config_path=row[10] if len(row) > 10 else "",
            l4_proposal_id=row[11] if len(row) > 11 else "",
            l4_recommendation=row[12] if len(row) > 12 else "",
            l4_confidence=float(row[13]) if len(row) > 13 and row[13] else 0.0,
            approved_by=row[14] if len(row) > 14 else "",
            approved_at=row[15] if len(row) > 15 else datetime.utcnow(),
            lineage=json.loads(row[16]) if len(row) > 16 and row[16] else {},
            is_active=True,
            deployed_at=row[17] if len(row) > 17 else datetime.utcnow(),
        )

    def validate_feature_order(self, current_hash: str) -> bool:
        """Validar que feature_order_hash coincide con el actual."""
        return self.feature_order_hash == current_hash

    def validate_norm_stats(self, current_hash: str) -> bool:
        """Validar que norm_stats_hash coincide con el actual."""
        return self.norm_stats_hash == current_hash

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a dict para API."""
        return {
            "model_id": self.model_id,
            "experiment_name": self.experiment_name,
            "model_path": self.model_path,
            "model_hash": self.model_hash,
            "config_hash": self.config_hash,
            "feature_order_hash": self.feature_order_hash,
            "norm_stats_hash": self.norm_stats_hash,
            "dataset_hash": self.dataset_hash,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat(),
            "lineage": self.lineage,
            "is_active": self.is_active,
        }
```

### 1.2 DAGs Nuevos

```
airflow/dags/
├── l4_backtest_promotion.py        # NEW: Fusión de backtest + promotion
└── services/
    └── promotion_service.py        # NEW: Service para L4
```

#### 1.2.1 `airflow/dags/l4_backtest_promotion.py`

```python
"""
L4: Backtest + Promotion Proposal DAG
=====================================
Primer Voto del sistema de promoción de doble voto.

Este DAG:
1. Recibe modelo entrenado de L3
2. Ejecuta backtest out-of-sample en test.parquet (2025-07-01 → HOY)
3. Evalúa success_criteria del experiment.yaml
4. Compara vs baseline (modelo en producción actual)
5. Genera promotion_proposal con recomendación
6. SIEMPRE requiere aprobación humana (segundo voto en Dashboard)

Schedule: Trigger después de L3 completado exitosamente
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import hashlib
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from contracts.dag_registry import (
    RL_L4_BACKTEST_PROMOTION,
    RL_L3_MODEL_TRAINING,
)
from contracts.xcom_contracts import L3Output
from utils.dag_common import get_db_connection

# Import contracts
from src.contracts.experiment_contract import ExperimentContract
from src.contracts.promotion_contract import (
    PromotionProposal,
    PromotionRecommendation,
    PromotionStatus,
    BacktestMetrics,
    CriteriaResult,
)

logger = logging.getLogger(__name__)
DAG_ID = RL_L4_BACKTEST_PROMOTION

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """Motor de backtest para evaluación out-of-sample."""

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 75.0,  # USDCOP spread
        position_size: float = 1.0,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost_bps / 10_000
        self.position_size = position_size

    def run(
        self,
        model_path: str,
        test_data_path: str,
        norm_stats_path: str,
    ) -> BacktestMetrics:
        """Ejecutar backtest en datos out-of-sample."""
        import pandas as pd
        import numpy as np
        from stable_baselines3 import PPO

        # Load model
        model = PPO.load(model_path)

        # Load test data
        df = pd.read_parquet(test_data_path)

        # Load norm stats
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        # Initialize tracking
        capital = self.initial_capital
        peak_capital = capital
        trades = []
        position = 0
        entry_price = 0

        # Simulate
        for idx in range(len(df)):
            row = df.iloc[idx]

            # Build observation (15 features)
            obs = self._build_observation(row, position, idx, len(df))

            # Normalize
            obs = self._normalize(obs, norm_stats)

            # Get action
            action, _ = model.predict(obs, deterministic=True)
            signal = self._discretize(float(action))

            close_price = row.get('close', row.get('Close', 4200.0))

            # Execute trade
            if signal != position:
                # Close existing position
                if position != 0:
                    pnl = (close_price - entry_price) * position
                    pnl -= abs(pnl) * self.transaction_cost
                    capital += pnl
                    trades.append({
                        'pnl': pnl,
                        'pnl_pct': pnl / self.initial_capital,
                        'direction': 'LONG' if position > 0 else 'SHORT',
                    })

                # Open new position
                if signal != 0:
                    entry_price = close_price
                    position = signal
                else:
                    position = 0

                peak_capital = max(peak_capital, capital)

        # Calculate metrics
        returns = [t['pnl_pct'] for t in trades]
        winning = [r for r in returns if r > 0]
        losing = [r for r in returns if r < 0]

        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        max_dd = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
        win_rate = len(winning) / len(trades) if trades else 0
        profit_factor = abs(sum(winning) / sum(losing)) if losing and sum(losing) != 0 else 999

        # Get test period
        test_start = str(df.index[0] if hasattr(df.index[0], 'date') else df.iloc[0].get('timestamp', ''))
        test_end = str(df.index[-1] if hasattr(df.index[-1], 'date') else df.iloc[-1].get('timestamp', ''))

        return BacktestMetrics(
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_pnl=np.mean(returns) if returns else 0,
            test_period_start=test_start,
            test_period_end=test_end,
        )

    def _build_observation(self, row, position: float, idx: int, total: int) -> np.ndarray:
        """Build 15-dim observation from row."""
        import numpy as np
        from src.core.contracts.feature_contract import FEATURE_ORDER

        obs = []
        for feature in FEATURE_ORDER[:-2]:  # Exclude state features
            obs.append(float(row.get(feature, 0.0)))

        # Add state features
        obs.append(position)
        obs.append(idx / total)  # time_normalized

        return np.array(obs, dtype=np.float32)

    def _normalize(self, obs, norm_stats: Dict) -> np.ndarray:
        """Apply normalization using stored stats."""
        import numpy as np
        from src.core.contracts.feature_contract import FEATURE_ORDER

        normalized = []
        for i, feature in enumerate(FEATURE_ORDER):
            if feature in norm_stats:
                mean = norm_stats[feature].get('mean', 0)
                std = norm_stats[feature].get('std', 1)
                normalized.append((obs[i] - mean) / std if std > 0 else 0)
            else:
                normalized.append(obs[i])

        return np.array(normalized, dtype=np.float32)

    def _discretize(self, action: float) -> int:
        """Convert continuous action to discrete position."""
        if action > 0.33:
            return 1  # LONG
        elif action < -0.33:
            return -1  # SHORT
        return 0  # HOLD


# ============================================================================
# DAG TASKS
# ============================================================================

def load_l3_output(**context) -> Dict[str, Any]:
    """Task 1: Cargar output de L3 vía XCom."""
    ti = context['ti']

    # Pull L3 output
    l3_output = L3Output.pull_from_xcom(ti, dag_id=RL_L3_MODEL_TRAINING)

    if l3_output is None:
        raise ValueError("No L3 output found in XCom")

    logger.info(f"Loaded L3 output: model_path={l3_output.model_path}")
    logger.info(f"  model_hash={l3_output.model_hash}")
    logger.info(f"  dataset_hash={l3_output.dataset_hash}")
    logger.info(f"  config_hash={l3_output.config_hash}")

    return l3_output.to_dict()


def run_oos_backtest(**context) -> Dict[str, Any]:
    """Task 2: Ejecutar backtest out-of-sample."""
    ti = context['ti']
    l3_data = ti.xcom_pull(task_ids='load_l3_output')

    experiment_name = l3_data.get('experiment_name', 'default')

    # Paths
    model_path = l3_data['model_path']
    test_data_path = f"data/pipeline/07_output/5min/DS_{experiment_name}_test.parquet"
    norm_stats_path = f"data/pipeline/07_output/5min/DS_{experiment_name}_norm_stats.json"

    # Run backtest
    engine = BacktestEngine()
    metrics = engine.run(model_path, test_data_path, norm_stats_path)

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS (Out-of-Sample)")
    logger.info("=" * 60)
    logger.info(f"  Test Period: {metrics.test_period_start} to {metrics.test_period_end}")
    logger.info(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    logger.info(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
    logger.info(f"  Win Rate: {metrics.win_rate:.2%}")
    logger.info(f"  Profit Factor: {metrics.profit_factor:.2f}")
    logger.info(f"  Total Trades: {metrics.total_trades}")
    logger.info("=" * 60)

    ti.xcom_push(key='backtest_metrics', value={
        'sharpe_ratio': metrics.sharpe_ratio,
        'max_drawdown': metrics.max_drawdown,
        'win_rate': metrics.win_rate,
        'profit_factor': metrics.profit_factor,
        'total_trades': metrics.total_trades,
        'avg_trade_pnl': metrics.avg_trade_pnl,
        'test_period_start': metrics.test_period_start,
        'test_period_end': metrics.test_period_end,
    })

    return {'status': 'success', 'sharpe': metrics.sharpe_ratio}


def compare_vs_baseline(**context) -> Dict[str, Any]:
    """Task 3: Comparar vs modelo baseline (producción actual)."""
    ti = context['ti']
    l3_data = ti.xcom_pull(task_ids='load_l3_output')
    new_metrics = ti.xcom_pull(task_ids='run_oos_backtest', key='backtest_metrics')

    conn = get_db_connection()
    cur = conn.cursor()

    # Get baseline model (current production)
    cur.execute("""
        SELECT model_id, model_path, metrics
        FROM model_registry
        WHERE stage = 'production'
        ORDER BY promoted_at DESC LIMIT 1
    """)
    baseline_row = cur.fetchone()

    if not baseline_row:
        logger.info("No baseline model found - this is the first model")
        ti.xcom_push(key='vs_baseline', value=None)
        return {'has_baseline': False}

    baseline_id = baseline_row[0]
    baseline_metrics = json.loads(baseline_row[2]) if baseline_row[2] else {}

    # Calculate improvements
    baseline_sharpe = baseline_metrics.get('sharpe_ratio', 0)
    baseline_dd = baseline_metrics.get('max_drawdown', 1)

    sharpe_improvement = (
        (new_metrics['sharpe_ratio'] - baseline_sharpe) / baseline_sharpe
        if baseline_sharpe > 0 else 1.0
    )
    dd_improvement = (
        (baseline_dd - new_metrics['max_drawdown']) / baseline_dd
        if baseline_dd > 0 else 0.0
    )

    comparison = {
        'baseline_model_id': baseline_id,
        'sharpe_improvement': sharpe_improvement,
        'drawdown_improvement': dd_improvement,
        'baseline_sharpe': baseline_sharpe,
        'baseline_max_dd': baseline_dd,
    }

    logger.info(f"Comparison vs baseline ({baseline_id}):")
    logger.info(f"  Sharpe: {new_metrics['sharpe_ratio']:.3f} vs {baseline_sharpe:.3f} ({sharpe_improvement:+.1%})")
    logger.info(f"  MaxDD: {new_metrics['max_drawdown']:.2%} vs {baseline_dd:.2%} ({dd_improvement:+.1%})")

    ti.xcom_push(key='vs_baseline', value=comparison)
    cur.close()
    conn.close()

    return {'has_baseline': True, 'baseline_id': baseline_id}


def evaluate_criteria(**context) -> Dict[str, Any]:
    """Task 4: Evaluar success_criteria del experiment.yaml."""
    ti = context['ti']
    l3_data = ti.xcom_pull(task_ids='load_l3_output')
    metrics = ti.xcom_pull(task_ids='run_oos_backtest', key='backtest_metrics')
    comparison = ti.xcom_pull(task_ids='compare_vs_baseline', key='vs_baseline')

    experiment_name = l3_data.get('experiment_name', 'default')

    # Load experiment YAML
    yaml_path = Path(f"config/experiments/{experiment_name}.yaml")
    if not yaml_path.exists():
        yaml_path = Path(f"config/experiments/exp1_curriculum_aggressive_v1.yaml")

    contract = ExperimentContract.from_yaml(yaml_path)
    success_criteria = contract.frozen_config.get('evaluation', {}).get('success_criteria', {})

    # Default criteria if not specified
    if not success_criteria:
        success_criteria = {
            'min_sharpe': 0.5,
            'max_drawdown': 0.15,
            'min_win_rate': 0.45,
            'min_trades': 50,
            'improvement_threshold': 0.05,
        }

    # Evaluate each criterion
    results = []
    all_passed = True

    # Sharpe
    cr = CriteriaResult(
        name='min_sharpe',
        threshold=success_criteria.get('min_sharpe', 0.5),
        actual=metrics['sharpe_ratio'],
        passed=metrics['sharpe_ratio'] >= success_criteria.get('min_sharpe', 0.5)
    )
    results.append(cr)
    all_passed = all_passed and cr.passed

    # Max Drawdown
    cr = CriteriaResult(
        name='max_drawdown',
        threshold=success_criteria.get('max_drawdown', 0.15),
        actual=metrics['max_drawdown'],
        passed=metrics['max_drawdown'] <= success_criteria.get('max_drawdown', 0.15)
    )
    results.append(cr)
    all_passed = all_passed and cr.passed

    # Win Rate
    cr = CriteriaResult(
        name='min_win_rate',
        threshold=success_criteria.get('min_win_rate', 0.45),
        actual=metrics['win_rate'],
        passed=metrics['win_rate'] >= success_criteria.get('min_win_rate', 0.45)
    )
    results.append(cr)
    all_passed = all_passed and cr.passed

    # Min Trades
    cr = CriteriaResult(
        name='min_trades',
        threshold=success_criteria.get('min_trades', 50),
        actual=float(metrics['total_trades']),
        passed=metrics['total_trades'] >= success_criteria.get('min_trades', 50)
    )
    results.append(cr)
    all_passed = all_passed and cr.passed

    # Improvement vs baseline (if exists)
    improvement_threshold = success_criteria.get('improvement_threshold', 0.05)
    if comparison:
        cr = CriteriaResult(
            name='improvement_threshold',
            threshold=improvement_threshold,
            actual=comparison['sharpe_improvement'],
            passed=comparison['sharpe_improvement'] >= improvement_threshold
        )
        results.append(cr)
        # Note: improvement is not required for all_passed if it's close

    logger.info("=" * 60)
    logger.info("CRITERIA EVALUATION")
    logger.info("=" * 60)
    for cr in results:
        status = "PASS" if cr.passed else "FAIL"
        logger.info(f"  {cr.name}: {status} ({cr.actual:.4f} vs {cr.threshold:.4f})")
    logger.info(f"  OVERALL: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    logger.info("=" * 60)

    ti.xcom_push(key='criteria_results', value=[
        {'name': cr.name, 'threshold': cr.threshold, 'actual': cr.actual, 'passed': cr.passed}
        for cr in results
    ])
    ti.xcom_push(key='all_criteria_passed', value=all_passed)

    return {'all_passed': all_passed, 'criteria_count': len(results)}


def generate_promotion_proposal(**context) -> Dict[str, Any]:
    """Task 5: Generar promotion proposal (PRIMER VOTO)."""
    ti = context['ti']
    l3_data = ti.xcom_pull(task_ids='load_l3_output')
    metrics = ti.xcom_pull(task_ids='run_oos_backtest', key='backtest_metrics')
    comparison = ti.xcom_pull(task_ids='compare_vs_baseline', key='vs_baseline')
    criteria_results = ti.xcom_pull(task_ids='evaluate_criteria', key='criteria_results')
    all_passed = ti.xcom_pull(task_ids='evaluate_criteria', key='all_criteria_passed')

    experiment_name = l3_data.get('experiment_name', 'default')
    model_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Determine recommendation
    improvement_met = True
    if comparison:
        improvement_met = comparison.get('sharpe_improvement', 0) >= 0.05

    if all_passed and improvement_met:
        recommendation = PromotionRecommendation.PROMOTE
        confidence = 0.85
        reason = f"All criteria passed. Sharpe {metrics['sharpe_ratio']:.2f}"
        if comparison:
            reason += f" (+{comparison['sharpe_improvement']:.0%} vs baseline)"
    elif all_passed and not improvement_met:
        recommendation = PromotionRecommendation.REVIEW
        confidence = 0.60
        reason = "All criteria passed but improvement below threshold"
    else:
        recommendation = PromotionRecommendation.REJECT
        confidence = 0.90
        failed = [cr['name'] for cr in criteria_results if not cr['passed']]
        reason = f"Criteria failed: {', '.join(failed)}"

    # Build lineage
    lineage = {
        'model_hash': l3_data.get('model_hash'),
        'dataset_hash': l3_data.get('dataset_hash'),
        'config_hash': l3_data.get('config_hash'),
        'test_period': f"{metrics['test_period_start']} to {metrics['test_period_end']}",
        'baseline_model_id': comparison.get('baseline_model_id') if comparison else None,
    }

    # Create proposal
    proposal = PromotionProposal(
        proposal_id=f"PROP-{model_id}",
        model_id=model_id,
        experiment_name=experiment_name,
        recommendation=recommendation,
        confidence=confidence,
        reason=reason,
        metrics=BacktestMetrics(
            sharpe_ratio=metrics['sharpe_ratio'],
            max_drawdown=metrics['max_drawdown'],
            win_rate=metrics['win_rate'],
            profit_factor=metrics['profit_factor'],
            total_trades=metrics['total_trades'],
            avg_trade_pnl=metrics['avg_trade_pnl'],
            test_period_start=metrics['test_period_start'],
            test_period_end=metrics['test_period_end'],
        ),
        vs_baseline=comparison,
        baseline_model_id=comparison.get('baseline_model_id') if comparison else None,
        criteria_results=[
            CriteriaResult(
                name=cr['name'],
                threshold=cr['threshold'],
                actual=cr['actual'],
                passed=cr['passed'],
            )
            for cr in criteria_results
        ],
        lineage=lineage,
        status=PromotionStatus.PENDING_APPROVAL,
        requires_human_approval=True,
    )

    # Save to database
    conn = get_db_connection()
    proposal.save_to_db(conn)
    conn.close()

    logger.info("=" * 60)
    logger.info("PROMOTION PROPOSAL GENERATED (PRIMER VOTO)")
    logger.info("=" * 60)
    logger.info(f"  Proposal ID: {proposal.proposal_id}")
    logger.info(f"  Model ID: {proposal.model_id}")
    logger.info(f"  Recommendation: {recommendation.value}")
    logger.info(f"  Confidence: {confidence:.0%}")
    logger.info(f"  Reason: {reason}")
    logger.info(f"  Status: PENDING_APPROVAL (requiere segundo voto en Dashboard)")
    logger.info("=" * 60)

    # TODO: Send notification to Slack/Dashboard

    ti.xcom_push(key='proposal', value=proposal.to_dict())

    return {
        'proposal_id': proposal.proposal_id,
        'recommendation': recommendation.value,
        'status': 'PENDING_APPROVAL',
    }


# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='L4: Backtest out-of-sample + Promotion proposal (Primer Voto)',
    schedule_interval=None,  # Triggered by L3
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l4', 'backtest', 'promotion', 'two-vote'],
)

with dag:

    # Wait for L3 to complete
    wait_l3 = ExternalTaskSensor(
        task_id='wait_for_l3',
        external_dag_id=RL_L3_MODEL_TRAINING,
        external_task_id='save_model_artifacts',
        mode='reschedule',
        timeout=3600,
        poke_interval=60,
    )

    # Load L3 output
    task_load = PythonOperator(
        task_id='load_l3_output',
        python_callable=load_l3_output,
        provide_context=True,
    )

    # Run out-of-sample backtest
    task_backtest = PythonOperator(
        task_id='run_oos_backtest',
        python_callable=run_oos_backtest,
        provide_context=True,
    )

    # Compare vs baseline
    task_compare = PythonOperator(
        task_id='compare_vs_baseline',
        python_callable=compare_vs_baseline,
        provide_context=True,
    )

    # Evaluate criteria
    task_evaluate = PythonOperator(
        task_id='evaluate_criteria',
        python_callable=evaluate_criteria,
        provide_context=True,
    )

    # Generate promotion proposal
    task_proposal = PythonOperator(
        task_id='generate_promotion_proposal',
        python_callable=generate_promotion_proposal,
        provide_context=True,
    )

    # Task chain
    wait_l3 >> task_load >> task_backtest >> task_compare >> task_evaluate >> task_proposal
```

### 1.3 Database Migrations

```
database/migrations/
├── 034_promotion_proposals.sql
├── 035_approval_audit_log.sql
├── 036_model_registry_enhanced.sql
└── 037_experiment_contracts.sql
```

#### 1.3.1 `database/migrations/034_promotion_proposals.sql`

```sql
-- Promotion proposals from L4 (primer voto)
CREATE TABLE IF NOT EXISTS promotion_proposals (
    id SERIAL PRIMARY KEY,
    proposal_id VARCHAR(255) UNIQUE NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    recommendation VARCHAR(20) NOT NULL CHECK (recommendation IN ('PROMOTE', 'REJECT', 'REVIEW')),
    confidence DECIMAL(5,4),
    reason TEXT,
    metrics JSONB NOT NULL,
    vs_baseline JSONB,
    criteria_results JSONB NOT NULL,
    lineage JSONB NOT NULL,
    status VARCHAR(30) DEFAULT 'PENDING_APPROVAL' CHECK (
        status IN ('PENDING_APPROVAL', 'APPROVED', 'REJECTED', 'EXPIRED')
    ),
    reviewer VARCHAR(255),
    reviewer_notes TEXT,
    reviewed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days')
);

CREATE INDEX idx_promotion_proposals_status ON promotion_proposals (status);
CREATE INDEX idx_promotion_proposals_experiment ON promotion_proposals (experiment_name);
CREATE INDEX idx_promotion_proposals_created ON promotion_proposals (created_at DESC);
```

#### 1.3.2 `database/migrations/035_approval_audit_log.sql`

```sql
-- Audit log for all approvals/rejections (segundo voto)
CREATE TABLE IF NOT EXISTS approval_audit_log (
    id SERIAL PRIMARY KEY,
    action VARCHAR(50) NOT NULL CHECK (
        action IN ('APPROVE', 'REJECT', 'REQUEST_MORE_TESTS', 'EXPIRE')
    ),
    model_id VARCHAR(255) NOT NULL,
    proposal_id VARCHAR(255) REFERENCES promotion_proposals(proposal_id),
    reviewer VARCHAR(255) NOT NULL,
    reviewer_email VARCHAR(255),
    notes TEXT,
    previous_production_model VARCHAR(255),
    client_ip VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_audit_log_model ON approval_audit_log (model_id);
CREATE INDEX idx_audit_log_created ON approval_audit_log (created_at DESC);
CREATE INDEX idx_audit_log_reviewer ON approval_audit_log (reviewer);
```

#### 1.3.3 `database/migrations/036_model_registry_enhanced.sql`

```sql
-- Enhanced model registry with full lineage
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    model_path VARCHAR(512) NOT NULL,
    model_hash VARCHAR(64) NOT NULL,
    norm_stats_path VARCHAR(512),
    norm_stats_hash VARCHAR(64),
    config_hash VARCHAR(64),
    feature_order_hash VARCHAR(64),
    dataset_hash VARCHAR(64),
    stage VARCHAR(20) DEFAULT 'staging' CHECK (
        stage IN ('staging', 'production', 'archived', 'canary')
    ),
    is_active BOOLEAN DEFAULT FALSE,
    metrics JSONB,
    lineage JSONB,
    -- Approval tracking
    l4_proposal_id VARCHAR(255),
    l4_recommendation VARCHAR(20),
    l4_confidence DECIMAL(5,4),
    approved_by VARCHAR(255),
    approved_at TIMESTAMPTZ,
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    promoted_at TIMESTAMPTZ,
    archived_at TIMESTAMPTZ
);

CREATE INDEX idx_model_registry_stage ON model_registry (stage);
CREATE INDEX idx_model_registry_experiment ON model_registry (experiment_name);
CREATE INDEX idx_model_registry_active ON model_registry (is_active) WHERE is_active = TRUE;
```

#### 1.3.4 `database/migrations/037_experiment_contracts.sql`

```sql
-- Experiment contracts (inmutables)
CREATE TABLE IF NOT EXISTS experiment_contracts (
    id SERIAL PRIMARY KEY,
    contract_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    experiment_version VARCHAR(50) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,
    feature_order_hash VARCHAR(64) NOT NULL,
    reward_config_hash VARCHAR(64),
    frozen_config JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_experiment_contracts_name ON experiment_contracts (experiment_name);
CREATE INDEX idx_experiment_contracts_hash ON experiment_contracts (config_hash);
```

### 1.4 Dashboard API Endpoints

```
usdcop-trading-dashboard/app/api/
├── experiments/
│   ├── route.ts                    # GET /api/experiments - List all
│   ├── pending/
│   │   └── route.ts                # GET /api/experiments/pending
│   └── [id]/
│       ├── route.ts                # GET /api/experiments/[id]
│       ├── approve/
│       │   └── route.ts            # POST /api/experiments/[id]/approve
│       └── reject/
│           └── route.ts            # POST /api/experiments/[id]/reject
└── models/
    └── production/
        └── route.ts                # GET /api/models/production
```

#### 1.4.1 `app/api/experiments/route.ts`

```typescript
// GET /api/experiments - List all experiments with status
import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const status = searchParams.get('status'); // Optional filter

  try {
    let query = `
      SELECT
        pp.id,
        pp.proposal_id,
        pp.model_id,
        pp.experiment_name,
        pp.recommendation,
        pp.confidence,
        pp.reason,
        pp.metrics,
        pp.vs_baseline,
        pp.criteria_results,
        pp.status,
        pp.reviewer,
        pp.reviewed_at,
        pp.created_at
      FROM promotion_proposals pp
    `;

    const params: string[] = [];
    if (status) {
      query += ' WHERE pp.status = $1';
      params.push(status);
    }

    query += ' ORDER BY pp.created_at DESC LIMIT 50';

    const result = await pool.query(query, params);

    return NextResponse.json({
      experiments: result.rows,
      total: result.rowCount,
    });
  } catch (error) {
    console.error('Error fetching experiments:', error);
    return NextResponse.json(
      { error: 'Failed to fetch experiments' },
      { status: 500 }
    );
  }
}
```

#### 1.4.2 `app/api/experiments/[id]/approve/route.ts`

```typescript
// POST /api/experiments/[id]/approve - Approve experiment (SEGUNDO VOTO)
import { NextRequest, NextResponse } from 'next/server';
import { Pool } from 'pg';
import { getServerSession } from 'next-auth';

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

interface ApproveRequest {
  notes?: string;
  promote_to_production: boolean;
}

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  const session = await getServerSession();
  if (!session?.user?.email) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }

  const proposalId = params.id;
  const body: ApproveRequest = await request.json();

  const client = await pool.connect();

  try {
    await client.query('BEGIN');

    // 1. Get proposal
    const proposalResult = await client.query(
      'SELECT * FROM promotion_proposals WHERE proposal_id = $1',
      [proposalId]
    );

    if (proposalResult.rows.length === 0) {
      throw new Error('Proposal not found');
    }

    const proposal = proposalResult.rows[0];

    if (proposal.status !== 'PENDING_APPROVAL') {
      throw new Error(`Cannot approve: status is ${proposal.status}`);
    }

    // 2. Get current production model (to archive)
    const currentProdResult = await client.query(
      "SELECT model_id FROM model_registry WHERE stage = 'production' AND is_active = TRUE"
    );
    const previousProdModelId = currentProdResult.rows[0]?.model_id;

    // 3. Archive previous production model
    if (previousProdModelId) {
      await client.query(`
        UPDATE model_registry
        SET stage = 'archived', is_active = FALSE, archived_at = NOW()
        WHERE model_id = $1
      `, [previousProdModelId]);
    }

    // 4. Update proposal status
    await client.query(`
      UPDATE promotion_proposals
      SET status = 'APPROVED',
          reviewer = $1,
          reviewer_notes = $2,
          reviewed_at = NOW()
      WHERE proposal_id = $3
    `, [session.user.email, body.notes, proposalId]);

    // 5. Update model registry to production
    await client.query(`
      UPDATE model_registry
      SET stage = 'production',
          is_active = TRUE,
          approved_by = $1,
          approved_at = NOW(),
          promoted_at = NOW()
      WHERE model_id = $2
    `, [session.user.email, proposal.model_id]);

    // 6. Insert audit log
    await client.query(`
      INSERT INTO approval_audit_log
      (action, model_id, proposal_id, reviewer, reviewer_email, notes, previous_production_model)
      VALUES ('APPROVE', $1, $2, $3, $4, $5, $6)
    `, [
      proposal.model_id,
      proposalId,
      session.user.name || session.user.email,
      session.user.email,
      body.notes,
      previousProdModelId,
    ]);

    await client.query('COMMIT');

    return NextResponse.json({
      success: true,
      model_id: proposal.model_id,
      new_stage: 'production',
      previous_model_archived: previousProdModelId,
      approved_by: session.user.email,
    });

  } catch (error) {
    await client.query('ROLLBACK');
    console.error('Error approving experiment:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Approval failed' },
      { status: 500 }
    );
  } finally {
    client.release();
  }
}
```

### 1.5 Dashboard UI Components

```
usdcop-trading-dashboard/components/experiments/
├── PendingApprovalsBadge.tsx       # Badge con contador
├── PendingApprovalsList.tsx        # Lista de pendientes
├── ExperimentReviewCard.tsx        # Card con métricas
├── ExperimentDetailPage.tsx        # Página de detalle
├── LineageViewer.tsx               # Visualización de lineage
├── CriteriaResultsTable.tsx        # Tabla de criterios
└── ApprovalActions.tsx             # Botones Approve/Reject
```

---

## 2. ARCHIVOS A MODIFICAR (MODIFY)

### 2.1 DAG Registry

**File: `airflow/dags/contracts/dag_registry.py`**

```python
# AGREGAR:
RL_L4_BACKTEST_PROMOTION = "rl_l4_01_backtest_promotion"

# DEPRECAR (mantener para backward compatibility):
RL_L4_EXPERIMENT_RUNNER = "rl_l4_01_experiment_runner"  # DEPRECATED
```

### 2.2 L1 Feature Refresh - Leer Contrato de Producción

**File: `airflow/dags/l1_feature_refresh.py`**

**Cambios principales:**
1. Importar `ProductionContract`
2. Validar feature_order_hash contra contrato de producción
3. Validar norm_stats_hash
4. Asegurar que tabla histórica existe

```python
# AGREGAR imports:
from src.contracts.production_contract import ProductionContract

# AGREGAR función de validación:
def validate_production_contract(**context) -> Dict[str, Any]:
    """
    Validar que L1 usa el mismo feature_order y norm_stats
    que el modelo en producción.

    NUEVO: Lee el contrato del modelo aprobado (ambos votos).
    """
    conn = get_db_connection()

    # Cargar contrato de producción
    prod_contract = ProductionContract.from_db(conn)

    if prod_contract is None:
        logging.warning("No production contract found - using defaults")
        return {'status': 'no_contract', 'using_defaults': True}

    # Validar feature_order_hash
    from src.core.contracts.feature_contract import FEATURE_ORDER_HASH

    if not prod_contract.validate_feature_order(FEATURE_ORDER_HASH):
        raise ValueError(
            f"FEATURE_ORDER_HASH mismatch: "
            f"L1={FEATURE_ORDER_HASH}, "
            f"Production={prod_contract.feature_order_hash}"
        )

    # Validar norm_stats_hash (si disponible)
    if prod_contract.norm_stats_hash:
        current_hash = compute_norm_stats_hash(prod_contract.norm_stats_path)
        if current_hash != prod_contract.norm_stats_hash:
            logging.warning(
                f"norm_stats.json modified since model training: "
                f"expected={prod_contract.norm_stats_hash}, "
                f"current={current_hash}"
            )

    logging.info(f"Production contract validated: {prod_contract.model_id}")
    logging.info(f"  feature_order_hash: {prod_contract.feature_order_hash}")
    logging.info(f"  norm_stats_hash: {prod_contract.norm_stats_hash}")

    context['ti'].xcom_push(key='production_contract', value=prod_contract.to_dict())

    conn.close()
    return {'status': 'validated', 'model_id': prod_contract.model_id}


# MODIFICAR DAG para agregar task de validación:
with dag:
    # ... existing tasks ...

    # NUEVO: Validar contrato de producción
    task_validate_contract = PythonOperator(
        task_id='validate_production_contract',
        python_callable=validate_production_contract,
        provide_context=True,
    )

    # MODIFICAR chain:
    task_check >> task_validate_contract >> task_wait_ohlcv >> task_compute >> ...
```

### 2.3 L5 Multi-Model Inference - Usar Contrato de Producción

**File: `airflow/dags/l5_multi_model_inference.py`**

**Cambios principales:**
1. Cargar modelo desde `ProductionContract` en vez de config.models
2. Validar hashes antes de inference
3. Log lineage completo

```python
# AGREGAR imports:
from src.contracts.production_contract import ProductionContract

# MODIFICAR ModelRegistry.get_enabled_models():
def get_production_model(self) -> Optional[ModelConfig]:
    """
    Get production model from ProductionContract.

    NUEVO: Lee el contrato del modelo aprobado (ambos votos).
    """
    conn = get_db_connection()

    prod_contract = ProductionContract.from_db(conn)
    conn.close()

    if prod_contract is None:
        logging.warning("No production contract - falling back to config.models")
        return None

    return ModelConfig(
        model_id=prod_contract.model_id,
        model_name=prod_contract.experiment_name,
        model_type="PPO",
        model_path=prod_contract.model_path,
        version="production",
        enabled=True,
        is_production=True,
        priority=1,
        feature_order_hash=prod_contract.feature_order_hash,
        norm_stats_hash=prod_contract.norm_stats_hash,
    )


# MODIFICAR load_models task:
def load_models(**ctx) -> Dict[str, Any]:
    """Task 2: Load production model from contract."""
    logging.info("Loading production model from contract...")
    create_output_tables()

    # NUEVO: Primero intentar cargar desde ProductionContract
    prod_model = model_registry.get_production_model()

    if prod_model:
        logging.info(f"Using production model from contract: {prod_model.model_id}")
        model = model_registry.load_model(prod_model)
        if model:
            loaded_models = [{
                "model_id": prod_model.model_id,
                "model_name": prod_model.model_name,
                "model_type": prod_model.model_type,
                "is_production": True,
                "feature_order_hash": prod_model.feature_order_hash,
                "norm_stats_hash": prod_model.norm_stats_hash,
            }]
            ctx["ti"].xcom_push(key="loaded_models", value=loaded_models)
            ctx["ti"].xcom_push(key="using_contract", value=True)
            return {"loaded_count": 1, "from_contract": True}

    # Fallback to config.models
    logging.info("Falling back to config.models")
    model_configs = model_registry.get_enabled_models()
    # ... rest of existing code ...
```

### 2.4 L2 Dataset Builder - Generar ExperimentContract

**File: `airflow/dags/l2_dataset_builder.py`**

**Cambios:**
1. Crear ExperimentContract desde YAML
2. Guardar contrato en DB
3. Incluir config_hash en L2Output

```python
# AGREGAR import:
from src.contracts.experiment_contract import ExperimentContract

# MODIFICAR build_dataset task:
def build_dataset(**context):
    # ... existing code ...

    # NUEVO: Crear ExperimentContract desde YAML
    yaml_path = Path(f"config/experiments/{experiment_name}.yaml")
    if yaml_path.exists():
        exp_contract = ExperimentContract.from_yaml(yaml_path)

        # Guardar en DB
        conn = get_db_connection()
        exp_contract.save_to_db(conn)
        conn.close()

        logging.info(f"Created ExperimentContract: {exp_contract.contract_id}")
        logging.info(f"  config_hash: {exp_contract.config_hash}")

        # Incluir en L2Output
        context['ti'].xcom_push(key='experiment_contract', value=exp_contract.to_dict())

    # ... rest of existing code ...
```

### 2.5 L3 Model Training - Propagar Hashes

**File: `airflow/dags/l3_model_training.py`**

**Cambios:**
1. Recibir ExperimentContract de L2
2. Incluir todos los hashes en L3Output
3. Registrar modelo en model_registry con lineage

```python
# MODIFICAR save_model_artifacts task:
def save_model_artifacts(**context):
    # ... existing code ...

    # NUEVO: Obtener experiment contract de L2
    exp_contract = context['ti'].xcom_pull(
        task_ids='build_dataset',
        key='experiment_contract',
        dag_id=RL_L2_DATASET_BUILD,
    )

    # Registrar en model_registry con lineage completo
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO model_registry (
            model_id, experiment_name, model_path, model_hash,
            config_hash, feature_order_hash, norm_stats_hash, dataset_hash,
            norm_stats_path, lineage, stage
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'staging')
        ON CONFLICT (model_id) DO UPDATE SET
            model_path = EXCLUDED.model_path,
            model_hash = EXCLUDED.model_hash,
            stage = 'staging'
    """, (
        model_id,
        experiment_name,
        model_path,
        model_hash,
        exp_contract.get('config_hash') if exp_contract else None,
        exp_contract.get('feature_order_hash') if exp_contract else FEATURE_ORDER_HASH,
        norm_stats_hash,
        dataset_hash,
        norm_stats_path,
        json.dumps({
            'l2_dataset_hash': dataset_hash,
            'l2_config_hash': exp_contract.get('config_hash') if exp_contract else None,
            'training_run_id': mlflow_run_id,
        }),
    ))

    conn.commit()
    cur.close()
    conn.close()

    # ... rest of existing code ...
```

---

## 3. ARCHIVOS A ELIMINAR (DELETE)

### 3.1 DAGs Deprecados

```
airflow/dags/
├── l4_experiment_runner.py         # DELETE: Reemplazado por l4_backtest_promotion.py
└── l4_backtest_validation.py       # DELETE: Fusionado en l4_backtest_promotion.py
```

**Nota:** No eliminar físicamente hasta que el nuevo L4 esté en producción. Marcar como deprecated primero.

### 3.2 Scripts Obsoletos

```
scripts/
├── backfill_daily_ohlcv_investing.py    # Ya marcado para DELETE
├── backfill_forecasting_dataset.py      # Ya marcado para DELETE
├── backfill_fred_cpilfesl.py            # Ya marcado para DELETE
├── backfill_investing_complete.py       # Ya marcado para DELETE
├── backfill_investing_migration.py      # Ya marcado para DELETE
├── fix_wti_data.py                      # Ya marcado para DELETE
├── scraper_banrep_selenium.py           # Ya marcado para DELETE
└── scraper_ibr_banrep.py                # Ya marcado para DELETE
```

---

## 4. ORDEN DE IMPLEMENTACIÓN

### Fase 1: Contracts Core (Día 1-2)

1. Crear `src/contracts/experiment_contract.py`
2. Crear `src/contracts/promotion_contract.py`
3. Crear `src/contracts/production_contract.py`
4. Ejecutar migraciones de BD

### Fase 2: L4 Backtest + Promotion (Día 3-4)

1. Crear `airflow/dags/l4_backtest_promotion.py`
2. Agregar a dag_registry.py
3. Probar con experimento existente

### Fase 3: L1/L5 Integration (Día 5-6)

1. Modificar L1 para validar ProductionContract
2. Modificar L5 para cargar modelo desde ProductionContract
3. Probar flujo completo L2→L3→L4→Dashboard

### Fase 4: Dashboard (Día 7-9)

1. Crear API endpoints
2. Crear componentes UI
3. Integrar autenticación
4. Probar approve/reject flow

### Fase 5: Testing E2E (Día 10)

1. Test completo: YAML → L2 → L3 → L4 → Dashboard → L1/L5
2. Verificar lineage chain completo
3. Validar hash matching en cada etapa

---

## 5. CHECKLIST DE VERIFICACIÓN

### Contracts
- [ ] ExperimentContract genera config_hash único por YAML
- [ ] PromotionProposal incluye todos los hashes de lineage
- [ ] ProductionContract se carga correctamente en L1/L5

### L4 Backtest
- [ ] Backtest usa test.parquet (nunca visto en training)
- [ ] Criterios se evalúan desde experiment.yaml
- [ ] Comparación vs baseline funciona
- [ ] Proposal se guarda en DB con status PENDING_APPROVAL

### Dashboard
- [ ] Lista de pendientes muestra badge correcto
- [ ] Detalle muestra métricas + lineage
- [ ] Approve actualiza model_registry.stage='production'
- [ ] Audit log registra todas las acciones

### L1/L5 Production
- [ ] L1 valida feature_order_hash contra ProductionContract
- [ ] L5 carga modelo desde ProductionContract
- [ ] Hot reload funciona cuando hay nuevo modelo

---

**Próximo paso:** ¿Empezamos con Fase 1 (Contracts Core)?
