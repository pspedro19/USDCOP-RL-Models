"""
Unified Lineage Service
=======================

Provides complete lineage tracking across the ML pipeline.
Unifies lineage information from MLflow, DVC, and PostgreSQL.

Principle: Single query should show full lineage chain:
    Dataset (DVC) → Training (MLflow) → Model (Registry) → Inference (DB)

@version 1.0.0
@principle MLflow-First + DVC-Tracked
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class LineageNodeType(str, Enum):
    """Types of nodes in the lineage graph."""
    DATASET = "dataset"
    FEATURE_SET = "feature_set"
    TRAINING_RUN = "training_run"
    MODEL = "model"
    MODEL_VERSION = "model_version"
    BACKTEST = "backtest"
    INFERENCE = "inference"
    FORECAST = "forecast"


class LineageRelation(str, Enum):
    """Types of relationships between lineage nodes."""
    DERIVED_FROM = "derived_from"       # Dataset → Features
    TRAINED_ON = "trained_on"           # Model → Dataset
    PRODUCED_BY = "produced_by"         # Model → Training Run
    VALIDATED_BY = "validated_by"       # Model → Backtest
    USED_BY = "used_by"                 # Model → Inference
    PROMOTED_FROM = "promoted_from"     # Model Version → Previous Version


# =============================================================================
# LINEAGE DATA CLASSES
# =============================================================================

@dataclass
class LineageNode:
    """A node in the lineage graph."""
    id: str
    node_type: LineageNodeType
    name: str
    version: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source references
    mlflow_run_id: Optional[str] = None
    dvc_tag: Optional[str] = None
    db_record_id: Optional[int] = None

    # Hashes for reproducibility
    content_hash: Optional[str] = None
    config_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "mlflow_run_id": self.mlflow_run_id,
            "dvc_tag": self.dvc_tag,
            "db_record_id": self.db_record_id,
            "content_hash": self.content_hash,
            "config_hash": self.config_hash,
        }


@dataclass
class LineageEdge:
    """An edge connecting two lineage nodes."""
    source_id: str
    target_id: str
    relation: LineageRelation
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class LineageRecord:
    """
    Complete lineage record for a model or inference.

    This is the unified view combining:
    - DVC dataset version
    - MLflow training run
    - Model registry version
    - Backtest validation
    - Inference records
    """
    # Identity
    record_id: str
    pipeline: str  # "rl" or "forecasting"
    created_at: datetime = field(default_factory=datetime.now)

    # Dataset lineage
    dataset_path: Optional[str] = None
    dataset_hash: Optional[str] = None
    dataset_dvc_tag: Optional[str] = None
    dataset_version: Optional[str] = None

    # Feature lineage
    feature_config_hash: Optional[str] = None
    feature_order_hash: Optional[str] = None
    num_features: Optional[int] = None

    # Training lineage
    mlflow_experiment_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    training_config_hash: Optional[str] = None
    training_duration_seconds: Optional[float] = None

    # Model lineage
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    model_stage: Optional[str] = None
    model_uri: Optional[str] = None
    model_hash: Optional[str] = None

    # Validation lineage
    backtest_id: Optional[str] = None
    validation_metrics: Dict[str, float] = field(default_factory=dict)

    # Inference lineage (for deployed models)
    inference_count: int = 0
    last_inference_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "pipeline": self.pipeline,
            "created_at": self.created_at.isoformat(),
            "dataset": {
                "path": self.dataset_path,
                "hash": self.dataset_hash,
                "dvc_tag": self.dataset_dvc_tag,
                "version": self.dataset_version,
            },
            "features": {
                "config_hash": self.feature_config_hash,
                "order_hash": self.feature_order_hash,
                "num_features": self.num_features,
            },
            "training": {
                "mlflow_experiment_id": self.mlflow_experiment_id,
                "mlflow_run_id": self.mlflow_run_id,
                "config_hash": self.training_config_hash,
                "duration_seconds": self.training_duration_seconds,
            },
            "model": {
                "name": self.model_name,
                "version": self.model_version,
                "stage": self.model_stage,
                "uri": self.model_uri,
                "hash": self.model_hash,
            },
            "validation": {
                "backtest_id": self.backtest_id,
                "metrics": self.validation_metrics,
            },
            "inference": {
                "count": self.inference_count,
                "last_at": self.last_inference_at.isoformat() if self.last_inference_at else None,
            },
        }

    def compute_lineage_hash(self) -> str:
        """Compute hash of the entire lineage for comparison."""
        content = json.dumps({
            "dataset_hash": self.dataset_hash,
            "feature_config_hash": self.feature_config_hash,
            "training_config_hash": self.training_config_hash,
            "model_hash": self.model_hash,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


# =============================================================================
# LINEAGE SERVICE
# =============================================================================

class LineageService:
    """
    Unified lineage tracking service.

    Provides a single interface to:
    - Record lineage events
    - Query lineage history
    - Validate lineage completeness
    - Export lineage for audit

    Usage:
        service = LineageService(db_connection_string="...")

        # Record dataset lineage
        service.record_dataset(
            name="train_v1",
            path="data/processed/train.parquet",
            dvc_tag="dataset-v1.0.0"
        )

        # Record training lineage
        service.record_training(
            dataset_id="...",
            mlflow_run_id="...",
            model_name="ppo-usdcop"
        )

        # Query full lineage
        lineage = service.get_model_lineage("ppo-usdcop", "v1")
    """

    def __init__(
        self,
        db_connection_string: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
    ):
        self.db_connection_string = db_connection_string or os.environ.get("DATABASE_URL")
        self.mlflow_tracking_uri = mlflow_tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )

        # In-memory cache (for testing/quick lookups)
        self._nodes: Dict[str, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        self._records: Dict[str, LineageRecord] = {}

    # =========================================================================
    # RECORD METHODS
    # =========================================================================

    def record_dataset(
        self,
        name: str,
        path: str,
        dvc_tag: Optional[str] = None,
        content_hash: Optional[str] = None,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """Record a dataset in the lineage graph."""
        node_id = f"dataset:{name}:{version or 'latest'}"

        node = LineageNode(
            id=node_id,
            node_type=LineageNodeType.DATASET,
            name=name,
            version=version,
            dvc_tag=dvc_tag,
            content_hash=content_hash,
            metadata=metadata or {"path": path},
        )

        self._nodes[node_id] = node
        self._persist_node(node)

        logger.info(f"Recorded dataset lineage: {node_id}")
        return node

    def record_training_run(
        self,
        name: str,
        mlflow_run_id: str,
        dataset_node_id: str,
        config_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """Record a training run in the lineage graph."""
        node_id = f"training:{name}:{mlflow_run_id[:8]}"

        node = LineageNode(
            id=node_id,
            node_type=LineageNodeType.TRAINING_RUN,
            name=name,
            mlflow_run_id=mlflow_run_id,
            config_hash=config_hash,
            metadata=metadata or {},
        )

        self._nodes[node_id] = node

        # Create edge: training TRAINED_ON dataset
        edge = LineageEdge(
            source_id=node_id,
            target_id=dataset_node_id,
            relation=LineageRelation.TRAINED_ON,
        )
        self._edges.append(edge)

        self._persist_node(node)
        self._persist_edge(edge)

        logger.info(f"Recorded training lineage: {node_id} -> {dataset_node_id}")
        return node

    def record_model(
        self,
        name: str,
        version: str,
        training_node_id: str,
        mlflow_run_id: Optional[str] = None,
        model_uri: Optional[str] = None,
        model_hash: Optional[str] = None,
        stage: str = "Staging",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LineageNode:
        """Record a model version in the lineage graph."""
        node_id = f"model:{name}:{version}"

        node = LineageNode(
            id=node_id,
            node_type=LineageNodeType.MODEL_VERSION,
            name=name,
            version=version,
            mlflow_run_id=mlflow_run_id,
            content_hash=model_hash,
            metadata={
                **(metadata or {}),
                "uri": model_uri,
                "stage": stage,
            },
        )

        self._nodes[node_id] = node

        # Create edge: model PRODUCED_BY training
        edge = LineageEdge(
            source_id=node_id,
            target_id=training_node_id,
            relation=LineageRelation.PRODUCED_BY,
        )
        self._edges.append(edge)

        self._persist_node(node)
        self._persist_edge(edge)

        logger.info(f"Recorded model lineage: {node_id} -> {training_node_id}")
        return node

    def record_model_promotion(
        self,
        model_name: str,
        from_version: str,
        to_version: str,
        from_stage: str,
        to_stage: str,
        reason: str,
    ) -> LineageEdge:
        """Record a model promotion event."""
        from_node_id = f"model:{model_name}:{from_version}"
        to_node_id = f"model:{model_name}:{to_version}"

        edge = LineageEdge(
            source_id=to_node_id,
            target_id=from_node_id,
            relation=LineageRelation.PROMOTED_FROM,
            metadata={
                "from_stage": from_stage,
                "to_stage": to_stage,
                "reason": reason,
            },
        )

        self._edges.append(edge)
        self._persist_edge(edge)

        logger.info(f"Recorded promotion: {model_name} {from_stage} -> {to_stage}")
        return edge

    def record_complete_lineage(
        self,
        pipeline: str,
        dataset_path: str,
        dataset_hash: str,
        dataset_dvc_tag: Optional[str],
        mlflow_run_id: str,
        model_name: str,
        model_version: str,
        model_uri: str,
        feature_config_hash: Optional[str] = None,
        training_config_hash: Optional[str] = None,
        validation_metrics: Optional[Dict[str, float]] = None,
    ) -> LineageRecord:
        """
        Record complete lineage in a single call.

        This is the preferred method for recording lineage at the end of training.
        """
        record_id = f"{pipeline}:{model_name}:{model_version}:{datetime.now().strftime('%Y%m%d%H%M%S')}"

        record = LineageRecord(
            record_id=record_id,
            pipeline=pipeline,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            dataset_dvc_tag=dataset_dvc_tag,
            mlflow_run_id=mlflow_run_id,
            feature_config_hash=feature_config_hash,
            training_config_hash=training_config_hash,
            model_name=model_name,
            model_version=model_version,
            model_uri=model_uri,
            validation_metrics=validation_metrics or {},
        )

        self._records[record_id] = record
        self._persist_record(record)

        logger.info(f"Recorded complete lineage: {record_id}")
        return record

    # =========================================================================
    # QUERY METHODS
    # =========================================================================

    def get_model_lineage(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Optional[LineageRecord]:
        """
        Get complete lineage for a model.

        If version is None, returns lineage for the latest version.
        """
        # First check cache
        for record_id, record in self._records.items():
            if record.model_name == model_name:
                if version is None or record.model_version == version:
                    return record

        # Query from database
        return self._query_lineage_from_db(model_name, version)

    def get_dataset_descendants(
        self,
        dataset_id: str,
    ) -> List[LineageNode]:
        """Get all models trained on a specific dataset."""
        descendants = []

        for edge in self._edges:
            if edge.target_id == dataset_id and edge.relation == LineageRelation.TRAINED_ON:
                if edge.source_id in self._nodes:
                    descendants.append(self._nodes[edge.source_id])

        return descendants

    def get_lineage_chain(
        self,
        node_id: str,
        direction: str = "upstream",  # "upstream" or "downstream"
    ) -> List[LineageNode]:
        """Get the full lineage chain for a node."""
        chain = []
        visited = set()

        def traverse(current_id: str):
            if current_id in visited:
                return
            visited.add(current_id)

            if current_id in self._nodes:
                chain.append(self._nodes[current_id])

            for edge in self._edges:
                if direction == "upstream":
                    if edge.source_id == current_id:
                        traverse(edge.target_id)
                else:
                    if edge.target_id == current_id:
                        traverse(edge.source_id)

        traverse(node_id)
        return chain

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    def validate_lineage_completeness(
        self,
        model_name: str,
        version: str,
    ) -> Tuple[bool, List[str]]:
        """
        Validate that a model has complete lineage.

        Checks:
        - Dataset reference exists
        - DVC tag exists
        - MLflow run ID exists
        - Model is registered in MLflow
        - All hashes are present
        """
        issues = []
        record = self.get_model_lineage(model_name, version)

        if not record:
            return False, [f"No lineage record found for {model_name}:{version}"]

        # Check dataset lineage
        if not record.dataset_path:
            issues.append("Missing dataset_path")
        if not record.dataset_hash:
            issues.append("Missing dataset_hash")
        if not record.dataset_dvc_tag:
            issues.append("Missing dataset_dvc_tag (DVC tracking required)")

        # Check training lineage
        if not record.mlflow_run_id:
            issues.append("Missing mlflow_run_id (MLflow tracking required)")

        # Check model lineage
        if not record.model_uri:
            issues.append("Missing model_uri (MLflow Model Registry required)")

        # Check feature lineage
        if not record.feature_config_hash:
            issues.append("Missing feature_config_hash")

        is_complete = len(issues) == 0
        return is_complete, issues

    # =========================================================================
    # PERSISTENCE METHODS
    # =========================================================================

    def _persist_node(self, node: LineageNode):
        """Persist a lineage node to the database."""
        if not self.db_connection_string:
            return

        try:
            import psycopg2
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ml.lineage_nodes
                (id, node_type, name, version, mlflow_run_id, dvc_tag,
                 content_hash, config_hash, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    metadata = EXCLUDED.metadata,
                    content_hash = EXCLUDED.content_hash
            """, (
                node.id,
                node.node_type.value,
                node.name,
                node.version,
                node.mlflow_run_id,
                node.dvc_tag,
                node.content_hash,
                node.config_hash,
                json.dumps(node.metadata),
                node.created_at,
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to persist lineage node: {e}")

    def _persist_edge(self, edge: LineageEdge):
        """Persist a lineage edge to the database."""
        if not self.db_connection_string:
            return

        try:
            import psycopg2
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ml.lineage_edges
                (source_id, target_id, relation, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_id, target_id, relation) DO NOTHING
            """, (
                edge.source_id,
                edge.target_id,
                edge.relation.value,
                json.dumps(edge.metadata),
                edge.created_at,
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to persist lineage edge: {e}")

    def _persist_record(self, record: LineageRecord):
        """Persist a complete lineage record to the database."""
        if not self.db_connection_string:
            return

        try:
            import psycopg2
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            cur.execute("""
                INSERT INTO ml.lineage_records
                (record_id, pipeline, dataset_path, dataset_hash, dataset_dvc_tag,
                 mlflow_run_id, feature_config_hash, training_config_hash,
                 model_name, model_version, model_uri, validation_metrics, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (record_id) DO UPDATE SET
                    validation_metrics = EXCLUDED.validation_metrics
            """, (
                record.record_id,
                record.pipeline,
                record.dataset_path,
                record.dataset_hash,
                record.dataset_dvc_tag,
                record.mlflow_run_id,
                record.feature_config_hash,
                record.training_config_hash,
                record.model_name,
                record.model_version,
                record.model_uri,
                json.dumps(record.validation_metrics),
                record.created_at,
            ))

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            logger.warning(f"Failed to persist lineage record: {e}")

    def _query_lineage_from_db(
        self,
        model_name: str,
        version: Optional[str],
    ) -> Optional[LineageRecord]:
        """Query lineage from database."""
        if not self.db_connection_string:
            return None

        try:
            import psycopg2
            conn = psycopg2.connect(self.db_connection_string)
            cur = conn.cursor()

            if version:
                cur.execute("""
                    SELECT * FROM ml.lineage_records
                    WHERE model_name = %s AND model_version = %s
                    ORDER BY created_at DESC LIMIT 1
                """, (model_name, version))
            else:
                cur.execute("""
                    SELECT * FROM ml.lineage_records
                    WHERE model_name = %s
                    ORDER BY created_at DESC LIMIT 1
                """, (model_name,))

            row = cur.fetchone()
            cur.close()
            conn.close()

            if row:
                # Convert row to LineageRecord
                return LineageRecord(
                    record_id=row[0],
                    pipeline=row[1],
                    dataset_path=row[2],
                    dataset_hash=row[3],
                    dataset_dvc_tag=row[4],
                    mlflow_run_id=row[5],
                    feature_config_hash=row[6],
                    training_config_hash=row[7],
                    model_name=row[8],
                    model_version=row[9],
                    model_uri=row[10],
                    validation_metrics=json.loads(row[11]) if row[11] else {},
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to query lineage: {e}")
            return None


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_lineage_service: Optional[LineageService] = None


def get_lineage_service() -> LineageService:
    """Get the lineage service singleton."""
    global _lineage_service
    if _lineage_service is None:
        _lineage_service = LineageService()
    return _lineage_service


# =============================================================================
# DATABASE SCHEMA
# =============================================================================

CREATE_LINEAGE_TABLES_SQL = """
-- Lineage schema
CREATE SCHEMA IF NOT EXISTS ml;

-- Lineage nodes table
CREATE TABLE IF NOT EXISTS ml.lineage_nodes (
    id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50),
    mlflow_run_id VARCHAR(50),
    dvc_tag VARCHAR(100),
    db_record_id INTEGER,
    content_hash VARCHAR(64),
    config_hash VARCHAR(64),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Lineage edges table
CREATE TABLE IF NOT EXISTS ml.lineage_edges (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL REFERENCES ml.lineage_nodes(id),
    target_id VARCHAR(255) NOT NULL REFERENCES ml.lineage_nodes(id),
    relation VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(source_id, target_id, relation)
);

-- Complete lineage records table
CREATE TABLE IF NOT EXISTS ml.lineage_records (
    record_id VARCHAR(255) PRIMARY KEY,
    pipeline VARCHAR(50) NOT NULL,
    dataset_path TEXT,
    dataset_hash VARCHAR(64),
    dataset_dvc_tag VARCHAR(100),
    dataset_version VARCHAR(50),
    mlflow_experiment_id VARCHAR(100),
    mlflow_run_id VARCHAR(50),
    feature_config_hash VARCHAR(64),
    feature_order_hash VARCHAR(64),
    num_features INTEGER,
    training_config_hash VARCHAR(64),
    training_duration_seconds FLOAT,
    model_name VARCHAR(255),
    model_version VARCHAR(50),
    model_stage VARCHAR(50),
    model_uri TEXT,
    model_hash VARCHAR(64),
    backtest_id VARCHAR(100),
    validation_metrics JSONB DEFAULT '{}',
    inference_count INTEGER DEFAULT 0,
    last_inference_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_type ON ml.lineage_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_name ON ml.lineage_nodes(name);
CREATE INDEX IF NOT EXISTS idx_lineage_records_model ON ml.lineage_records(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_lineage_records_pipeline ON ml.lineage_records(pipeline);
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "LineageNodeType",
    "LineageRelation",
    # Classes
    "LineageNode",
    "LineageEdge",
    "LineageRecord",
    "LineageService",
    # Functions
    "get_lineage_service",
    # SQL
    "CREATE_LINEAGE_TABLES_SQL",
]
