"""
Model Registry - Trazabilidad de modelos en BD.
CLAUDE-T15 | Plan Item: P1-11
Contrato: CTR-010

Proporciona:
- Registro de modelos con hashes de integridad
- Verificacion de integridad antes de inference
- Tracking de modelos desplegados
"""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Metadata de un modelo registrado."""
    model_id: str
    model_version: str
    model_path: str
    model_hash: str
    norm_stats_hash: str
    config_hash: Optional[str]
    observation_dim: int
    action_space: int
    feature_order: List[str]
    status: str
    created_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None


class ModelIntegrityError(Exception):
    """Error de integridad del modelo."""
    pass


class ModelNotFoundError(Exception):
    """Modelo no encontrado en el registro."""
    pass


class ModelRegistry:
    """
    Registro de modelos en BD para trazabilidad.
    Contrato: CTR-010

    Funcionalidades:
    - Registrar modelos con hashes de integridad
    - Verificar integridad antes de usar un modelo
    - Marcar modelos como deployed/retired
    - Listar modelos activos

    Ejemplo:
        registry = ModelRegistry(db_connection)
        model_id = registry.register_model(
            model_path=Path("models/ppo_model.onnx"),
            version="v1",
            training_info={"dataset_id": 123}
        )
        registry.verify_model_integrity(model_id)
    """

    def __init__(self, conn=None):
        """
        Inicializa el registro.

        Args:
            conn: Conexion a la base de datos (opcional para modo standalone)
        """
        self.conn = conn

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """
        Computa SHA256 hash de un archivo.

        Args:
            file_path: Path al archivo

        Returns:
            Hash SHA256 como string hexadecimal (64 chars)

        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")

        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)

        return sha256.hexdigest()

    def register_model(
        self,
        model_path: Path,
        version: str,
        training_info: Optional[Dict[str, Any]] = None,
        feature_order: Optional[List[str]] = None
    ) -> str:
        """
        Registra un modelo con todos sus hashes.

        Args:
            model_path: Path al archivo ONNX
            version: Version del modelo (e.g., "v1")
            training_info: Info adicional de training
            feature_order: Orden de features (usa contrato si no se provee)

        Returns:
            model_id generado

        Raises:
            FileNotFoundError: Si los archivos no existen
        """
        # Importar contrato para feature order default
        from features.contract import FEATURE_CONTRACT

        # Computar hashes
        model_hash = self.compute_file_hash(model_path)

        # Path a norm_stats
        project_root = Path(__file__).parent.parent.parent
        norm_stats_path = project_root / f"config/{version}_norm_stats.json"
        if not norm_stats_path.exists():
            norm_stats_path = project_root / "config/norm_stats.json"

        norm_stats_hash = self.compute_file_hash(norm_stats_path)

        # Config hash (opcional)
        config_path = project_root / f"config/{version}_config.yaml"
        config_hash = None
        if config_path.exists():
            config_hash = self.compute_file_hash(config_path)

        # Feature order
        if feature_order is None:
            feature_order = list(FEATURE_CONTRACT.feature_order)

        # Generar model_id
        model_id = f"ppo_{version}_{model_hash[:8]}"

        training_info = training_info or {}

        if self.conn is not None:
            self._insert_model_record(
                model_id=model_id,
                version=version,
                model_path=str(model_path),
                model_hash=model_hash,
                norm_stats_hash=norm_stats_hash,
                config_hash=config_hash,
                feature_order=feature_order,
                training_info=training_info
            )

        logger.info(f"Modelo registrado: {model_id} (hash: {model_hash[:16]}...)")
        return model_id

    def _insert_model_record(
        self,
        model_id: str,
        version: str,
        model_path: str,
        model_hash: str,
        norm_stats_hash: str,
        config_hash: Optional[str],
        feature_order: List[str],
        training_info: Dict
    ):
        """Inserta registro en BD."""
        self.conn.execute("""
            INSERT INTO model_registry
            (model_id, model_version, model_path, model_hash,
             norm_stats_hash, config_hash, observation_dim, action_space,
             feature_order, training_dataset_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id) DO UPDATE SET
                model_path = EXCLUDED.model_path,
                model_hash = EXCLUDED.model_hash
        """, (
            model_id, version, model_path, model_hash,
            norm_stats_hash, config_hash, 15, 3,
            json.dumps(feature_order),
            training_info.get('dataset_id')
        ))

    def verify_model_integrity(
        self,
        model_id: str,
        model_path: Optional[Path] = None
    ) -> bool:
        """
        Verifica integridad del modelo contra registro.

        Args:
            model_id: ID del modelo a verificar
            model_path: Path actual del modelo (opcional, usa registro si no se provee)

        Returns:
            True si la integridad es valida

        Raises:
            ModelNotFoundError: Si el modelo no esta registrado
            ModelIntegrityError: Si el hash no coincide
        """
        if self.conn is None:
            logger.warning("Sin conexion a BD - no se puede verificar integridad")
            return True

        row = self.conn.fetchone(
            "SELECT model_path, model_hash FROM model_registry WHERE model_id = %s",
            (model_id,)
        )

        if not row:
            raise ModelNotFoundError(f"Modelo {model_id} no registrado")

        if model_path is None:
            model_path = Path(row['model_path'])

        current_hash = self.compute_file_hash(model_path)

        if current_hash != row['model_hash']:
            raise ModelIntegrityError(
                f"Model integrity check FAILED!\n"
                f"Model ID: {model_id}\n"
                f"Expected hash: {row['model_hash']}\n"
                f"Actual hash:   {current_hash}\n"
                f"El modelo ha sido modificado desde su registro."
            )

        logger.info(f"Integridad verificada: {model_id}")
        return True

    def deploy_model(self, model_id: str) -> bool:
        """
        Marca un modelo como deployed.

        Args:
            model_id: ID del modelo

        Returns:
            True si se actualizo correctamente
        """
        if self.conn is None:
            logger.warning("Sin conexion a BD")
            return False

        self.conn.execute("""
            UPDATE model_registry
            SET status = 'deployed', deployed_at = CURRENT_TIMESTAMP
            WHERE model_id = %s
        """, (model_id,))

        logger.info(f"Modelo deployed: {model_id}")
        return True

    def retire_model(self, model_id: str) -> bool:
        """
        Marca un modelo como retired.

        Args:
            model_id: ID del modelo

        Returns:
            True si se actualizo correctamente
        """
        if self.conn is None:
            logger.warning("Sin conexion a BD")
            return False

        self.conn.execute("""
            UPDATE model_registry
            SET status = 'retired', retired_at = CURRENT_TIMESTAMP
            WHERE model_id = %s
        """, (model_id,))

        logger.info(f"Modelo retired: {model_id}")
        return True

    def get_active_models(self) -> List[ModelMetadata]:
        """
        Obtiene lista de modelos deployed.

        Returns:
            Lista de ModelMetadata
        """
        if self.conn is None:
            return []

        rows = self.conn.fetchall("""
            SELECT * FROM model_registry WHERE status = 'deployed'
            ORDER BY deployed_at DESC
        """)

        return [self._row_to_metadata(row) for row in rows]

    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Obtiene metadata de un modelo.

        Args:
            model_id: ID del modelo

        Returns:
            ModelMetadata o None si no existe
        """
        if self.conn is None:
            return None

        row = self.conn.fetchone(
            "SELECT * FROM model_registry WHERE model_id = %s",
            (model_id,)
        )

        if not row:
            return None

        return self._row_to_metadata(row)

    def _row_to_metadata(self, row: Dict) -> ModelMetadata:
        """Convierte row de BD a ModelMetadata."""
        return ModelMetadata(
            model_id=row['model_id'],
            model_version=row['model_version'],
            model_path=row['model_path'],
            model_hash=row['model_hash'],
            norm_stats_hash=row['norm_stats_hash'],
            config_hash=row.get('config_hash'),
            observation_dim=row['observation_dim'],
            action_space=row['action_space'],
            feature_order=json.loads(row['feature_order']) if isinstance(row['feature_order'], str) else row['feature_order'],
            status=row['status'],
            created_at=row.get('created_at'),
            deployed_at=row.get('deployed_at')
        )


def verify_model_before_inference(
    registry: ModelRegistry,
    model_id: str,
    model_path: Path
) -> bool:
    """
    Funcion de conveniencia para verificar modelo antes de inference.

    Args:
        registry: Instancia de ModelRegistry
        model_id: ID del modelo
        model_path: Path al archivo ONNX

    Returns:
        True si la verificacion pasa

    Raises:
        ModelIntegrityError: Si hay problemas de integridad
    """
    return registry.verify_model_integrity(model_id, model_path)
