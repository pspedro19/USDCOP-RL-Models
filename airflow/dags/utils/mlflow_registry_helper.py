"""
MLflow Model Registry Helper
============================
Funciones para registrar modelos RL y ML en MLflow Registry

Usage:
    from utils.mlflow_registry_helper import register_rl_model, register_ml_model

Note: MLflow is optional. If not installed, functions will log warnings but not fail.
"""

import logging

# ✅ CORRECCIÓN: Imports opcionales (mlflow no siempre está instalado)
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

logger = logging.getLogger(__name__)

def register_rl_model(run_id: str, model_name: str = "RL_PPO_USDCOP", metrics: dict = None):
    """
    Registra modelo RL en MLflow Registry y lo promociona si pasa gates

    Args:
        run_id: MLflow run ID
        model_name: Nombre en Model Registry
        metrics: Dict con métricas (sortino, max_dd, etc.)
    """

    # ✅ CORRECCIÓN: Check si mlflow está disponible
    if not MLFLOW_AVAILABLE:
        logger.warning(f"⚠️ MLflow not available. Skipping model registration for {model_name}")
        return None

    try:
        # Register model
        model_uri = f"runs:/{run_id}/policy"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logger.info(f"✅ Modelo RL registrado: {model_name} version {result.version}")

        # Si pasa gates, promover a Production
        if metrics:
            sortino = metrics.get('sortino_ratio', 0)
            max_dd = metrics.get('max_drawdown_pct', 0)
            calmar = metrics.get('calmar_ratio', 0)

            gates_passed = (
                sortino >= 1.3 and
                max_dd <= 15.0 and
                calmar >= 0.8
            )

            if gates_passed:
                client = MlflowClient()

                # Promover a Production
                client.transition_model_version_stage(
                    name=model_name,
                    version=result.version,
                    stage="Production"
                )

                logger.info(f"✅ Modelo promovido a Production (Sortino={sortino:.2f})")
            else:
                logger.warning(f"⚠️ Modelo NO promovido (gates failed)")

        return result

    except Exception as e:
        logger.error(f"Error registrando modelo RL: {e}")
        raise


def register_ml_model(run_id: str, model_name: str, metrics: dict = None):
    """
    Registra modelo ML en MLflow Registry

    Args:
        run_id: MLflow run ID
        model_name: "ML_LGBM_USDCOP" o "ML_XGB_USDCOP"
        metrics: Dict con accuracy, auc, etc.
    """

    # ✅ CORRECCIÓN: Check si mlflow está disponible
    if not MLFLOW_AVAILABLE:
        logger.warning(f"⚠️ MLflow not available. Skipping model registration for {model_name}")
        return None

    try:
        # Determine model type from name
        if "LGBM" in model_name:
            model_uri = f"runs:/{run_id}/lightgbm_model"
        else:
            model_uri = f"runs:/{run_id}/xgboost_model"

        result = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        logger.info(f"✅ Modelo ML registrado: {model_name} version {result.version}")

        # Gates para ML
        if metrics:
            accuracy = metrics.get('accuracy', 0)
            auc = metrics.get('auc', 0)

            gates_passed = accuracy >= 0.55 and auc >= 0.60

            if gates_passed:
                client = MlflowClient()

                client.transition_model_version_stage(
                    name=model_name,
                    version=result.version,
                    stage="Production"
                )

                logger.info(f"✅ Modelo ML promovido a Production (Acc={accuracy:.3f}, AUC={auc:.3f})")
            else:
                logger.warning(f"⚠️ Modelo ML NO promovido (gates failed)")

        return result

    except Exception as e:
        logger.error(f"Error registrando modelo ML: {e}")
        raise


def get_production_model(model_name: str):
    """
    Obtiene última versión Production de un modelo

    Args:
        model_name: "RL_PPO_USDCOP", "ML_LGBM_USDCOP", etc.

    Returns:
        ModelVersion object o None
    """

    # ✅ CORRECCIÓN: Check si mlflow está disponible
    if not MLFLOW_AVAILABLE:
        logger.warning(f"⚠️ MLflow not available. Cannot get production model {model_name}")
        return None

    try:
        client = MlflowClient()

        versions = client.get_latest_versions(
            name=model_name,
            stages=["Production"]
        )

        if versions:
            logger.info(f"✅ Modelo Production encontrado: {model_name} v{versions[0].version}")
            return versions[0]
        else:
            logger.warning(f"⚠️ No hay versión Production de {model_name}")
            return None

    except Exception as e:
        logger.error(f"Error obteniendo modelo Production: {e}")
        return None


def load_production_model(model_name: str):
    """
    Carga modelo desde MLflow Registry (Production stage)

    Args:
        model_name: Nombre del modelo

    Returns:
        Modelo cargado (PyTorch, LightGBM, XGBoost, etc.)
    """

    # ✅ CORRECCIÓN: Check si mlflow está disponible
    if not MLFLOW_AVAILABLE:
        logger.warning(f"⚠️ MLflow not available. Cannot load production model {model_name}")
        return None

    try:
        model_uri = f"models:/{model_name}/Production"

        # Detectar tipo de modelo
        if "RL" in model_name:
            model = mlflow.pytorch.load_model(model_uri)
        elif "LGBM" in model_name:
            model = mlflow.lightgbm.load_model(model_uri)
        elif "XGB" in model_name:
            model = mlflow.xgboost.load_model(model_uri)
        else:
            model = mlflow.pyfunc.load_model(model_uri)

        logger.info(f"✅ Modelo cargado desde Registry: {model_name}")

        return model

    except Exception as e:
        logger.error(f"Error cargando modelo desde Registry: {e}")
        raise


def archive_old_production_models(model_name: str, keep_last: int = 3):
    """
    Archiva versiones antiguas de Production, manteniendo solo las últimas N

    Args:
        model_name: Nombre del modelo
        keep_last: Número de versiones a mantener
    """

    # ✅ CORRECCIÓN: Check si mlflow está disponible
    if not MLFLOW_AVAILABLE:
        logger.warning(f"⚠️ MLflow not available. Cannot archive models for {model_name}")
        return None

    try:
        client = MlflowClient()

        versions = client.get_latest_versions(
            name=model_name,
            stages=["Production"]
        )

        if len(versions) > keep_last:
            # Archivar las más viejas
            to_archive = versions[keep_last:]

            for version in to_archive:
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )

                logger.info(f"📦 Archived {model_name} v{version.version}")

    except Exception as e:
        logger.error(f"Error archivando modelos: {e}")
