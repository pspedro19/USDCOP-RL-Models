"""
Training Callbacks - Auto-registro de modelos post-entrenamiento.
CLAUDE-T16 | Plan Item: P1-18
Contrato: CTR-011

Proporciona callbacks para Stable Baselines 3 que:
1. Registran el modelo en model_registry al terminar training
2. Guardan metricas de backtest con el modelo
3. Notifican al sistema que hay un nuevo modelo disponible
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Callable
import json

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

logger = logging.getLogger(__name__)


class ModelRegistrationCallback(BaseCallback):
    """
    Callback que registra automáticamente el modelo al terminar training.
    Contrato: CTR-011

    Uso:
        from src.ml_workflow.training_callbacks import ModelRegistrationCallback

        callback = ModelRegistrationCallback(
            model_save_path="models/ppo_primary.zip",
            version="current",
            db_connection_string=os.environ.get("DATABASE_URL")
        )

        model.learn(
            total_timesteps=100_000,
            callback=callback
        )
        # Al terminar, el modelo queda registrado en model_registry
    """

    def __init__(
        self,
        model_save_path: str,
        version: str,
        db_connection_string: Optional[str] = None,
        training_info: Optional[Dict[str, Any]] = None,
        on_registration_complete: Optional[Callable[[str], None]] = None,
        verbose: int = 1
    ):
        """
        Inicializa el callback.

        Args:
            model_save_path: Path donde se guardará el modelo .zip
            version: Versión del modelo (e.g., "v1")
            db_connection_string: URL de conexión a PostgreSQL
            training_info: Info adicional del training (dataset_id, etc.)
            on_registration_complete: Callback opcional al completar registro
            verbose: Nivel de verbosidad
        """
        super().__init__(verbose)
        self.model_save_path = Path(model_save_path)
        self.version = version
        self.db_connection_string = db_connection_string or os.environ.get("DATABASE_URL")
        self.training_info = training_info or {}
        self.on_registration_complete = on_registration_complete

        self._training_start_time: Optional[datetime] = None
        self._best_reward: float = float('-inf')
        self._final_metrics: Dict[str, Any] = {}

    def _on_training_start(self) -> None:
        """Llamado al inicio del training."""
        self._training_start_time = datetime.now()
        if self.verbose > 0:
            logger.info(f"Training started at {self._training_start_time}")
            logger.info(f"Model will be saved to: {self.model_save_path}")

    def _on_step(self) -> bool:
        """Llamado en cada step. Trackea mejor reward."""
        # Obtener reward del último episodio si está disponible
        if len(self.model.ep_info_buffer) > 0:
            episode_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            if episode_rewards:
                mean_reward = sum(episode_rewards) / len(episode_rewards)
                if mean_reward > self._best_reward:
                    self._best_reward = mean_reward
        return True

    def _on_training_end(self) -> None:
        """Llamado al finalizar el training. Registra el modelo."""
        training_end_time = datetime.now()
        training_duration = (training_end_time - self._training_start_time).total_seconds()

        if self.verbose > 0:
            logger.info(f"Training completed in {training_duration:.2f} seconds")

        # Guardar el modelo primero
        self.model.save(str(self.model_save_path))
        if self.verbose > 0:
            logger.info(f"Model saved to {self.model_save_path}")

        # Preparar métricas de training
        self._final_metrics = {
            "training_duration_seconds": training_duration,
            "total_timesteps": self.num_timesteps,
            "best_mean_reward": self._best_reward if self._best_reward != float('-inf') else None,
            "training_start": self._training_start_time.isoformat(),
            "training_end": training_end_time.isoformat(),
        }

        # Registrar en base de datos
        model_id = self._register_model()

        if model_id and self.on_registration_complete:
            self.on_registration_complete(model_id)

    def _register_model(self) -> Optional[str]:
        """Registra el modelo en la base de datos."""
        try:
            from src.ml_workflow.model_registry import ModelRegistry

            # Conectar a BD
            conn = self._get_db_connection()
            registry = ModelRegistry(conn=conn)

            # Combinar training_info con métricas finales
            full_training_info = {
                **self.training_info,
                **self._final_metrics
            }

            # Registrar modelo
            model_id = registry.register_model(
                model_path=self.model_save_path,
                version=self.version,
                training_info=full_training_info
            )

            if self.verbose > 0:
                logger.info(f"Model registered successfully: {model_id}")

            # Cerrar conexión
            if conn:
                conn.close()

            return model_id

        except ImportError as e:
            logger.warning(f"Could not import ModelRegistry: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None

    def _get_db_connection(self):
        """Obtiene conexión a la base de datos."""
        if not self.db_connection_string:
            logger.warning("No database connection string provided")
            return None

        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(self.db_connection_string)
            conn.autocommit = True

            # Crear wrapper simple para compatibilidad con ModelRegistry
            class DBWrapper:
                def __init__(self, connection):
                    self._conn = connection
                    self._cursor = connection.cursor(cursor_factory=RealDictCursor)

                def execute(self, query, params=None):
                    self._cursor.execute(query, params)

                def fetchone(self, query, params=None):
                    self._cursor.execute(query, params)
                    return self._cursor.fetchone()

                def fetchall(self, query, params=None):
                    self._cursor.execute(query, params)
                    return self._cursor.fetchall()

                def close(self):
                    self._cursor.close()
                    self._conn.close()

            return DBWrapper(conn)

        except ImportError:
            logger.warning("psycopg2 not installed")
            return None
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None


class ModelRegistrationEvalCallback(EvalCallback):
    """
    EvalCallback extendido que registra el mejor modelo automáticamente.

    Combina evaluación periódica con registro automático del mejor modelo.

    Uso:
        callback = ModelRegistrationEvalCallback(
            eval_env=eval_env,
            best_model_save_path="models/best_ppo_primary",
            version="current",
            n_eval_episodes=10,
            eval_freq=5000,
        )

        model.learn(total_timesteps=100_000, callback=callback)
    """

    def __init__(
        self,
        eval_env,
        best_model_save_path: str,
        version: str,
        db_connection_string: Optional[str] = None,
        training_info: Optional[Dict[str, Any]] = None,
        n_eval_episodes: int = 10,
        eval_freq: int = 10000,
        verbose: int = 1,
        **kwargs
    ):
        """
        Inicializa el callback de evaluación con registro.

        Args:
            eval_env: Entorno de evaluación
            best_model_save_path: Path para guardar el mejor modelo
            version: Versión del modelo
            db_connection_string: URL de conexión a PostgreSQL
            training_info: Info adicional del training
            n_eval_episodes: Número de episodios para evaluación
            eval_freq: Frecuencia de evaluación (en timesteps)
            verbose: Nivel de verbosidad
            **kwargs: Argumentos adicionales para EvalCallback
        """
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            verbose=verbose,
            **kwargs
        )

        self.version = version
        self.db_connection_string = db_connection_string or os.environ.get("DATABASE_URL")
        self.training_info = training_info or {}
        self._registered = False
        self._eval_results: list = []

    def _on_step(self) -> bool:
        """Override para trackear evaluaciones."""
        result = super()._on_step()

        # Guardar resultados de evaluación
        if hasattr(self, 'last_mean_reward') and self.last_mean_reward is not None:
            self._eval_results.append({
                "timestep": self.num_timesteps,
                "mean_reward": self.last_mean_reward,
                "std_reward": getattr(self, 'last_std_reward', 0),
            })

        return result

    def _on_training_end(self) -> None:
        """Registra el mejor modelo al finalizar."""
        if self._registered:
            return

        best_model_path = Path(self.best_model_save_path + ".zip")
        if not best_model_path.exists():
            logger.warning(f"Best model not found at {best_model_path}")
            return

        try:
            from src.ml_workflow.model_registry import ModelRegistry

            # Preparar métricas
            training_info = {
                **self.training_info,
                "total_timesteps": self.num_timesteps,
                "best_mean_reward": self.best_mean_reward,
                "n_eval_episodes": self.n_eval_episodes,
                "eval_results": self._eval_results[-10:],  # Últimas 10 evaluaciones
            }

            # Conectar y registrar
            conn = self._get_db_connection()
            registry = ModelRegistry(conn=conn)

            model_id = registry.register_model(
                model_path=best_model_path,
                version=self.version,
                training_info=training_info
            )

            self._registered = True

            if self.verbose > 0:
                logger.info(f"Best model registered: {model_id}")
                logger.info(f"Best mean reward: {self.best_mean_reward:.2f}")

            if conn:
                conn.close()

        except Exception as e:
            logger.error(f"Failed to register best model: {e}")

    def _get_db_connection(self):
        """Obtiene conexión a la base de datos."""
        # Mismo código que ModelRegistrationCallback
        if not self.db_connection_string:
            return None

        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor

            conn = psycopg2.connect(self.db_connection_string)
            conn.autocommit = True

            class DBWrapper:
                def __init__(self, connection):
                    self._conn = connection
                    self._cursor = connection.cursor(cursor_factory=RealDictCursor)

                def execute(self, query, params=None):
                    self._cursor.execute(query, params)

                def fetchone(self, query, params=None):
                    self._cursor.execute(query, params)
                    return self._cursor.fetchone()

                def fetchall(self, query, params=None):
                    self._cursor.execute(query, params)
                    return self._cursor.fetchall()

                def close(self):
                    self._cursor.close()
                    self._conn.close()

            return DBWrapper(conn)

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return None


def create_training_callback(
    model_save_path: str,
    version: str,
    eval_env=None,
    db_connection_string: Optional[str] = None,
    training_info: Optional[Dict[str, Any]] = None,
    use_eval_callback: bool = True,
    n_eval_episodes: int = 10,
    eval_freq: int = 10000,
) -> BaseCallback:
    """
    Factory function para crear el callback de training apropiado.

    Args:
        model_save_path: Path para guardar el modelo
        version: Versión del modelo
        eval_env: Entorno de evaluación (opcional)
        db_connection_string: URL de conexión a PostgreSQL
        training_info: Info adicional del training
        use_eval_callback: Si True y hay eval_env, usa ModelRegistrationEvalCallback
        n_eval_episodes: Número de episodios para evaluación
        eval_freq: Frecuencia de evaluación

    Returns:
        Callback apropiado para el training

    Example:
        callback = create_training_callback(
            model_save_path="models/ppo_primary.zip",
            version="current",
            eval_env=eval_env,
        )

        model.learn(total_timesteps=100_000, callback=callback)
    """
    if use_eval_callback and eval_env is not None:
        return ModelRegistrationEvalCallback(
            eval_env=eval_env,
            best_model_save_path=model_save_path.replace(".zip", ""),
            version=version,
            db_connection_string=db_connection_string,
            training_info=training_info,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
        )
    else:
        return ModelRegistrationCallback(
            model_save_path=model_save_path,
            version=version,
            db_connection_string=db_connection_string,
            training_info=training_info,
        )
