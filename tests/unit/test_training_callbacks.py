"""
Tests para Training Callbacks - Auto-registro de modelos.
CLAUDE-T16 | Plan Item: P1-18
Contrato: CTR-011
"""

import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import numpy as np

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import gymnasium for creating test environments
import gymnasium as gym
from gymnasium import spaces


class DummyTradingEnv(gym.Env):
    """
    Minimal gymnasium environment for testing EvalCallback.
    Satisfies SB3's type checking requirements.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        # Simple observation and action spaces matching current dimensions
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self._step_count = 0
        self._max_steps = 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = self.observation_space.sample()
        reward = np.random.uniform(-1, 1)
        terminated = self._step_count >= self._max_steps
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


class TestModelRegistrationCallback:
    """Tests para ModelRegistrationCallback."""

    @pytest.fixture
    def mock_model(self):
        """Mock de modelo SB3."""
        model = Mock()
        model.ep_info_buffer = [{"r": 100.0}, {"r": 150.0}, {"r": 120.0}]
        model.save = Mock()
        return model

    @pytest.fixture
    def temp_model_path(self, tmp_path):
        """Path temporal para guardar modelo."""
        model_path = tmp_path / "test_model.zip"
        return str(model_path)

    def test_callback_initialization(self, temp_model_path):
        """Callback debe inicializarse correctamente."""
        from ml_workflow.training_callbacks import ModelRegistrationCallback

        callback = ModelRegistrationCallback(
            model_save_path=temp_model_path,
            version="current",
            verbose=0
        )

        assert callback.model_save_path == Path(temp_model_path)
        assert callback.version == "current"
        assert callback._training_start_time is None
        assert callback._best_reward == float('-inf')

    def test_callback_on_training_start(self, temp_model_path, mock_model):
        """on_training_start debe registrar timestamp."""
        from ml_workflow.training_callbacks import ModelRegistrationCallback

        callback = ModelRegistrationCallback(
            model_save_path=temp_model_path,
            version="current",
            verbose=0
        )

        callback.init_callback(mock_model)
        callback._on_training_start()

        assert callback._training_start_time is not None
        assert isinstance(callback._training_start_time, datetime)

    def test_callback_on_step_tracks_best_reward(self, temp_model_path, mock_model):
        """on_step debe trackear mejor reward."""
        from ml_workflow.training_callbacks import ModelRegistrationCallback

        callback = ModelRegistrationCallback(
            model_save_path=temp_model_path,
            version="current",
            verbose=0
        )

        callback.init_callback(mock_model)
        callback.model = mock_model

        # Simulate step
        result = callback._on_step()

        assert result is True
        # Best reward should be mean of ep_info_buffer: (100 + 150 + 120) / 3 = 123.33
        assert callback._best_reward == pytest.approx(123.33, rel=0.01)

    def test_callback_saves_model_on_training_end(self, temp_model_path, mock_model):
        """on_training_end debe guardar el modelo."""
        from ml_workflow.training_callbacks import ModelRegistrationCallback

        callback = ModelRegistrationCallback(
            model_save_path=temp_model_path,
            version="current",
            verbose=0
        )

        callback.init_callback(mock_model)
        callback.model = mock_model
        callback._training_start_time = datetime.now()
        callback.num_timesteps = 10000

        # Mock de _register_model para evitar conexion a BD
        with patch.object(callback, '_register_model', return_value="test_model_id"):
            callback._on_training_end()

        # Verificar que el modelo fue guardado
        mock_model.save.assert_called_once_with(temp_model_path)

    def test_callback_calls_on_registration_complete(self, temp_model_path, mock_model):
        """Callback opcional debe ser llamado al completar registro."""
        from ml_workflow.training_callbacks import ModelRegistrationCallback

        registration_callback = Mock()

        callback = ModelRegistrationCallback(
            model_save_path=temp_model_path,
            version="current",
            on_registration_complete=registration_callback,
            verbose=0
        )

        callback.init_callback(mock_model)
        callback.model = mock_model
        callback._training_start_time = datetime.now()
        callback.num_timesteps = 10000

        with patch.object(callback, '_register_model', return_value="test_model_id"):
            callback._on_training_end()

        # Verificar que callback fue llamado con model_id
        registration_callback.assert_called_once_with("test_model_id")


class TestModelRegistrationEvalCallback:
    """Tests para ModelRegistrationEvalCallback."""

    def test_eval_callback_initialization(self, tmp_path):
        """EvalCallback debe inicializarse correctamente."""
        from ml_workflow.training_callbacks import ModelRegistrationEvalCallback

        # Create a real gymnasium environment for SB3
        eval_env = DummyTradingEnv()

        callback = ModelRegistrationEvalCallback(
            eval_env=eval_env,
            best_model_save_path=str(tmp_path / "best_model"),
            version="current",
            n_eval_episodes=5,
            eval_freq=1000,
            verbose=0
        )

        assert callback.version == "current"
        assert callback.n_eval_episodes == 5
        assert callback.eval_freq == 1000

        # Cleanup
        eval_env.close()


class TestCreateTrainingCallback:
    """Tests para factory function create_training_callback."""

    def test_create_callback_without_eval_env(self, tmp_path):
        """Debe crear ModelRegistrationCallback sin eval_env."""
        from ml_workflow.training_callbacks import create_training_callback, ModelRegistrationCallback

        callback = create_training_callback(
            model_save_path=str(tmp_path / "model.zip"),
            version="current",
            eval_env=None,
            use_eval_callback=True
        )

        assert isinstance(callback, ModelRegistrationCallback)

    def test_create_callback_with_eval_env(self, tmp_path):
        """Debe crear ModelRegistrationEvalCallback con eval_env."""
        from ml_workflow.training_callbacks import create_training_callback, ModelRegistrationEvalCallback

        # Create a real gymnasium environment for SB3
        eval_env = DummyTradingEnv()

        callback = create_training_callback(
            model_save_path=str(tmp_path / "model.zip"),
            version="current",
            eval_env=eval_env,
            use_eval_callback=True
        )

        assert isinstance(callback, ModelRegistrationEvalCallback)

        # Cleanup
        eval_env.close()


class TestDBWrapper:
    """Tests para la conexion a BD del callback."""

    def test_callback_works_without_db(self, tmp_path):
        """Callback debe funcionar sin conexion a BD."""
        from ml_workflow.training_callbacks import ModelRegistrationCallback

        callback = ModelRegistrationCallback(
            model_save_path=str(tmp_path / "model.zip"),
            version="current",
            db_connection_string=None,  # Sin conexion
            verbose=0
        )

        # _get_db_connection debe retornar None
        conn = callback._get_db_connection()
        assert conn is None

    def test_callback_handles_db_error_gracefully(self, tmp_path):
        """Callback debe manejar errores de BD gracefully."""
        from ml_workflow.training_callbacks import ModelRegistrationCallback

        mock_model = Mock()
        mock_model.ep_info_buffer = []
        mock_model.save = Mock()

        callback = ModelRegistrationCallback(
            model_save_path=str(tmp_path / "model.zip"),
            version="current",
            db_connection_string="postgresql://invalid:invalid@localhost/invalid",
            verbose=0
        )

        callback.init_callback(mock_model)
        callback.model = mock_model
        callback._training_start_time = datetime.now()
        callback.num_timesteps = 10000

        # Mock _register_model para evitar conexion real
        with patch.object(callback, '_register_model', return_value=None):
            callback._on_training_end()

        # Modelo debe haberse guardado de todas formas
        mock_model.save.assert_called_once()


class TestTrainingInfoPreservation:
    """Tests para preservacion de training_info."""

    def test_training_info_merged_with_metrics(self, tmp_path):
        """training_info debe combinarse con metricas finales."""
        from ml_workflow.training_callbacks import ModelRegistrationCallback

        mock_model = Mock()
        mock_model.ep_info_buffer = [{"r": 100.0}]
        mock_model.save = Mock()

        custom_info = {
            "dataset_id": "dataset_123",
            "experiment_name": "test_experiment"
        }

        callback = ModelRegistrationCallback(
            model_save_path=str(tmp_path / "model.zip"),
            version="current",
            training_info=custom_info,
            verbose=0
        )

        callback.init_callback(mock_model)
        callback.model = mock_model
        callback._training_start_time = datetime.now()
        callback.num_timesteps = 10000

        # Llamar _on_step para que actualice best_reward
        callback._on_step()

        with patch.object(callback, '_register_model') as mock_register:
            mock_register.return_value = "test_id"
            callback._on_training_end()

        # Verificar que _final_metrics contiene los datos esperados
        assert callback._final_metrics["total_timesteps"] == 10000
        assert callback._final_metrics["best_mean_reward"] == 100.0
        assert "training_duration_seconds" in callback._final_metrics
        assert "training_start" in callback._final_metrics
        assert "training_end" in callback._final_metrics
