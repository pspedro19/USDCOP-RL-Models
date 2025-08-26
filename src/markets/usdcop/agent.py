"""
USDCOP Trading Agent - Production-Ready Multi-Algorithm RL System
================================================================
Sistema completo de agentes RL para trading algorítmico con
soporte para múltiples algoritmos, gestión de riesgo y optimización.

Algoritmos soportados:
- PPO (Proximal Policy Optimization) - Recomendado para Forex
- DQN (Deep Q-Network) con Double DQN y Dueling
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic) para acciones continuas
- TD3 (Twin Delayed DDPG) para estabilidad

Características:
- Entrenamiento robusto con early stopping y checkpoints
- Hyperparameter tuning automático con Optuna
- Risk management y position sizing dinámico
- Backtesting con métricas detalladas
- CLI completo para train/eval/predict
- Logging estructurado y telemetría
- Soporte para ensemble de modelos
- Compatible con Stable-Baselines3 v2.0+

Autor: USDCOP Trading System
Versión: 3.0.0
Licencia: MIT
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import pickle
import shutil
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Type, Callable
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Numerical and data processing
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Gym/Gymnasium
try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore

# Stable Baselines 3
try:
    from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback, StopTrainingOnRewardThreshold,
        StopTrainingOnNoModelImprovement, CheckpointCallback, CallbackList
    )
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.preprocessing import get_flattened_obs_dim
    from stable_baselines3.common.logger import configure as sb3_configure_logger
    HAS_SB3 = True
except ImportError as e:
    HAS_SB3 = False
    SB3_IMPORT_ERROR = e

# Hyperparameter optimization
try:
    import optuna
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
    from optuna.samplers import TPESampler, CmaEsSampler
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Progress bars
from tqdm import tqdm

# Local imports - ajustar según tu estructura
try:
    from src.markets.usdcop.environment import USDCOPTradingEnv, make_usdcop_env, EnvConfig
    from src.markets.usdcop.features import FeatureEngineer
    from src.core.risk_manager import RiskManager
    from src.utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown
    from src.utils.logger import setup_logger
except ImportError:
    # Fallback para desarrollo/testing
    pass

# Configurar logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s %(name)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Suprimir warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =====================================================
# CONSTANTES Y CONFIGURACIÓN GLOBAL
# =====================================================

# Columnas requeridas en los datos
REQUIRED_COLS = ["time", "open", "high", "low", "close", "volume"]

# Algoritmos soportados
SUPPORTED_ALGORITHMS = ["PPO", "DQN", "A2C", "SAC", "TD3"]

# Métricas de evaluación
EVAL_METRICS = [
    "total_reward", "episode_length", "n_trades", "win_rate", 
    "profit_factor", "sharpe_ratio", "max_drawdown", "final_equity"
]


# =====================================================
# CONFIGURACIÓN DE AGENTE
# =====================================================

@dataclass
class AgentConfig:
    """Configuración completa del agente RL"""
    
    # === Algoritmo y modelo ===
    algorithm: str = "PPO"  # PPO, DQN, A2C, SAC, TD3
    model_name: str = "usdcop_agent"
    
    # === Entrenamiento ===
    total_timesteps: int = 1_000_000
    learning_rate: Union[float, Callable] = 3e-4
    batch_size: int = 64
    n_epochs: int = 10  # Para PPO/A2C
    n_steps: int = 2048  # Steps antes de update
    
    # === Arquitectura de red ===
    policy: str = "MlpPolicy"  # MlpPolicy, CnnPolicy, MultiInputPolicy
    net_arch: Optional[List[int]] = None  # [256, 256] por defecto
    activation_fn: str = "tanh"  # relu, tanh, elu, leaky_relu
    features_extractor_class: Optional[Type] = None
    features_extractor_kwargs: Optional[Dict] = None
    
    # === Exploración (DQN) ===
    exploration_fraction: float = 0.1
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    
    # === Replay buffer (DQN/SAC/TD3) ===
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    train_freq: Union[int, Tuple[int, str]] = 4
    gradient_steps: int = 1
    target_update_interval: int = 1000
    
    # === PPO/A2C específico ===
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: Union[float, Callable] = 0.2
    clip_range_vf: Optional[Union[float, Callable]] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # === SAC/TD3 específico ===
    tau: float = 0.005  # Polyak averaging
    use_sde: bool = False  # State Dependent Exploration
    sde_sample_freq: int = -1
    
    # === Normalización ===
    normalize_observations: bool = True
    normalize_rewards: bool = True
    norm_obs_clip: float = 10.0
    norm_reward_clip: float = 10.0
    
    # === Evaluación ===
    eval_freq: int = 10_000
    n_eval_episodes: int = 10
    eval_deterministic: bool = True
    
    # === Callbacks y logging ===
    checkpoint_freq: int = 50_000
    checkpoint_path: str = "./models/checkpoints"
    log_path: str = "./logs"
    tensorboard_log: str = "./tensorboard"
    verbose: int = 1
    
    # === Early stopping ===
    early_stop_patience: int = 10
    early_stop_min_delta: float = 0.01
    reward_threshold: Optional[float] = None
    
    # === Risk management ===
    max_drawdown_threshold: float = 0.20
    min_win_rate_threshold: float = 0.40
    max_consecutive_losses: int = 10
    position_sizing: str = "fixed"  # fixed, kelly, risk_parity, volatility
    max_position_size: float = 1.0
    
    # === Hardware y paralelización ===
    device: str = "auto"  # cuda, cpu, auto
    n_envs: int = 4  # Ambientes paralelos
    vec_env_type: str = "dummy"  # dummy, subproc
    
    # === Data split ===
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # === Environment config ===
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validación y configuración post-init"""
        # Validar algoritmo
        if self.algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(f"Algorithm {self.algorithm} not supported. "
                           f"Choose from: {SUPPORTED_ALGORITHMS}")
        
        # Configurar arquitectura por defecto
        if self.net_arch is None:
            if self.algorithm in ["PPO", "A2C"]:
                self.net_arch = [dict(pi=[256, 256], vf=[256, 256])]
            else:
                self.net_arch = [256, 256]
        
        # Configurar device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Crear directorios
        for path in [self.checkpoint_path, self.log_path, self.tensorboard_log]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return asdict(self)
    
    def save(self, filepath: str) -> None:
        """Guardar configuración"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'AgentConfig':
        """Cargar configuración"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# =====================================================
# EXTRACTORES DE FEATURES PERSONALIZADOS
# =====================================================

class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extractor de features con atención y normalización batch
    """
    
    def __init__(self, observation_space: gym.Space, 
                 features_dim: int = 256,
                 dropout_rate: float = 0.2):
        super().__init__(observation_space, features_dim)
        
        n_input = get_flattened_obs_dim(observation_space)
        
        # Red principal
        self.network = nn.Sequential(
            # Capa de entrada expandida
            nn.Linear(n_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Capa de procesamiento
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Capa de atención simplificada
            nn.Linear(512, 512),
            nn.Tanh(),
            
            # Proyección final
            nn.Linear(512, features_dim),
            nn.BatchNorm1d(features_dim),
            nn.ReLU()
        )
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización de pesos Xavier/He"""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class LSTMTradingExtractor(BaseFeaturesExtractor):
    """
    Extractor con LSTM bidireccional para series temporales
    """
    
    def __init__(self, observation_space: gym.Space,
                 features_dim: int = 256,
                 lstm_hidden_size: int = 128,
                 n_lstm_layers: int = 2,
                 dropout_rate: float = 0.2):
        super().__init__(observation_space, features_dim)
        
        n_input = get_flattened_obs_dim(observation_space)
        
        # LSTM bidireccional
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=lstm_hidden_size,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if n_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, features_dim),  # *2 por bidireccional
            nn.LayerNorm(features_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización LSTM"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Reshape para LSTM
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
        
        lstm_out, (hidden, cell) = self.lstm(observations)
        
        # Concatenar hidden states de ambas direcciones
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        return self.projection(hidden)


class TransformerTradingExtractor(BaseFeaturesExtractor):
    """
    Extractor con arquitectura Transformer para patrones complejos
    """
    
    def __init__(self, observation_space: gym.Space,
                 features_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 2,
                 dropout_rate: float = 0.1):
        super().__init__(observation_space, features_dim)
        
        n_input = get_flattened_obs_dim(observation_space)
        
        # Embedding de entrada
        self.input_projection = nn.Linear(n_input, features_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, features_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=features_dim,
            nhead=n_heads,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(features_dim),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_projection(observations)
        
        # Add positional encoding
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x + self.pos_encoding
        
        # Transformer
        x = self.transformer(x)
        
        # Global pooling
        x = x.mean(dim=1) if x.dim() == 3 else x
        
        return self.output_projection(x)


# =====================================================
# CALLBACKS PERSONALIZADOS
# =====================================================

class TradingMetricsCallback(BaseCallback):
    """
    Callback avanzado para métricas de trading
    """
    
    def __init__(self, log_freq: int = 1000, 
                 save_freq: int = 10000,
                 save_path: str = "./metrics",
                 verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.metrics_history = []
        self.trade_history = []
        
    def _on_step(self) -> bool:
        # Log periódico
        if self.n_calls % self.log_freq == 0:
            metrics = self._collect_metrics()
            if metrics:
                self._log_metrics(metrics)
                self.metrics_history.append(metrics)
        
        # Guardar histórico
        if self.n_calls % self.save_freq == 0:
            self._save_metrics()
        
        return True
    
    def _collect_metrics(self) -> Optional[Dict[str, float]]:
        """Recolectar métricas de todos los environments"""
        try:
            all_metrics = []
            
            if hasattr(self.training_env, 'envs'):
                for env in self.training_env.envs:
                    if hasattr(env, 'get_wrapper_attr'):
                        stats = env.get_wrapper_attr('get_trading_statistics')()
                        if stats:
                            all_metrics.append(stats)
                    elif hasattr(env, 'get_trading_statistics'):
                        stats = env.get_trading_statistics()
                        if stats:
                            all_metrics.append(stats)
            
            if not all_metrics:
                return None
            
            # Promediar métricas
            avg_metrics = {}
            for key in all_metrics[0].keys():
                values = [m.get(key, 0) for m in all_metrics]
                avg_metrics[key] = np.mean(values)
            
            avg_metrics['timestep'] = self.num_timesteps
            avg_metrics['episode'] = len(self.episode_rewards)
            
            return avg_metrics
            
        except Exception as e:
            logger.warning(f"Error collecting metrics: {e}")
            return None
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to tensorboard"""
        for key, value in metrics.items():
            if key not in ['timestep', 'episode']:
                self.logger.record(f"trading/{key}", value)
    
    def _save_metrics(self):
        """Guardar métricas a disco"""
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            filepath = self.save_path / f"metrics_{self.num_timesteps}.csv"
            df.to_csv(filepath, index=False)
            
            # Guardar resumen JSON
            summary = {
                'timesteps': self.num_timesteps,
                'episodes': len(self.episode_rewards),
                'best_sharpe': df['sharpe_ratio'].max() if 'sharpe_ratio' in df else 0,
                'avg_win_rate': df['win_rate'].mean() if 'win_rate' in df else 0,
                'final_metrics': metrics.to_dict() if hasattr(metrics, 'to_dict') else metrics
            }
            
            with open(self.save_path / f"summary_{self.num_timesteps}.json", 'w') as f:
                json.dump(summary, f, indent=2)
    
    def _on_training_end(self):
        """Guardar métricas finales"""
        self._save_metrics()
        logger.info(f"Saved {len(self.metrics_history)} metric records")


class RiskManagementCallback(BaseCallback):
    """
    Callback para gestión de riesgo con múltiples criterios
    """
    
    def __init__(self, 
                 max_drawdown: float = 0.20,
                 min_win_rate: float = 0.40,
                 max_consecutive_losses: int = 10,
                 check_freq: int = 5000,
                 warmup_steps: int = 10000):
        super().__init__()
        self.max_drawdown = max_drawdown
        self.min_win_rate = min_win_rate
        self.max_consecutive_losses = max_consecutive_losses
        self.check_freq = check_freq
        self.warmup_steps = warmup_steps
        
        self.consecutive_losses = 0
        self.risk_violations = 0
        
    def _on_step(self) -> bool:
        # Skip durante warmup
        if self.num_timesteps < self.warmup_steps:
            return True
        
        if self.n_calls % self.check_freq == 0:
            return self._check_risk_conditions()
        
        return True
    
    def _check_risk_conditions(self) -> bool:
        """Verificar condiciones de riesgo"""
        stats = self._get_trading_stats()
        
        if not stats:
            return True
        
        violations = []
        
        # Check drawdown
        current_dd = stats.get('max_drawdown', 0)
        if current_dd > self.max_drawdown:
            violations.append(f"Max drawdown exceeded: {current_dd:.2%} > {self.max_drawdown:.2%}")
        
        # Check win rate
        win_rate = stats.get('win_rate', 1.0)
        if win_rate < self.min_win_rate and stats.get('n_trades', 0) > 20:
            violations.append(f"Low win rate: {win_rate:.2%} < {self.min_win_rate:.2%}")
        
        # Check consecutive losses
        if self.consecutive_losses > self.max_consecutive_losses:
            violations.append(f"Consecutive losses: {self.consecutive_losses} > {self.max_consecutive_losses}")
        
        if violations:
            self.risk_violations += 1
            logger.warning(f"Risk violations ({self.risk_violations}): {'; '.join(violations)}")
            
            # Detener si hay múltiples violaciones
            if self.risk_violations >= 3:
                logger.error("Multiple risk violations. Stopping training.")
                return False
        
        return True
    
    def _get_trading_stats(self) -> Optional[Dict]:
        """Obtener estadísticas de trading"""
        try:
            if hasattr(self.training_env, 'envs'):
                env = self.training_env.envs[0]
                if hasattr(env, 'get_wrapper_attr'):
                    return env.get_wrapper_attr('get_trading_statistics')()
                elif hasattr(env, 'get_trading_statistics'):
                    return env.get_trading_statistics()
        except Exception as e:
            logger.debug(f"Error getting trading stats: {e}")
        return None


class AdaptiveLearningRateCallback(BaseCallback):
    """
    Callback para ajustar learning rate dinámicamente
    """
    
    def __init__(self, 
                 patience: int = 10,
                 factor: float = 0.5,
                 min_lr: float = 1e-6,
                 check_freq: int = 10000):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.check_freq = check_freq
        
        self.best_reward = -np.inf
        self.patience_counter = 0
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            self._adjust_learning_rate()
        return True
    
    def _adjust_learning_rate(self):
        """Ajustar learning rate basado en performance"""
        # Obtener reward promedio reciente
        if hasattr(self.model, 'ep_info_buffer') and len(self.model.ep_info_buffer) > 0:
            recent_rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            avg_reward = np.mean(recent_rewards)
            
            # Check mejora
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Reducir LR si no hay mejora
            if self.patience_counter >= self.patience:
                current_lr = self.model.learning_rate
                if isinstance(current_lr, float):
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    self.model.learning_rate = new_lr
                    logger.info(f"Reduced learning rate: {current_lr:.2e} -> {new_lr:.2e}")
                    self.patience_counter = 0


# =====================================================
# AGENTE BASE
# =====================================================

class BaseRLTradingAgent(ABC):
    """Clase base para todos los agentes de trading"""
    
    def __init__(self, config: AgentConfig):
        """
        Inicializar agente
        
        Args:
            config: Configuración del agente
        """
        self.config = config
        self.model = None
        self.env = None
        self.eval_env = None
        self.vec_normalize = None
        
        # Métricas
        self.training_history = []
        self.evaluation_results = []
        
        # Estado
        self.is_trained = False
        self.training_start_time = None
        self.total_training_time = 0
        
        # Logger
        self.logger = logger
        
        # Validar dependencias
        if not HAS_SB3:
            raise ImportError(
                "stable-baselines3 not installed. "
                "Install with: pip install 'stable-baselines3[extra]>=2.0.0'"
            )
        
        self.logger.info(f"Initialized {self.__class__.__name__} "
                        f"with algorithm={config.algorithm}")
    
    @abstractmethod
    def create_model(self, env: gym.Env) -> Any:
        """Crear modelo específico del algoritmo"""
        pass
    
    def train(self, 
              train_data: pd.DataFrame,
              val_data: Optional[pd.DataFrame] = None,
              resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Entrenar el agente
        
        Args:
            train_data: Datos de entrenamiento
            val_data: Datos de validación (opcional)
            resume_from: Path para continuar entrenamiento
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        self.logger.info("Starting training...")
        self.training_start_time = time.time()
        
        try:
            # Crear environments
            self.env = self._create_train_env(train_data)
            if val_data is not None:
                self.eval_env = self._create_eval_env(val_data)
            
            # Crear o cargar modelo
            if resume_from and os.path.exists(resume_from):
                self.load(resume_from)
                self.logger.info(f"Resumed from checkpoint: {resume_from}")
            else:
                self.model = self.create_model(self.env)
            
            # Callbacks
            callbacks = self._create_callbacks()
            
            # Entrenar
            self.model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False if resume_from else True
            )
            
            self.is_trained = True
            self.total_training_time = time.time() - self.training_start_time
            
            # Guardar modelo final
            final_path = os.path.join(self.config.checkpoint_path, "final_model")
            self.save(final_path)
            
            # Recopilar métricas finales
            results = self._get_training_summary()
            self.logger.info(f"Training completed in {self.total_training_time:.2f}s")
            
            return results
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            self.save(os.path.join(self.config.checkpoint_path, "interrupted_model"))
            raise
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
    
    def predict(self, 
                observation: np.ndarray,
                state: Optional[np.ndarray] = None,
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predecir acción
        
        Args:
            observation: Observación actual
            state: Estado interno (para RNNs)
            deterministic: Si usar política determinística
            
        Returns:
            action: Acción predicha
            state: Nuevo estado interno
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Normalizar observación si es necesario
        if self.vec_normalize is not None:
            observation = self.vec_normalize.normalize_obs(observation)
        
        action, state = self.model.predict(
            observation, 
            state=state,
            deterministic=deterministic
        )
        
        return action, state
    
    def evaluate(self, 
                 test_data: pd.DataFrame,
                 n_episodes: int = 10,
                 deterministic: bool = True,
                 render: bool = False) -> Dict[str, Any]:
        """
        Evaluar el agente
        
        Args:
            test_data: Datos de prueba
            n_episodes: Número de episodios
            deterministic: Si usar política determinística
            render: Si renderizar environment
            
        Returns:
            Diccionario con métricas de evaluación
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.logger.info(f"Evaluating on {len(test_data)} samples...")
        
        # Crear environment de evaluación
        eval_env = self._create_eval_env(test_data)
        
        # Métricas agregadas
        all_rewards = []
        all_lengths = []
        all_metrics = []
        
        for episode in tqdm(range(n_episodes), desc="Evaluating"):
            obs, info = eval_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if render:
                    eval_env.render()
            
            all_rewards.append(episode_reward)
            all_lengths.append(episode_length)
            
            # Métricas de trading
            if hasattr(eval_env, 'get_trading_statistics'):
                metrics = eval_env.get_trading_statistics()
                all_metrics.append(metrics)
        
        # Calcular estadísticas
        results = {
            'n_episodes': n_episodes,
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'mean_length': np.mean(all_lengths),
        }
        
        # Agregar métricas de trading
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m.get(key, 0) for m in all_metrics]
                results[f'mean_{key}'] = np.mean(values)
                results[f'std_{key}'] = np.std(values)
        
        self.evaluation_results.append(results)
        
        return results
    
    def backtest(self, 
                 data: pd.DataFrame,
                 initial_balance: float = 10000,
                 commission: float = 0.001,
                 spread: float = 0.0001) -> Dict[str, Any]:
        """
        Realizar backtest detallado
        
        Args:
            data: Datos históricos
            initial_balance: Balance inicial
            commission: Comisión por operación
            spread: Spread del instrumento
            
        Returns:
            Resultados del backtest con trades y métricas
        """
        self.logger.info(f"Running backtest on {len(data)} bars...")
        
        # Configurar environment para backtest
        env_config = {
            'initial_balance': initial_balance,
            'commission': commission,
            'spread': spread,
            'training_mode': False
        }
        
        backtest_env = self._create_eval_env(data, **env_config)
        
        # Ejecutar episodio completo
        obs, info = backtest_env.reset()
        done = False
        
        observations = []
        actions = []
        rewards = []
        infos = []
        
        while not done:
            action, _ = self.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = backtest_env.step(action)
            done = terminated or truncated
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
        
        # Recopilar resultados
        results = {
            'initial_balance': initial_balance,
            'final_balance': info.get('balance', initial_balance),
            'total_return': (info.get('balance', initial_balance) - initial_balance) / initial_balance,
            'n_trades': info.get('n_trades', 0),
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'infos': infos
        }
        
        # Métricas detalladas
        if hasattr(backtest_env, 'get_trading_statistics'):
            stats = backtest_env.get_trading_statistics()
            results.update(stats)
        
        # Histórico de trades
        if hasattr(backtest_env, 'trade_history'):
            results['trades'] = backtest_env.trade_history
        
        # Equity curve
        if hasattr(backtest_env, 'equity_curve'):
            results['equity_curve'] = backtest_env.equity_curve
        
        return results
    
    def save(self, path: str) -> None:
        """
        Guardar modelo y configuración
        
        Args:
            path: Ruta para guardar (sin extensión)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Guardar modelo
        self.model.save(f"{path}.zip")
        
        # Guardar normalizador si existe
        if self.vec_normalize is not None:
            self.vec_normalize.save(f"{path}_vecnormalize.pkl")
        
        # Guardar configuración y metadatos
        metadata = {
            'config': self.config.to_dict(),
            'algorithm': self.config.algorithm,
            'is_trained': self.is_trained,
            'total_training_time': self.total_training_time,
            'training_history': self.training_history[-100:],  # Últimas 100 entradas
            'evaluation_results': self.evaluation_results,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str, env: Optional[gym.Env] = None) -> None:
        """
        Cargar modelo y configuración
        
        Args:
            path: Ruta del modelo (sin extensión)
            env: Environment opcional para el modelo
        """
        # Cargar metadatos
        with open(f"{path}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Restaurar configuración
        self.config = AgentConfig(**metadata['config'])
        self.is_trained = metadata.get('is_trained', True)
        self.total_training_time = metadata.get('total_training_time', 0)
        self.training_history = metadata.get('training_history', [])
        self.evaluation_results = metadata.get('evaluation_results', [])
        
        # Cargar modelo
        model_class = self._get_model_class()
        self.model = model_class.load(f"{path}.zip", env=env, device=self.config.device)
        
        # Cargar normalizador si existe
        vecnorm_path = f"{path}_vecnormalize.pkl"
        if os.path.exists(vecnorm_path):
            self.vec_normalize = VecNormalize.load(vecnorm_path, env)
        
        self.logger.info(f"Model loaded from {path}")
    
    def _create_train_env(self, data: pd.DataFrame) -> gym.Env:
        """Crear environment de entrenamiento"""
        # Función para crear un environment
        def make_env():
            env = self._make_single_env(data, training=True)
            return Monitor(env)
        
        # Crear environments paralelos
        if self.config.n_envs > 1:
            if self.config.vec_env_type == "subproc":
                vec_env = SubprocVecEnv([make_env for _ in range(self.config.n_envs)])
            else:
                vec_env = DummyVecEnv([make_env for _ in range(self.config.n_envs)])
        else:
            vec_env = DummyVecEnv([make_env])
        
        # Normalización
        if self.config.normalize_observations or self.config.normalize_rewards:
            self.vec_normalize = VecNormalize(
                vec_env,
                norm_obs=self.config.normalize_observations,
                norm_reward=self.config.normalize_rewards,
                clip_obs=self.config.norm_obs_clip,
                clip_reward=self.config.norm_reward_clip
            )
            return self.vec_normalize
        
        return vec_env
    
    def _create_eval_env(self, data: pd.DataFrame, **kwargs) -> gym.Env:
        """Crear environment de evaluación"""
        env = self._make_single_env(data, training=False, **kwargs)
        return Monitor(env)
    
    def _make_single_env(self, data: pd.DataFrame, training: bool = True, **kwargs) -> gym.Env:
        """Crear un solo environment"""
        # Configuración del environment
        env_kwargs = self.config.env_kwargs.copy()
        env_kwargs.update(kwargs)
        env_kwargs['training_mode'] = training
        
        # Crear environment
        # Asumiendo que tienes una función make_usdcop_env
        env = make_usdcop_env(
            df=data,
            **env_kwargs
        )
        
        return env
    
    def _get_model_class(self) -> Type:
        """Obtener clase del modelo según algoritmo"""
        mapping = {
            "PPO": PPO,
            "DQN": DQN,
            "A2C": A2C,
            "SAC": SAC,
            "TD3": TD3
        }
        return mapping[self.config.algorithm]
    
    def _create_callbacks(self) -> List[BaseCallback]:
        """Crear lista de callbacks para entrenamiento"""
        callbacks = []
        
        # Trading metrics
        callbacks.append(TradingMetricsCallback(
            log_freq=1000,
            save_freq=10000,
            save_path=os.path.join(self.config.log_path, "metrics")
        ))
        
        # Risk management
        callbacks.append(RiskManagementCallback(
            max_drawdown=self.config.max_drawdown_threshold,
            min_win_rate=self.config.min_win_rate_threshold,
            max_consecutive_losses=self.config.max_consecutive_losses,
            check_freq=5000
        ))
        
        # Adaptive learning rate
        callbacks.append(AdaptiveLearningRateCallback(
            patience=10,
            factor=0.5,
            min_lr=1e-6,
            check_freq=10000
        ))
        
        # Checkpoints
        callbacks.append(CheckpointCallback(
            save_freq=self.config.checkpoint_freq,
            save_path=self.config.checkpoint_path,
            name_prefix=f"{self.config.algorithm}_{self.config.model_name}",
            save_replay_buffer=self.config.algorithm in ["DQN", "SAC", "TD3"],
            save_vecnormalize=True
        ))
        
        # Evaluación
        if self.eval_env is not None:
            # Early stopping callback
            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=self.config.early_stop_patience,
                min_evals=5,
                verbose=1
            )
            
            eval_callback = EvalCallback(
                self.eval_env,
                best_model_save_path=os.path.join(self.config.checkpoint_path, "best_model"),
                log_path=self.config.log_path,
                eval_freq=max(self.config.eval_freq // self.config.n_envs, 1),
                n_eval_episodes=self.config.n_eval_episodes,
                deterministic=self.config.eval_deterministic,
                callback_after_eval=stop_callback,
                verbose=1
            )
            callbacks.append(eval_callback)
        
        # Reward threshold
        if self.config.reward_threshold is not None:
            callbacks.append(StopTrainingOnRewardThreshold(
                reward_threshold=self.config.reward_threshold,
                verbose=1
            ))
        
        return callbacks
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Obtener resumen del entrenamiento"""
        summary = {
            'algorithm': self.config.algorithm,
            'total_timesteps': self.config.total_timesteps,
            'training_time': self.total_training_time,
            'final_model_path': os.path.join(self.config.checkpoint_path, "final_model.zip")
        }
        
        # Agregar métricas si están disponibles
        if self.training_history:
            last_metrics = self.training_history[-1]
            summary.update({
                'final_' + k: v for k, v in last_metrics.items()
                if k not in ['timestep', 'episode']
            })
        
        return summary
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Obtener importancia de features (si está disponible)
        
        Returns:
            Array con importancias o None
        """
        # Esto requeriría técnicas como SHAP o perturbación
        # Por ahora retornamos None
        return None


# =====================================================
# IMPLEMENTACIONES ESPECÍFICAS
# =====================================================

class PPOTradingAgent(BaseRLTradingAgent):
    """Agente PPO optimizado para trading"""
    
    def create_model(self, env: gym.Env) -> PPO:
        """Crear modelo PPO"""
        # Configurar política
        policy_kwargs = {
            'net_arch': self.config.net_arch,
            'activation_fn': self._get_activation_fn()
        }
        
        # Feature extractor personalizado
        if self.config.features_extractor_class is not None:
            policy_kwargs['features_extractor_class'] = self.config.features_extractor_class
            policy_kwargs['features_extractor_kwargs'] = self.config.features_extractor_kwargs or {}
        else:
            # Usar extractor por defecto
            policy_kwargs['features_extractor_class'] = TradingFeaturesExtractor
            policy_kwargs['features_extractor_kwargs'] = {'features_dim': 256}
        
        # Crear modelo
        model = PPO(
            policy=self.config.policy,
            env=env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            clip_range_vf=self.config.clip_range_vf,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.tensorboard_log,
            device=self.config.device,
            verbose=self.config.verbose,
            seed=42
        )
        
        return model
    
    def _get_activation_fn(self):
        """Obtener función de activación"""
        mapping = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU,
            'gelu': nn.GELU
        }
        return mapping.get(self.config.activation_fn, nn.Tanh)


class DQNTradingAgent(BaseRLTradingAgent):
    """Agente DQN con Double DQN y Dueling"""
    
    def create_model(self, env: gym.Env) -> DQN:
        """Crear modelo DQN"""
        # Configurar política
        policy_kwargs = {
            'net_arch': self.config.net_arch,
            'activation_fn': self._get_activation_fn(),
            'dueling': True,  # Dueling DQN
            'dueling_type': 'avg'  # avg, max, naive
        }
        
        # Feature extractor
        if self.config.features_extractor_class is not None:
            policy_kwargs['features_extractor_class'] = self.config.features_extractor_class
            policy_kwargs['features_extractor_kwargs'] = self.config.features_extractor_kwargs or {}
        else:
            policy_kwargs['features_extractor_class'] = TradingFeaturesExtractor
            policy_kwargs['features_extractor_kwargs'] = {'features_dim': 256}
        
        # Crear modelo
        model = DQN(
            policy=self.config.policy,
            env=env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            learning_starts=self.config.learning_starts,
            batch_size=self.config.batch_size,
            tau=self.config.tau,
            gamma=self.config.gamma,
            train_freq=self.config.train_freq,
            gradient_steps=self.config.gradient_steps,
            target_update_interval=self.config.target_update_interval,
            exploration_fraction=self.config.exploration_fraction,
            exploration_initial_eps=self.config.exploration_initial_eps,
            exploration_final_eps=self.config.exploration_final_eps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.tensorboard_log,
            device=self.config.device,
            verbose=self.config.verbose,
            seed=42
        )
        
        return model
    
    def _get_activation_fn(self):
        """Obtener función de activación"""
        mapping = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU,
            'gelu': nn.GELU
        }
        return mapping.get(self.config.activation_fn, nn.ReLU)


class A2CTradingAgent(BaseRLTradingAgent):
    """Agente A2C para trading"""
    
    def create_model(self, env: gym.Env) -> A2C:
        """Crear modelo A2C"""
        # Configurar política
        policy_kwargs = {
            'net_arch': self.config.net_arch,
            'activation_fn': self._get_activation_fn()
        }
        
        # Feature extractor con LSTM
        policy_kwargs['features_extractor_class'] = LSTMTradingExtractor
        policy_kwargs['features_extractor_kwargs'] = {
            'features_dim': 256,
            'lstm_hidden_size': 128,
            'n_lstm_layers': 2
        }
        
        # Crear modelo
        model = A2C(
            policy=self.config.policy,
            env=env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ent_coef=self.config.ent_coef,
            vf_coef=self.config.vf_coef,
            max_grad_norm=self.config.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=self.config.tensorboard_log,
            device=self.config.device,
            verbose=self.config.verbose,
            seed=42
        )
        
        return model
    
    def _get_activation_fn(self):
        """Obtener función de activación"""
        mapping = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'elu': nn.ELU,
            'leaky_relu': nn.LeakyReLU,
            'gelu': nn.GELU
        }
        return mapping.get(self.config.activation_fn, nn.Tanh)


# =====================================================
# FACTORY Y UTILIDADES
# =====================================================

class TradingAgentFactory:
    """Factory para crear agentes"""
    
    _agents = {
        "PPO": PPOTradingAgent,
        "DQN": DQNTradingAgent,
        "A2C": A2CTradingAgent,
        # SAC y TD3 requieren ambientes de acción continua
    }
    
    @classmethod
    def create_agent(cls, config: Union[AgentConfig, Dict[str, Any]]) -> BaseRLTradingAgent:
        """
        Crear agente según configuración
        
        Args:
            config: Configuración del agente
            
        Returns:
            Agente inicializado
        """
        if isinstance(config, dict):
            config = AgentConfig(**config)
        
        agent_class = cls._agents.get(config.algorithm)
        if agent_class is None:
            raise ValueError(f"Unknown algorithm: {config.algorithm}. "
                           f"Available: {list(cls._agents.keys())}")
        
        return agent_class(config)
    
    @classmethod
    def register_agent(cls, name: str, agent_class: Type[BaseRLTradingAgent]):
        """Registrar nuevo tipo de agente"""
        cls._agents[name] = agent_class
    
    @classmethod
    def list_algorithms(cls) -> List[str]:
        """Listar algoritmos disponibles"""
        return list(cls._agents.keys())


# =====================================================
# UTILIDADES DE DATOS
# =====================================================

def load_trading_data(path: str, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Cargar datos de trading desde archivo o directorio
    
    Args:
        path: Ruta al archivo o directorio
        start_date: Fecha inicial (YYYY-MM-DD)
        end_date: Fecha final (YYYY-MM-DD)
        
    Returns:
        DataFrame con datos OHLCV
    """
    # Detectar tipo de archivo/directorio
    if os.path.isdir(path):
        # Cargar todos los archivos del directorio
        files = []
        for ext in ['*.parquet', '*.csv']:
            files.extend(Path(path).rglob(ext))
        
        if not files:
            raise FileNotFoundError(f"No data files found in {path}")
        
        # Cargar y concatenar
        dfs = []
        for file in sorted(files):
            try:
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                else:
                    df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error loading {file}: {e}")
        
        data = pd.concat(dfs, ignore_index=True)
    else:
        # Cargar archivo único
        if path.endswith('.parquet'):
            data = pd.read_parquet(path)
        elif path.endswith('.csv'):
            data = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    
    # Procesar columnas de tiempo
    if 'time' in data.columns:
        data['time'] = pd.to_datetime(data['time'], utc=True)
        data = data.sort_values('time').reset_index(drop=True)
    elif 'timestamp' in data.columns:
        data['time'] = pd.to_datetime(data['timestamp'], utc=True)
        data = data.sort_values('time').reset_index(drop=True)
    
    # Filtrar por fechas
    if start_date:
        data = data[data['time'] >= pd.to_datetime(start_date, utc=True)]
    if end_date:
        data = data[data['time'] <= pd.to_datetime(end_date, utc=True)]
    
    # Validar columnas requeridas
    for col in REQUIRED_COLS:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Limpiar datos
    data = data.dropna(subset=['open', 'high', 'low', 'close'])
    
    logger.info(f"Loaded {len(data)} rows from {data['time'].min()} to {data['time'].max()}")
    
    return data


def split_data(data: pd.DataFrame, 
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split temporal de datos
    
    Args:
        data: DataFrame con datos
        train_ratio: Proporción para entrenamiento
        val_ratio: Proporción para validación
        test_ratio: Proporción para prueba
        
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = data.iloc[:train_end].reset_index(drop=True)
    val_df = data.iloc[train_end:val_end].reset_index(drop=True)
    test_df = data.iloc[val_end:].reset_index(drop=True)
    
    logger.info(f"Data split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


# =====================================================
# HYPERPARAMETER TUNING
# =====================================================

class HyperparameterOptimizer:
    """Optimizador de hiperparámetros con Optuna"""
    
    def __init__(self, 
                 base_config: AgentConfig,
                 train_data: pd.DataFrame,
                 val_data: pd.DataFrame,
                 n_trials: int = 50,
                 n_jobs: int = 1,
                 study_name: Optional[str] = None):
        """
        Inicializar optimizador
        
        Args:
            base_config: Configuración base
            train_data: Datos de entrenamiento
            val_data: Datos de validación
            n_trials: Número de trials
            n_jobs: Trabajos paralelos
            study_name: Nombre del estudio
        """
        if not HAS_OPTUNA:
            raise ImportError("optuna not installed. Install with: pip install optuna")
        
        self.base_config = base_config
        self.train_data = train_data
        self.val_data = val_data
        self.n_trials = n_trials
        self.n_jobs = n_jobs
        self.study_name = study_name or f"trading_hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    def optimize(self) -> Dict[str, Any]:
        """
        Ejecutar optimización
        
        Returns:
            Mejores hiperparámetros encontrados
        """
        # Crear estudio
        study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimizar
        study.optimize(
            self._objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Resultados
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best value: {best_value}")
        
        # Guardar estudio
        study_path = f"optuna_{self.study_name}.pkl"
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'n_trials': len(study.trials),
            'study_path': study_path
        }
    
    def _objective(self, trial) -> float:
        """Función objetivo para optimización"""
        # Sugerir hiperparámetros
        config = self._suggest_hyperparameters(trial)
        
        # Reducir timesteps para optimización rápida
        config.total_timesteps = 50000
        config.eval_freq = 5000
        config.checkpoint_freq = 25000
        
        try:
            # Crear y entrenar agente
            agent = TradingAgentFactory.create_agent(config)
            agent.train(self.train_data, self.val_data)
            
            # Evaluar
            results = agent.evaluate(self.val_data, n_episodes=5)
            
            # Métrica objetivo: Sharpe ratio penalizado por drawdown
            sharpe = results.get('mean_sharpe_ratio', 0)
            max_dd = results.get('mean_max_drawdown', 0)
            
            # Penalizar drawdown alto
            if abs(max_dd) > 0.15:
                sharpe *= (1 - abs(max_dd))
            
            return sharpe
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return -1000  # Valor muy bajo para trials fallidos
    
    def _suggest_hyperparameters(self, trial) -> AgentConfig:
        """Sugerir hiperparámetros para trial"""
        # Copiar configuración base
        config_dict = self.base_config.to_dict()
        
        # Sugerir según algoritmo
        if self.base_config.algorithm == "PPO":
            config_dict.update({
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'n_steps': trial.suggest_categorical('n_steps', [128, 256, 512, 1024, 2048]),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'n_epochs': trial.suggest_int('n_epochs', 3, 20),
                'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
                'gae_lambda': trial.suggest_uniform('gae_lambda', 0.9, 1.0),
                'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
                'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-2),
                'vf_coef': trial.suggest_uniform('vf_coef', 0.1, 1.0),
            })
            
        elif self.base_config.algorithm == "DQN":
            config_dict.update({
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                'buffer_size': trial.suggest_int('buffer_size', 10000, 200000),
                'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
                'target_update_interval': trial.suggest_int('target_update_interval', 100, 10000),
                'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
                'exploration_fraction': trial.suggest_uniform('exploration_fraction', 0.05, 0.3),
                'exploration_final_eps': trial.suggest_uniform('exploration_final_eps', 0.01, 0.1),
            })
        
        # Arquitectura de red
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layer_sizes = []
        for i in range(n_layers):
            layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 64, 512))
        
        config_dict['net_arch'] = layer_sizes
        
        return AgentConfig(**config_dict)


# =====================================================
# ENSEMBLE DE AGENTES
# =====================================================

class EnsembleTradingAgent:
    """Ensemble de múltiples agentes para trading robusto"""
    
    def __init__(self, 
                 agents: List[BaseRLTradingAgent],
                 weights: Optional[List[float]] = None,
                 voting: str = "soft"):
        """
        Inicializar ensemble
        
        Args:
            agents: Lista de agentes entrenados
            weights: Pesos para cada agente
            voting: Tipo de votación ('soft' o 'hard')
        """
        if not agents:
            raise ValueError("At least one agent required")
        
        self.agents = agents
        self.weights = weights or [1.0] * len(agents)
        self.voting = voting
        
        # Normalizar pesos
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        logger.info(f"Created ensemble with {len(agents)} agents")
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, float]:
        """
        Predecir usando ensemble
        
        Returns:
            action: Acción predicha
            confidence: Confianza en la predicción
        """
        if self.voting == "hard":
            # Votación dura
            votes = []
            for agent, weight in zip(self.agents, self.weights):
                action, _ = agent.predict(observation, deterministic=deterministic)
                # Replicar voto según peso
                votes.extend([action] * int(weight * 100))
            
            # Acción más votada
            from collections import Counter
            vote_counts = Counter(votes)
            action = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[action] / len(votes)
            
        else:  # soft voting
            # Para soft voting necesitaríamos acceso a las probabilidades
            # Por ahora usamos votación ponderada simple
            action_scores = {}
            
            for agent, weight in zip(self.agents, self.weights):
                action, _ = agent.predict(observation, deterministic=deterministic)
                action_scores[action] = action_scores.get(action, 0) + weight
            
            # Acción con mayor score
            action = max(action_scores.items(), key=lambda x: x[1])[0]
            confidence = action_scores[action]
        
        return action, confidence
    
    def evaluate(self, test_data: pd.DataFrame, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluar ensemble"""
        # Crear environment
        env = self.agents[0]._create_eval_env(test_data)
        
        all_rewards = []
        all_lengths = []
        all_metrics = []
        
        for episode in range(n_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, confidence = self.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            all_rewards.append(episode_reward)
            all_lengths.append(episode_length)
            
            if hasattr(env, 'get_trading_statistics'):
                all_metrics.append(env.get_trading_statistics())
        
        # Resultados
        results = {
            'ensemble_size': len(self.agents),
            'voting': self.voting,
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_length': np.mean(all_lengths)
        }
        
        # Métricas de trading
        if all_metrics:
            for key in all_metrics[0].keys():
                values = [m.get(key, 0) for m in all_metrics]
                results[f'mean_{key}'] = np.mean(values)
        
        return results


# =====================================================
# CLI Y FUNCIONES PRINCIPALES
# =====================================================

def train_agent_cli():
    """CLI para entrenar agente"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train USDCOP RL Trading Agent")
    
    # Datos
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data (file or directory)')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    
    # Algoritmo
    parser.add_argument('--algo', type=str, default='PPO',
                       choices=SUPPORTED_ALGORITHMS,
                       help='RL algorithm to use')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                       help='Total training timesteps')
    
    # Modelo
    parser.add_argument('--model-name', type=str, default='usdcop_agent',
                       help='Model name for saving')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    # Hiperparámetros
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--n-envs', type=int, default=4,
                       help='Number of parallel environments')
    
    # Logging
    parser.add_argument('--tb-log', type=str, default='./tensorboard',
                       help='Tensorboard log directory')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level')
    
    args = parser.parse_args()
    
    # Cargar datos
    logger.info(f"Loading data from {args.data}")
    data = load_trading_data(args.data, args.start_date, args.end_date)
    
    # Split datos
    train_data, val_data, test_data = split_data(data)
    
    # Configuración
    config = AgentConfig(
        algorithm=args.algo,
        model_name=args.model_name,
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        n_envs=args.n_envs,
        tensorboard_log=args.tb_log,
        verbose=args.verbose
    )
    
    # Crear y entrenar agente
    agent = TradingAgentFactory.create_agent(config)
    results = agent.train(train_data, val_data, resume_from=args.resume)
    
    # Evaluar en test
    logger.info("Evaluating on test data...")
    test_results = agent.evaluate(test_data, n_episodes=10)
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("TRAINING COMPLETED")
    print("="*60)
    print(f"Algorithm: {config.algorithm}")
    print(f"Training time: {results.get('training_time', 0):.2f}s")
    print(f"Final Sharpe ratio: {test_results.get('mean_sharpe_ratio', 0):.3f}")
    print(f"Final Win rate: {test_results.get('mean_win_rate', 0):.2%}")
    print(f"Final Max drawdown: {test_results.get('mean_max_drawdown', 0):.2%}")
    print("="*60)


def evaluate_agent_cli():
    """CLI para evaluar agente"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate USDCOP RL Trading Agent")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model (without extension)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--n-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    
    args = parser.parse_args()
    
    # Cargar datos
    data = load_trading_data(args.data)
    
    # Cargar agente
    # Primero necesitamos cargar la configuración para saber el algoritmo
    with open(f"{args.model}_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    config = AgentConfig(**metadata['config'])
    agent = TradingAgentFactory.create_agent(config)
    agent.load(args.model)
    
    # Evaluar
    results = agent.evaluate(data, n_episodes=args.n_episodes, render=args.render)
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("="*60)


def backtest_agent_cli():
    """CLI para backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Backtest USDCOP RL Trading Agent")
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to backtest data')
    parser.add_argument('--initial-balance', type=float, default=10000,
                       help='Initial balance')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Cargar datos
    data = load_trading_data(args.data)
    
    # Cargar agente
    with open(f"{args.model}_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    config = AgentConfig(**metadata['config'])
    agent = TradingAgentFactory.create_agent(config)
    agent.load(args.model)
    
    # Backtest
    results = agent.backtest(data, initial_balance=args.initial_balance)
    
    # Guardar resultados
    if args.output:
        # Convertir arrays a listas para JSON
        json_results = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                json_results[k] = v.tolist()
            elif isinstance(v, pd.DataFrame):
                json_results[k] = v.to_dict()
            else:
                json_results[k] = v
        
        with open(args.output, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Initial balance: ${args.initial_balance:,.2f}")
    print(f"Final balance: ${results.get('final_balance', 0):,.2f}")
    print(f"Total return: {results.get('total_return', 0):.2%}")
    print(f"Number of trades: {results.get('n_trades', 0)}")
    print(f"Win rate: {results.get('win_rate', 0):.2%}")
    print(f"Sharpe ratio: {results.get('sharpe_ratio', 0):.3f}")
    print(f"Max drawdown: {results.get('max_drawdown', 0):.2%}")
    print("="*60)


def optimize_hyperparameters_cli():
    """CLI para optimización de hiperparámetros"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize hyperparameters")
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--algo', type=str, default='PPO',
                       choices=SUPPORTED_ALGORITHMS,
                       help='Algorithm to optimize')
    parser.add_argument('--n-trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Cargar datos
    data = load_trading_data(args.data)
    train_data, val_data, _ = split_data(data)
    
    # Configuración base
    base_config = AgentConfig(algorithm=args.algo)
    
    # Optimizar
    optimizer = HyperparameterOptimizer(
        base_config=base_config,
        train_data=train_data,
        val_data=val_data,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs
    )
    
    results = optimizer.optimize()
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"\nBest value: {results['best_value']:.4f}")
    print(f"Total trials: {results['n_trials']}")
    print("="*60)


# =====================================================
# EJEMPLO DE USO Y TESTING
# =====================================================

def example_usage():
    """Ejemplo completo de uso del sistema"""
    logger.info("=== USDCOP RL Trading Agent Example ===")
    
    # Generar datos sintéticos para ejemplo
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='5min')
    n = len(dates)
    
    # Simular precio con tendencia y ruido
    trend = np.linspace(4000, 4200, n)
    seasonal = 50 * np.sin(np.linspace(0, 4*np.pi, n))
    noise = np.random.randn(n) * 10
    
    price = trend + seasonal + noise
    
    # Crear OHLCV
    data = pd.DataFrame({
        'time': dates,
        'open': price + np.random.randn(n) * 2,
        'high': price + np.abs(np.random.randn(n) * 5),
        'low': price - np.abs(np.random.randn(n) * 5),
        'close': price + np.random.randn(n) * 2,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # Ajustar OHLC
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    # Split datos
    train_data, val_data, test_data = split_data(data)
    
    # === 1. Entrenar agente PPO ===
    logger.info("\n1. Training PPO agent...")
    
    ppo_config = AgentConfig(
        algorithm="PPO",
        total_timesteps=50000,  # Reducido para ejemplo
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_envs=2
    )
    
    ppo_agent = PPOTradingAgent(ppo_config)
    ppo_results = ppo_agent.train(train_data, val_data)
    
    # === 2. Evaluar agente ===
    logger.info("\n2. Evaluating agent...")
    
    eval_results = ppo_agent.evaluate(test_data, n_episodes=5)
    logger.info(f"Evaluation results: {eval_results}")
    
    # === 3. Backtest ===
    logger.info("\n3. Running backtest...")
    
    backtest_results = ppo_agent.backtest(test_data, initial_balance=10000)
    
    logger.info(f"Final balance: ${backtest_results.get('final_balance', 0):,.2f}")
    logger.info(f"Total return: {backtest_results.get('total_return', 0):.2%}")
    logger.info(f"Sharpe ratio: {backtest_results.get('sharpe_ratio', 0):.3f}")
    
    # === 4. Comparar algoritmos ===
    logger.info("\n4. Comparing algorithms...")
    
    algorithms_to_test = ["PPO", "DQN", "A2C"]
    comparison_results = []
    
    for algo in algorithms_to_test:
        logger.info(f"Training {algo}...")
        
        config = AgentConfig(
            algorithm=algo,
            total_timesteps=20000,  # Muy reducido para ejemplo
            n_envs=1  # DQN no soporta multi-env
        )
        
        agent = TradingAgentFactory.create_agent(config)
        agent.train(train_data[:1000], val_data[:500])  # Datos reducidos
        
        results = agent.evaluate(test_data[:500], n_episodes=3)
        results['algorithm'] = algo
        comparison_results.append(results)
    
    # Mostrar comparación
    comparison_df = pd.DataFrame(comparison_results)
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON")
    print("="*60)
    print(comparison_df.to_string())
    print("="*60)
    
    logger.info("\n✅ Example completed successfully!")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    import sys
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Detectar comando
    if len(sys.argv) > 1:
        command = sys.argv[1]
        sys.argv = sys.argv[1:]  # Shift para argparse
        
        if command == "train":
            train_agent_cli()
        elif command == "evaluate":
            evaluate_agent_cli()
        elif command == "backtest":
            backtest_agent_cli()
        elif command == "optimize":
            optimize_hyperparameters_cli()
        elif command == "example":
            example_usage()
        else:
            print(f"Unknown command: {command}")
            print("Available commands: train, evaluate, backtest, optimize, example")
    else:
        # Sin argumentos, ejecutar ejemplo
        example_usage()