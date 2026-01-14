"""
USD/COP RL Trading System - Ensemble Predictor
================================================

Combina multiples modelos PPO con weighted voting para reducir
varianza manteniendo performance.

ESTRATEGIA (recomendada por 5 agentes):
- Model A (70%): ent_coef=0.10, conservador, baja varianza
- Model B (30%): ent_coef=0.05, agresivo, alta performance

BENEFICIOS:
- Reduce varianza entre folds
- Combina estabilidad (A) con alpha (B)
- Expected Sharpe ~1.95, variance ~1.8

Author: Claude Code
Version: 1.0.0
Date: 2025-12-25
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

from stable_baselines3 import PPO


@dataclass
class EnsembleConfig:
    """Configuration for ensemble members."""
    name: str
    weight: float
    model_params: Dict[str, Any]
    env_config: Dict[str, Any]


class EnsemblePredictor:
    """
    Weighted ensemble of PPO models.

    Combina predicciones de multiples modelos usando weighted averaging.

    Args:
        models: Lista de modelos PPO entrenados
        weights: Pesos para cada modelo (deben sumar 1.0)
        action_mode: 'weighted_mean' o 'weighted_vote'
    """

    def __init__(
        self,
        models: Optional[List[PPO]] = None,
        weights: Optional[List[float]] = None,
        action_mode: str = 'weighted_mean',
    ):
        self.models: List[PPO] = models or []
        self.weights: List[float] = weights or []
        self.action_mode = action_mode

        if self.weights and abs(sum(self.weights) - 1.0) > 1e-6:
            # Normalize weights
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]

    def add_model(self, model: PPO, weight: float, name: str = None):
        """
        Agregar modelo al ensemble.

        Args:
            model: Modelo PPO entrenado
            weight: Peso del modelo (se normalizara automaticamente)
            name: Nombre opcional para identificar el modelo
        """
        self.models.append(model)
        self.weights.append(weight)

        # Renormalize
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Predecir accion usando ensemble.

        Args:
            obs: Observacion del environment
            deterministic: Si usar prediccion deterministica

        Returns:
            action: Accion combinada
            state: None (para compatibilidad con SB3)
        """
        if not self.models:
            raise ValueError("No models in ensemble")

        if self.action_mode == 'weighted_mean':
            return self._predict_weighted_mean(obs, deterministic)
        elif self.action_mode == 'weighted_vote':
            return self._predict_weighted_vote(obs, deterministic)
        else:
            raise ValueError(f"Unknown action_mode: {self.action_mode}")

    def _predict_weighted_mean(
        self,
        obs: np.ndarray,
        deterministic: bool,
    ) -> Tuple[np.ndarray, None]:
        """Weighted mean of continuous actions."""
        all_actions = []

        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            all_actions.append(action)

        # Weighted average
        combined = np.zeros_like(all_actions[0])
        for action, weight in zip(all_actions, self.weights):
            combined += weight * action

        return combined, None

    def _predict_weighted_vote(
        self,
        obs: np.ndarray,
        deterministic: bool,
    ) -> Tuple[np.ndarray, None]:
        """Weighted voting based on action direction."""
        ACTION_THRESHOLD = 0.10

        all_actions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            all_actions.append(action)

        # Classify each action
        votes = {'long': 0.0, 'short': 0.0, 'hold': 0.0}
        action_values = {'long': [], 'short': [], 'hold': []}

        for action, weight in zip(all_actions, self.weights):
            a = float(action[0]) if hasattr(action, '__len__') else float(action)

            if a > ACTION_THRESHOLD:
                votes['long'] += weight
                action_values['long'].append(a)
            elif a < -ACTION_THRESHOLD:
                votes['short'] += weight
                action_values['short'].append(a)
            else:
                votes['hold'] += weight
                action_values['hold'].append(a)

        # Winner
        winner = max(votes, key=votes.get)

        # Use average of winning category
        if action_values[winner]:
            final_action = np.mean(action_values[winner])
        else:
            final_action = 0.0

        return np.array([final_action]), None

    def save(self, path: str):
        """
        Guardar ensemble a disco.

        Args:
            path: Directorio donde guardar los modelos
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save each model
        for i, model in enumerate(self.models):
            model.save(str(save_dir / f"model_{i}"))

        # Save weights and config
        import json
        config = {
            'weights': self.weights,
            'action_mode': self.action_mode,
            'n_models': len(self.models),
        }
        with open(save_dir / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str, env=None) -> 'EnsemblePredictor':
        """
        Cargar ensemble desde disco.

        Args:
            path: Directorio con los modelos guardados
            env: Environment para cargar modelos (opcional)

        Returns:
            EnsemblePredictor cargado
        """
        import json

        load_dir = Path(path)

        # Load config
        with open(load_dir / 'ensemble_config.json', 'r') as f:
            config = json.load(f)

        # Load models
        models = []
        for i in range(config['n_models']):
            model_path = load_dir / f"model_{i}"
            model = PPO.load(str(model_path), env=env)
            models.append(model)

        return cls(
            models=models,
            weights=config['weights'],
            action_mode=config['action_mode'],
        )

    def get_agreement_score(self, obs: np.ndarray) -> float:
        """
        Calcular score de acuerdo entre modelos.

        Util para detectar incertidumbre.

        Returns:
            Score 0-1 donde 1 = todos los modelos de acuerdo
        """
        ACTION_THRESHOLD = 0.10

        directions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=True)
            a = float(action[0]) if hasattr(action, '__len__') else float(action)

            if a > ACTION_THRESHOLD:
                directions.append(1)
            elif a < -ACTION_THRESHOLD:
                directions.append(-1)
            else:
                directions.append(0)

        # Agreement = todos iguales = 1.0, todos diferentes = 0.0
        if len(set(directions)) == 1:
            return 1.0
        elif len(set(directions)) == len(directions):
            return 0.0
        else:
            # Partial agreement
            from collections import Counter
            counts = Counter(directions)
            most_common = counts.most_common(1)[0][1]
            return most_common / len(directions)


# =============================================================================
# ENSEMBLE CONFIGURATIONS
# =============================================================================

# Model A: Conservative (High stability, lower returns)
MODEL_A_CONFIG = EnsembleConfig(
    name="A_conservative",
    weight=0.70,  # 70% weight
    model_params={
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 4,
        "gamma": 0.95,
        "ent_coef": 0.10,  # Higher entropy = more exploration = stable
        "clip_range": 0.1,
        "policy_kwargs": {"net_arch": [48, 32]},
    },
    env_config={
        "use_vol_scaling": True,
        "use_regime_detection": True,
        "protection_mode": "min",
    },
)

# Model B: Aggressive (Higher returns, more variance)
MODEL_B_CONFIG = EnsembleConfig(
    name="B_aggressive",
    weight=0.30,  # 30% weight
    model_params={
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 4,
        "gamma": 0.95,
        "ent_coef": 0.05,  # Lower entropy = more exploitation = higher returns
        "clip_range": 0.1,
        "policy_kwargs": {"net_arch": [48, 32]},
    },
    env_config={
        "use_vol_scaling": True,
        "use_regime_detection": True,
        "protection_mode": "min",
    },
)

DEFAULT_ENSEMBLE_CONFIGS = [MODEL_A_CONFIG, MODEL_B_CONFIG]
