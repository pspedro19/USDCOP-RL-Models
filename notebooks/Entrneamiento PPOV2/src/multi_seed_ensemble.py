"""
Multi-Seed Ensemble Training
============================

Entrena múltiples modelos con diferentes seeds y los combina.
Reduce varianza sin la complejidad de modelos con diferentes hiperparámetros.

PROBLEMA QUE RESUELVE:
- Un solo modelo puede tener alta varianza dependiendo de la seed
- Ensemble de 2 modelos con hiperparámetros diferentes tenía correlación negativa

SOLUCIÓN:
- Entrenar N modelos con MISMA arquitectura pero DIFERENTE seed
- Promediar acciones reduce varianza
- Más simple y robusto que ensemble heterogéneo

BENEFICIO vs Ensemble Tradicional:
- Misma arquitectura, mismos hiperparámetros
- Solo cambia la inicialización random
- Reduce varianza sin añadir complejidad
- No hay riesgo de correlación negativa entre modelos

Author: Claude Code
Version: 1.0.0
"""

import numpy as np
from typing import List, Optional, Tuple, Callable, Dict, Any
from pathlib import Path
import json
from collections import Counter

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable_baselines3 not installed. MultiSeedEnsemble will be limited.")


class MultiSeedEnsemble:
    """
    Ensemble de modelos entrenados con diferentes seeds.

    Beneficio vs ensemble tradicional:
    - Misma arquitectura, mismos hiperparámetros
    - Solo cambia la inicialización random
    - Reduce varianza sin añadir complejidad

    Args:
        n_seeds: Número de seeds/modelos a entrenar
        base_seed: Seed inicial (los demás se generan como base + i*1000)
        aggregation: Método de agregación ('mean', 'median', 'vote')
    """

    def __init__(
        self,
        n_seeds: int = 5,
        base_seed: int = 42,
        aggregation: str = 'mean',
    ):
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.aggregation = aggregation
        self.models: List = []
        self.seeds = [base_seed + i * 1000 for i in range(n_seeds)]

        # Métricas de training
        self.training_metrics: Dict[int, Dict] = {}

    def train(
        self,
        env_factory: Callable,
        model_params: dict,
        total_timesteps: int,
        callbacks: Optional[List] = None,
        verbose: int = 1,
    ) -> 'MultiSeedEnsemble':
        """
        Entrenar todos los modelos.

        Args:
            env_factory: Función que retorna un environment nuevo
            model_params: Parámetros para PPO
            total_timesteps: Timesteps por modelo
            callbacks: Callbacks de SB3 (opcional)
            verbose: Nivel de verbosidad

        Returns:
            self para chaining
        """
        if not HAS_SB3:
            raise RuntimeError("stable_baselines3 required for training")

        self.models = []

        for i, seed in enumerate(self.seeds):
            if verbose > 0:
                print(f"\n{'=' * 60}")
                print(f"Training model {i + 1}/{self.n_seeds} with seed {seed}")
                print('=' * 60)

            # Crear environment con seed específico
            env = env_factory()

            # Crear modelo con seed
            model = PPO(
                "MlpPolicy",
                env,
                seed=seed,
                **model_params,
                verbose=verbose,
            )

            # Entrenar
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
            )

            self.models.append(model)

            # Guardar métricas si el env las tiene
            if hasattr(env, 'get_episode_metrics'):
                self.training_metrics[seed] = env.get_episode_metrics()

            if verbose > 0:
                print(f"Model {i + 1} training completed")

        return self

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, None]:
        """
        Predecir usando ensemble.

        Args:
            obs: Observación (puede ser batch)
            deterministic: Si usar predicción determinística

        Returns:
            Acción agregada, None (para compatibilidad con SB3)
        """
        if not self.models:
            raise ValueError("No models trained yet. Call train() first.")

        # Obtener predicción de cada modelo
        all_actions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=deterministic)
            all_actions.append(action)

        all_actions = np.array(all_actions)

        # Agregar según método
        if self.aggregation == 'mean':
            final_action = np.mean(all_actions, axis=0)
        elif self.aggregation == 'median':
            final_action = np.median(all_actions, axis=0)
        elif self.aggregation == 'vote':
            final_action = self._weighted_vote(all_actions)
        else:
            final_action = np.mean(all_actions, axis=0)

        return final_action, None

    def _weighted_vote(
        self,
        all_actions: np.ndarray,
        action_threshold: float = 0.10,
    ) -> np.ndarray:
        """
        Voting por dirección con promedio de magnitud.

        Cada modelo "vota" por LONG, SHORT o HOLD.
        La dirección ganadora se usa con la magnitud promedio de los votantes.
        """
        votes = {'long': 0, 'short': 0, 'hold': 0}
        values = {'long': [], 'short': [], 'hold': []}

        for action in all_actions:
            a = float(action[0]) if hasattr(action, '__len__') else float(action)

            if a > action_threshold:
                votes['long'] += 1
                values['long'].append(a)
            elif a < -action_threshold:
                votes['short'] += 1
                values['short'].append(a)
            else:
                votes['hold'] += 1
                values['hold'].append(a)

        # Ganador
        winner = max(votes, key=votes.get)

        # Promedio del ganador
        if values[winner]:
            final = np.mean(values[winner])
        else:
            final = 0.0

        return np.array([final])

    def get_agreement_score(self, obs: np.ndarray) -> float:
        """
        Medir acuerdo entre modelos (0-1).

        Útil para detectar incertidumbre. Si los modelos no están de acuerdo,
        el sistema puede ser más conservador.

        Args:
            obs: Observación

        Returns:
            Score de acuerdo (1 = unanimidad, 0 = desacuerdo total)
        """
        if not self.models:
            return 0.5

        action_threshold = 0.10
        directions = []

        for model in self.models:
            action, _ = model.predict(obs, deterministic=True)
            a = float(action[0]) if hasattr(action, '__len__') else float(action)

            if a > action_threshold:
                directions.append(1)  # LONG
            elif a < -action_threshold:
                directions.append(-1)  # SHORT
            else:
                directions.append(0)  # HOLD

        # Contar mayoría
        counts = Counter(directions)
        most_common_count = counts.most_common(1)[0][1]

        return most_common_count / len(directions)

    def get_uncertainty(self, obs: np.ndarray) -> float:
        """
        Medir incertidumbre como desviación estándar de predicciones.

        Mayor std = mayor incertidumbre = considerar ser más conservador.

        Args:
            obs: Observación

        Returns:
            Desviación estándar de las acciones predichas
        """
        if not self.models:
            return 1.0

        all_actions = []
        for model in self.models:
            action, _ = model.predict(obs, deterministic=True)
            all_actions.append(float(action[0]))

        return float(np.std(all_actions))

    def get_action_with_confidence(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        uncertainty_threshold: float = 0.3,
        agreement_threshold: float = 0.6,
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Obtener acción con información de confianza.

        Útil para aplicar sizing dinámico basado en confianza del ensemble.

        Args:
            obs: Observación
            deterministic: Si usar predicción determinística
            uncertainty_threshold: Umbral de incertidumbre para reducir posición
            agreement_threshold: Umbral de acuerdo para mantener posición completa

        Returns:
            (action, confidence_score, should_reduce)
            - action: Acción del ensemble
            - confidence_score: Score de confianza (0-1)
            - should_reduce: Si debería reducir posición
        """
        action, _ = self.predict(obs, deterministic=deterministic)
        uncertainty = self.get_uncertainty(obs)
        agreement = self.get_agreement_score(obs)

        # Confidence score (0-1)
        # High agreement + low uncertainty = high confidence
        confidence = agreement * (1 - min(uncertainty / 0.5, 1.0))

        # Should reduce if low agreement OR high uncertainty
        should_reduce = (agreement < agreement_threshold or
                        uncertainty > uncertainty_threshold)

        return action, confidence, should_reduce

    def save(self, path: str) -> None:
        """
        Guardar ensemble completo.

        Args:
            path: Directorio donde guardar
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Guardar cada modelo
        for i, (model, seed) in enumerate(zip(self.models, self.seeds)):
            model.save(str(save_dir / f"model_seed_{seed}"))

        # Guardar configuración
        config = {
            'n_seeds': self.n_seeds,
            'base_seed': self.base_seed,
            'seeds': self.seeds,
            'aggregation': self.aggregation,
            'training_metrics': {
                str(k): v for k, v in self.training_metrics.items()
            },
        }

        with open(save_dir / 'ensemble_config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)

        print(f"Ensemble saved to {save_dir}")

    @classmethod
    def load(cls, path: str, env=None) -> 'MultiSeedEnsemble':
        """
        Cargar ensemble guardado.

        Args:
            path: Directorio donde está guardado
            env: Environment para cargar modelos (opcional)

        Returns:
            MultiSeedEnsemble cargado
        """
        if not HAS_SB3:
            raise RuntimeError("stable_baselines3 required for loading")

        load_dir = Path(path)

        with open(load_dir / 'ensemble_config.json', 'r') as f:
            config = json.load(f)

        ensemble = cls(
            n_seeds=config['n_seeds'],
            base_seed=config['base_seed'],
            aggregation=config['aggregation'],
        )
        ensemble.seeds = config['seeds']

        # Cargar métricas
        if 'training_metrics' in config:
            ensemble.training_metrics = {
                int(k): v for k, v in config['training_metrics'].items()
            }

        # Cargar modelos
        ensemble.models = []
        for seed in ensemble.seeds:
            model_path = str(load_dir / f"model_seed_{seed}")
            model = PPO.load(model_path, env=env)
            ensemble.models.append(model)

        print(f"Loaded ensemble with {len(ensemble.models)} models from {load_dir}")

        return ensemble

    def evaluate(
        self,
        env,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluar ensemble en un environment.

        Args:
            env: Environment de evaluación
            n_episodes: Número de episodios
            deterministic: Si usar predicción determinística

        Returns:
            Diccionario con métricas de evaluación
        """
        all_returns = []
        all_sharpes = []
        all_max_dd = []
        all_trades = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0.0

            while not done:
                action, confidence, should_reduce = self.get_action_with_confidence(
                    obs, deterministic=deterministic
                )

                # Aplicar reducción si hay incertidumbre
                if should_reduce:
                    action = action * 0.5

                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += info.get('step_return', 0)
                done = terminated or truncated

            # Obtener métricas del episodio
            if hasattr(env, 'get_episode_metrics'):
                metrics = env.get_episode_metrics()
                all_sharpes.append(metrics.get('sharpe', 0))
                all_max_dd.append(metrics.get('max_drawdown', 0))
                all_trades.append(metrics.get('trade_count', 0))

            all_returns.append(episode_return)

        return {
            'mean_return': float(np.mean(all_returns)),
            'std_return': float(np.std(all_returns)),
            'mean_sharpe': float(np.mean(all_sharpes)) if all_sharpes else 0.0,
            'std_sharpe': float(np.std(all_sharpes)) if all_sharpes else 0.0,
            'mean_max_dd': float(np.mean(all_max_dd)) if all_max_dd else 0.0,
            'mean_trades': float(np.mean(all_trades)) if all_trades else 0.0,
            'n_episodes': n_episodes,
        }


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def train_multi_seed_ensemble(
    df,
    feature_columns: List[str],
    n_seeds: int = 5,
    timesteps_per_model: int = 100_000,
    model_params: Optional[Dict] = None,
    output_dir: str = './models/multi_seed',
    verbose: int = 1,
) -> MultiSeedEnsemble:
    """
    Función de conveniencia para entrenar un MultiSeedEnsemble.

    Args:
        df: DataFrame con datos
        feature_columns: Columnas de features
        n_seeds: Número de seeds
        timesteps_per_model: Timesteps por modelo
        model_params: Parámetros de PPO (opcional)
        output_dir: Directorio de salida
        verbose: Verbosidad

    Returns:
        MultiSeedEnsemble entrenado
    """
    # Import environment
    try:
        from .environment_v19 import TradingEnvironmentV19
    except ImportError:
        from environment_v19 import TradingEnvironmentV19

    def env_factory():
        return TradingEnvironmentV19(
            df=df,
            feature_columns=feature_columns,
            use_vol_scaling=True,
            use_regime_detection=True,
        )

    # Default model params (Model B - production)
    default_params = {
        'learning_rate': 1e-4,
        'n_steps': 2048,
        'batch_size': 128,
        'n_epochs': 10,
        'gamma': 0.99,
        'ent_coef': 0.05,  # Model B entropy
        'clip_range': 0.2,
        'gae_lambda': 0.95,
        'policy_kwargs': {'net_arch': [256, 256]},
    }

    if model_params:
        default_params.update(model_params)

    # Entrenar ensemble
    ensemble = MultiSeedEnsemble(n_seeds=n_seeds, aggregation='mean')
    ensemble.train(
        env_factory=env_factory,
        model_params=default_params,
        total_timesteps=timesteps_per_model,
        verbose=verbose,
    )

    # Guardar
    ensemble.save(output_dir)

    print(f"\nEnsemble saved to {output_dir}")
    print(f"Seeds used: {ensemble.seeds}")

    return ensemble


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("MULTI-SEED ENSEMBLE - USD/COP RL Trading System")
    print("=" * 70)

    print("\nMultiSeedEnsemble class ready for use.")
    print("\nUsage example:")
    print("""
    from multi_seed_ensemble import MultiSeedEnsemble, train_multi_seed_ensemble

    # Option 1: Train from scratch
    ensemble = train_multi_seed_ensemble(
        df=your_dataframe,
        feature_columns=your_features,
        n_seeds=5,
        timesteps_per_model=100_000,
        output_dir='./models/ensemble',
    )

    # Option 2: Load existing
    ensemble = MultiSeedEnsemble.load('./models/ensemble')

    # Predict with confidence
    action, confidence, should_reduce = ensemble.get_action_with_confidence(obs)

    if should_reduce:
        action *= 0.5  # Be more conservative when uncertain
    """)
    print("=" * 70)
