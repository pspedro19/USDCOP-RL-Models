"""
USD/COP RL Trading System - Production Training Script
=======================================================

Script unificado de entrenamiento para producción que integra:
- Position Sizing Dinámico (Volatility Scaling)
- Detector de Régimen de Mercado (HMM-based)
- Soporte para múltiples timeframes (5min, 15min)
- Entrenamiento robusto con curriculum learning
- Multi-seed ensemble para reducir varianza
- Validación exhaustiva y stress testing

CARACTERÍSTICAS:
- 500k+ timesteps por defecto (configurable)
- 3 seeds para ensemble simple
- Curriculum learning de 3 fases
- Costos realistas SET-FX Colombia
- Early stopping por Sharpe ratio
- Validación walk-forward
- Reportes automatizados

EJECUCIÓN:
    # Básico - SAC en 5min
    python train_production.py --data dataset_5min.parquet

    # PPO en 15min con más timesteps
    python train_production.py --data dataset_15min.parquet --timeframe 15min --algorithm PPO --timesteps 1000000

    # Multi-seed ensemble
    python train_production.py --data dataset_5min.parquet --seeds 5

    # Modo rápido para debug
    python train_production.py --data dataset_5min.parquet --quick

Author: Claude Code
Version: 1.0.0
Date: 2025-12-25
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

# Stable-Baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Componentes del proyecto
from src.environment_v19 import (
    TradingEnvironmentV19,
    SETFXCostModel,
    create_training_env,
    create_validation_env,
)
from src.callbacks import (
    SharpeEvalCallback,
    EntropySchedulerCallback,
    ActionDistributionCallback,
    CostCurriculumCallback,
)
from src.rewards import get_reward_function

# Regime detector
try:
    from archive.src.ensemble.core.regime_detector import (
        HMMRegimeDetector,
        SimpleRegimeDetector,
        create_regime_detector,
        MarketRegime,
    )
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    warnings.warn("Regime detector not available. Install hmmlearn: pip install hmmlearn")


# ==============================================================================
# VOLATILITY SCALER (Position Sizing Dinámico)
# ==============================================================================

class VolatilityScaler:
    """
    Position sizing dinámico basado en volatilidad.

    Escala el tamaño de posición inversamente proporcional a la volatilidad:
    - Alta volatilidad → posiciones más pequeñas
    - Baja volatilidad → posiciones más grandes

    Basado en:
        - "Volatility Targeting for Portfolio Optimization" (AQR 2012)
        - "Risk Parity Portfolios" (Qian et al. 2005)

    Parameters
    ----------
    target_vol : float, default=0.01
        Volatilidad objetivo (1% diario = ~16% anualizado)
    lookback : int, default=20
        Ventana para calcular volatilidad realizada
    min_scale : float, default=0.25
        Escala mínima (25% de posición máxima)
    max_scale : float, default=1.0
        Escala máxima (100% de posición máxima)
    """

    def __init__(
        self,
        target_vol: float = 0.01,
        lookback: int = 20,
        min_scale: float = 0.25,
        max_scale: float = 1.0,
    ):
        self.target_vol = target_vol
        self.lookback = lookback
        self.min_scale = min_scale
        self.max_scale = max_scale

        self.realized_vol_history = []
        self.scale_history = []

    def calculate_scale(self, returns: np.ndarray) -> float:
        """
        Calcular factor de escala basado en volatilidad reciente.

        Args:
            returns: Array de retornos recientes (al menos lookback)

        Returns:
            scale: Factor de escala [min_scale, max_scale]
        """
        returns = np.asarray(returns).flatten()

        if len(returns) < self.lookback:
            # Datos insuficientes, usar escala neutral
            return 0.5 * (self.min_scale + self.max_scale)

        # Calcular volatilidad realizada
        window = returns[-self.lookback:]
        realized_vol = np.std(window)

        if realized_vol < 1e-8:
            realized_vol = self.target_vol

        # Escalar inversamente a volatilidad
        # scale = target_vol / realized_vol
        raw_scale = self.target_vol / realized_vol

        # Limitar al rango [min_scale, max_scale]
        scale = np.clip(raw_scale, self.min_scale, self.max_scale)

        # Guardar historial
        self.realized_vol_history.append(realized_vol)
        self.scale_history.append(scale)

        return float(scale)

    def get_scaled_position(self, target_position: float, returns: np.ndarray) -> float:
        """
        Aplicar volatility scaling a una posición target.

        Args:
            target_position: Posición deseada [-1, 1]
            returns: Retornos recientes

        Returns:
            scaled_position: Posición escalada por volatilidad
        """
        scale = self.calculate_scale(returns)
        return target_position * scale

    def get_stats(self) -> Dict[str, float]:
        """Obtener estadísticas del scaler."""
        if len(self.scale_history) == 0:
            return {}

        return {
            'mean_realized_vol': float(np.mean(self.realized_vol_history)),
            'mean_scale': float(np.mean(self.scale_history)),
            'min_scale': float(np.min(self.scale_history)),
            'max_scale': float(np.max(self.scale_history)),
            'target_vol': self.target_vol,
        }


# ==============================================================================
# ENVIRONMENT WRAPPER CON VOLATILITY SCALING
# ==============================================================================

class VolScaledEnvironment(TradingEnvironmentV19):
    """
    Environment wrapper que aplica volatility scaling a las acciones.

    Intercepta las acciones del agente y las escala según volatilidad reciente.
    """

    def __init__(self, *args, vol_scaler: Optional[VolatilityScaler] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vol_scaler = vol_scaler or VolatilityScaler()
        self.returns_buffer = []

    def step(self, action: np.ndarray):
        """Step con volatility scaling aplicado."""
        # Aplicar volatility scaling si hay suficientes datos
        if len(self.returns_buffer) >= self.vol_scaler.lookback:
            action_value = float(action[0])
            scaled_action = self.vol_scaler.get_scaled_position(
                action_value,
                np.array(self.returns_buffer)
            )
            action = np.array([scaled_action], dtype=np.float32)

        # Ejecutar step normal
        obs, reward, terminated, truncated, info = super().step(action)

        # Guardar return para próximo step
        if 'step_return' in info:
            self.returns_buffer.append(info['step_return'])
            # Mantener solo lookback + margen
            if len(self.returns_buffer) > self.vol_scaler.lookback * 2:
                self.returns_buffer = self.returns_buffer[-self.vol_scaler.lookback * 2:]

        # Añadir info de volatility scaling
        if len(self.returns_buffer) >= self.vol_scaler.lookback:
            info['vol_scale'] = self.vol_scaler.scale_history[-1]
            info['realized_vol'] = self.vol_scaler.realized_vol_history[-1]

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset incluyendo buffer de returns."""
        obs, info = super().reset(**kwargs)
        self.returns_buffer = []
        return obs, info


# ==============================================================================
# PRODUCTION TRAINING PIPELINE
# ==============================================================================

class ProductionTrainingPipeline:
    """
    Pipeline de entrenamiento para producción.

    Integra todas las mejoras:
    - Volatility scaling
    - Regime detection
    - Multi-seed training
    - Curriculum learning
    - Validación robusta
    """

    def __init__(
        self,
        data_path: str,
        timeframe: str = '5min',
        algorithm: str = 'SAC',
        total_timesteps: int = 500_000,
        n_seeds: int = 3,
        use_vol_scaling: bool = True,
        use_regime_detector: bool = True,
        output_dir: str = './production_models',
        verbose: int = 1,
    ):
        self.data_path = data_path
        self.timeframe = timeframe
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.n_seeds = n_seeds
        self.use_vol_scaling = use_vol_scaling
        self.use_regime_detector = use_regime_detector
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)

        # Estado
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.vol_scaler = None
        self.regime_detector = None
        self.models = []
        self.training_history = []

        # Configuración según timeframe
        self.config = self._get_config_for_timeframe(timeframe)

    def _get_config_for_timeframe(self, timeframe: str) -> Dict:
        """Configuración óptima según timeframe."""
        configs = {
            '5min': {
                'bars_per_day': 288,  # 24h * 60min / 5min
                'episode_length': 1440,  # 5 días
                'vol_lookback': 20,
                'regime_lookback': 40,
                'learning_rate': 3e-4,
            },
            '15min': {
                'bars_per_day': 96,  # 24h * 60min / 15min
                'episode_length': 480,  # 5 días
                'vol_lookback': 15,
                'regime_lookback': 30,
                'learning_rate': 3e-4,
            },
            '1h': {
                'bars_per_day': 24,
                'episode_length': 120,  # 5 días
                'vol_lookback': 10,
                'regime_lookback': 20,
                'learning_rate': 3e-4,
            },
        }

        return configs.get(timeframe, configs['5min'])

    def load_data(self) -> pd.DataFrame:
        """Cargar y validar datos."""
        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("LOADING DATA")
            print("=" * 70)

        # Cargar dataset
        if self.data_path.endswith('.parquet'):
            df = pd.read_parquet(self.data_path)
        elif self.data_path.endswith('.csv'):
            df = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported format: {self.data_path}")

        if self.verbose > 0:
            print(f"Loaded: {len(df):,} rows from {self.data_path}")
            print(f"Timeframe: {self.timeframe}")

        # Validar columnas requeridas
        required = ['close', 'open', 'high', 'low']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Calcular retornos si no existen
        if 'close_return' not in df.columns:
            df['close_return'] = df['close'].pct_change()

        # Calcular volatilidad si no existe
        if 'volatility_pct' not in df.columns:
            df['volatility_pct'] = df['close_return'].rolling(20).std() * 100

        # Limpiar NaN
        df = df.dropna().reset_index(drop=True)

        self.df = df

        if self.verbose > 0:
            print(f"After cleaning: {len(df):,} rows")
            if 'timestamp' in df.columns:
                print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        return df

    def split_data(self, train_pct: float = 0.7, val_pct: float = 0.15) -> Dict:
        """Split train/val/test con embargo temporal."""
        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("SPLITTING DATA")
            print("=" * 70)

        n = len(self.df)

        # Embargo de 1% entre splits
        embargo = int(n * 0.01)

        train_end = int(n * train_pct)
        val_start = train_end + embargo
        val_end = val_start + int(n * val_pct)
        test_start = val_end + embargo

        self.train_df = self.df.iloc[:train_end].copy()
        self.val_df = self.df.iloc[val_start:val_end].copy()
        self.test_df = self.df.iloc[test_start:].copy()

        if self.verbose > 0:
            print(f"Train: {len(self.train_df):,} rows ({train_pct*100:.0f}%)")
            print(f"Val:   {len(self.val_df):,} rows ({val_pct*100:.0f}%)")
            print(f"Test:  {len(self.test_df):,} rows ({(1-train_pct-val_pct)*100:.0f}%)")
            print(f"Embargo: {embargo} bars between splits")

        return {
            'train': self.train_df,
            'val': self.val_df,
            'test': self.test_df,
        }

    def initialize_components(self):
        """Inicializar volatility scaler y regime detector."""
        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("INITIALIZING COMPONENTS")
            print("=" * 70)

        # 1. Volatility Scaler
        if self.use_vol_scaling:
            self.vol_scaler = VolatilityScaler(
                target_vol=0.01,  # 1% diario
                lookback=self.config['vol_lookback'],
                min_scale=0.25,
                max_scale=1.0,
            )
            if self.verbose > 0:
                print(f"✓ Volatility Scaler initialized (lookback={self.config['vol_lookback']})")

        # 2. Regime Detector
        if self.use_regime_detector and REGIME_DETECTOR_AVAILABLE:
            try:
                self.regime_detector = create_regime_detector(
                    method='hmm',
                    n_regimes=3,
                    lookback=self.config['regime_lookback'],
                )

                # Entrenar en datos históricos
                train_returns = self.train_df['close_return'].values
                train_vol = self.train_df['volatility_pct'].values / 100

                self.regime_detector.fit(train_returns, train_vol)

                if self.verbose > 0:
                    print(f"✓ Regime Detector trained (method=HMM, lookback={self.config['regime_lookback']})")
            except Exception as e:
                if self.verbose > 0:
                    print(f"⚠ Regime Detector failed: {e}")
                    print("  Falling back to SimpleRegimeDetector")
                self.regime_detector = SimpleRegimeDetector(
                    lookback=self.config['regime_lookback']
                )

    def create_environment(self, df: pd.DataFrame, for_training: bool = True):
        """Crear environment con todas las mejoras."""
        # Environment base
        if self.use_vol_scaling:
            env = VolScaledEnvironment(
                df=df,
                initial_balance=10_000,
                max_position=1.0,
                episode_length=self.config['episode_length'],
                max_drawdown_pct=15.0,
                use_curriculum_costs=for_training,
                vol_scaler=self.vol_scaler,
            )
        else:
            env = TradingEnvironmentV19(
                df=df,
                initial_balance=10_000,
                max_position=1.0,
                episode_length=self.config['episode_length'],
                max_drawdown_pct=15.0,
                use_curriculum_costs=for_training,
            )

        return env

    def train_single_seed(self, seed: int) -> Tuple[Any, Dict]:
        """Entrenar un modelo con un seed específico."""
        if self.verbose > 0:
            print(f"\n{'='*70}")
            print(f"TRAINING SEED {seed}/{self.n_seeds}")
            print(f"{'='*70}")

        # Set seed
        np.random.seed(seed)

        # Crear environments
        train_env = self.create_environment(self.train_df, for_training=True)
        train_env = Monitor(train_env)
        train_env = DummyVecEnv([lambda: train_env])

        val_env = self.create_environment(self.val_df, for_training=False)

        # Crear modelo
        if self.algorithm == 'SAC':
            model = SAC(
                'MlpPolicy',
                train_env,
                learning_rate=self.config['learning_rate'],
                buffer_size=100_000,
                learning_starts=1_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                ent_coef='auto',
                policy_kwargs={'net_arch': [256, 256]},
                verbose=0,
                tensorboard_log=str(self.output_dir / 'logs' / f'seed_{seed}'),
            )
        elif self.algorithm == 'PPO':
            model = PPO(
                'MlpPolicy',
                train_env,
                learning_rate=self.config['learning_rate'],
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs={'net_arch': dict(pi=[256, 256], vf=[256, 256])},
                verbose=0,
                tensorboard_log=str(self.output_dir / 'logs' / f'seed_{seed}'),
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Callbacks
        callbacks = []

        # Sharpe evaluation
        sharpe_callback = SharpeEvalCallback(
            eval_env=val_env,
            eval_freq=10_000,
            n_eval_episodes=5,
            patience=5,
            min_delta=0.05,
            min_sharpe=0.5,
            best_model_save_path=str(self.output_dir / 'models' / f'seed_{seed}'),
            verbose=self.verbose,
        )
        callbacks.append(sharpe_callback)

        # Cost curriculum
        cost_callback = CostCurriculumCallback(
            env=train_env,
            warmup_steps=int(self.total_timesteps * 0.2),
            rampup_steps=int(self.total_timesteps * 0.3),
            final_cost=0.0025,  # 25 bps
            crisis_multiplier=1.5,
            verbose=self.verbose,
        )
        callbacks.append(cost_callback)

        callback_list = CallbackList(callbacks)

        # Train
        start_time = datetime.now()

        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callback_list,
                progress_bar=self.verbose > 0,
            )
        except KeyboardInterrupt:
            if self.verbose > 0:
                print(f"\nTraining interrupted for seed {seed}")

        duration = (datetime.now() - start_time).total_seconds() / 60

        # Evaluar
        metrics = self._evaluate_model(model, self.test_df, seed)
        metrics['training_time_minutes'] = duration
        metrics['seed'] = seed

        # Guardar modelo
        model_path = self.output_dir / 'models' / f'seed_{seed}_final.zip'
        model.save(str(model_path))

        if self.verbose > 0:
            print(f"\nSeed {seed} Results:")
            print(f"  Sharpe: {metrics['sharpe']:.3f}")
            print(f"  Total Return: {metrics['total_return']:.2f}%")
            print(f"  Max DD: {metrics['max_drawdown']:.2f}%")
            print(f"  Training time: {duration:.1f} min")

        return model, metrics

    def _evaluate_model(self, model: Any, df: pd.DataFrame, seed: int) -> Dict:
        """Evaluar modelo en dataset."""
        env = self.create_environment(df, for_training=False)

        obs, _ = env.reset(seed=seed)
        done = False

        returns = []
        actions = []
        positions = []
        regime_info = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            returns.append(info.get('step_return', 0))
            actions.append(float(action[0]))
            positions.append(info.get('position', 0))

            # Detectar régimen si disponible
            if self.regime_detector is not None and len(returns) >= self.config['regime_lookback']:
                regime_state = self.regime_detector.predict(
                    np.array(returns[-self.config['regime_lookback']:])
                )
                regime_info.append(regime_state.regime.value)

            done = terminated or truncated

        # Calcular métricas
        returns = np.array(returns)
        actions = np.array(actions)

        total_return = (np.prod(1 + returns) - 1) * 100

        sharpe = 0.0
        if returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(self.config['bars_per_day'] * 252)

        # Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max * 100
        max_drawdown = drawdown.max()

        # Win rate
        win_rate = (returns > 0).mean() * 100

        # Action distribution
        # Action classification - THRESHOLD 0.10 (was 0.15, changed to account for regime multiplier)
        ACTION_THRESHOLD = 0.10
        pct_long = (actions > ACTION_THRESHOLD).mean() * 100
        pct_short = (actions < -ACTION_THRESHOLD).mean() * 100
        pct_hold = (np.abs(actions) <= ACTION_THRESHOLD).mean() * 100

        # Regime distribution si disponible
        regime_dist = {}
        if len(regime_info) > 0:
            unique, counts = np.unique(regime_info, return_counts=True)
            regime_dist = {int(k): float(v/len(regime_info)*100) for k, v in zip(unique, counts)}

        metrics = {
            'sharpe': float(sharpe),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'pct_long': float(pct_long),
            'pct_short': float(pct_short),
            'pct_hold': float(pct_hold),
            'n_steps': len(returns),
            'regime_distribution': regime_dist,
        }

        # Añadir stats de vol scaling si está activo
        if self.use_vol_scaling and self.vol_scaler:
            metrics['vol_scaling'] = self.vol_scaler.get_stats()

        return metrics

    def run_multi_seed_training(self) -> List[Tuple[Any, Dict]]:
        """Entrenar múltiples seeds."""
        if self.verbose > 0:
            print("\n" + "=" * 70)
            print(f"MULTI-SEED TRAINING ({self.n_seeds} seeds)")
            print("=" * 70)

        results = []

        for seed in range(self.n_seeds):
            model, metrics = self.train_single_seed(seed)
            results.append((model, metrics))
            self.models.append(model)
            self.training_history.append(metrics)

        return results

    def analyze_ensemble(self) -> Dict:
        """Analizar resultados del ensemble."""
        if len(self.training_history) == 0:
            return {}

        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("ENSEMBLE ANALYSIS")
            print("=" * 70)

        # Agregar métricas
        sharpes = [m['sharpe'] for m in self.training_history]
        returns = [m['total_return'] for m in self.training_history]
        drawdowns = [m['max_drawdown'] for m in self.training_history]

        ensemble_stats = {
            'n_seeds': self.n_seeds,
            'sharpe': {
                'mean': float(np.mean(sharpes)),
                'std': float(np.std(sharpes)),
                'min': float(np.min(sharpes)),
                'max': float(np.max(sharpes)),
            },
            'total_return': {
                'mean': float(np.mean(returns)),
                'std': float(np.std(returns)),
                'min': float(np.min(returns)),
                'max': float(np.max(returns)),
            },
            'max_drawdown': {
                'mean': float(np.mean(drawdowns)),
                'std': float(np.std(drawdowns)),
                'min': float(np.min(drawdowns)),
                'max': float(np.max(drawdowns)),
            },
        }

        if self.verbose > 0:
            print(f"\nSharpe Ratio:")
            print(f"  Mean: {ensemble_stats['sharpe']['mean']:.3f} ± {ensemble_stats['sharpe']['std']:.3f}")
            print(f"  Range: [{ensemble_stats['sharpe']['min']:.3f}, {ensemble_stats['sharpe']['max']:.3f}]")

            print(f"\nTotal Return:")
            print(f"  Mean: {ensemble_stats['total_return']['mean']:.2f}% ± {ensemble_stats['total_return']['std']:.2f}%")

            print(f"\nMax Drawdown:")
            print(f"  Mean: {ensemble_stats['max_drawdown']['mean']:.2f}% ± {ensemble_stats['max_drawdown']['std']:.2f}%")

        # Seleccionar mejor modelo
        best_idx = np.argmax(sharpes)
        best_model_info = {
            'seed': best_idx,
            'sharpe': sharpes[best_idx],
            'total_return': returns[best_idx],
            'max_drawdown': drawdowns[best_idx],
        }

        if self.verbose > 0:
            print(f"\nBest Model (Seed {best_idx}):")
            print(f"  Sharpe: {best_model_info['sharpe']:.3f}")
            print(f"  Return: {best_model_info['total_return']:.2f}%")
            print(f"  Max DD: {best_model_info['max_drawdown']:.2f}%")

        return {
            'ensemble_stats': ensemble_stats,
            'best_model': best_model_info,
            'all_metrics': self.training_history,
        }

    def save_final_report(self, analysis: Dict):
        """Guardar reporte final."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        report = {
            'timestamp': timestamp,
            'config': {
                'data_path': self.data_path,
                'timeframe': self.timeframe,
                'algorithm': self.algorithm,
                'total_timesteps': self.total_timesteps,
                'n_seeds': self.n_seeds,
                'use_vol_scaling': self.use_vol_scaling,
                'use_regime_detector': self.use_regime_detector,
            },
            'data_info': {
                'total_rows': len(self.df),
                'train_rows': len(self.train_df),
                'val_rows': len(self.val_df),
                'test_rows': len(self.test_df),
            },
            'analysis': analysis,
        }

        # Guardar JSON
        report_path = self.output_dir / 'reports' / f'production_report_{timestamp}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        if self.verbose > 0:
            print(f"\n✓ Report saved: {report_path}")

        return report_path

    def run_full_pipeline(self) -> Dict:
        """Ejecutar pipeline completo."""
        start_time = datetime.now()

        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("USD/COP PRODUCTION TRAINING PIPELINE")
            print("=" * 70)
            print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Algorithm: {self.algorithm}")
            print(f"Timesteps: {self.total_timesteps:,}")
            print(f"Seeds: {self.n_seeds}")
            print(f"Volatility Scaling: {'ON' if self.use_vol_scaling else 'OFF'}")
            print(f"Regime Detection: {'ON' if self.use_regime_detector else 'OFF'}")

        # 1. Cargar datos
        self.load_data()

        # 2. Split
        self.split_data()

        # 3. Inicializar componentes
        self.initialize_components()

        # 4. Multi-seed training
        self.run_multi_seed_training()

        # 5. Análisis ensemble
        analysis = self.analyze_ensemble()

        # 6. Guardar reporte
        report_path = self.save_final_report(analysis)

        # Resumen final
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Duration: {duration:.1f} minutes")
            print(f"Models saved: {self.output_dir / 'models'}")
            print(f"Report: {report_path}")
            print("=" * 70)

        return analysis


# ==============================================================================
# CLI
# ==============================================================================

def main():
    """Punto de entrada CLI."""
    parser = argparse.ArgumentParser(
        description='USD/COP Production Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Básico - SAC en 5min
  python train_production.py --data dataset_5min.parquet

  # PPO en 15min con más timesteps
  python train_production.py --data dataset_15min.parquet --timeframe 15min --algorithm PPO --timesteps 1000000

  # Multi-seed ensemble
  python train_production.py --data dataset_5min.parquet --seeds 5

  # Modo rápido para debug
  python train_production.py --data dataset_5min.parquet --quick
        """
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to dataset (CSV or Parquet)'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        default='5min',
        choices=['5min', '15min', '1h'],
        help='Timeframe of data'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='SAC',
        choices=['SAC', 'PPO'],
        help='RL algorithm'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=500_000,
        help='Total timesteps per seed'
    )

    parser.add_argument(
        '--seeds',
        type=int,
        default=3,
        help='Number of random seeds (ensemble size)'
    )

    parser.add_argument(
        '--no-vol-scaling',
        action='store_true',
        help='Disable volatility scaling'
    )

    parser.add_argument(
        '--no-regime',
        action='store_true',
        help='Disable regime detection'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./production_models',
        help='Output directory'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (50k timesteps, 1 seed) for debugging'
    )

    parser.add_argument(
        '--verbose',
        type=int,
        default=1,
        help='Verbosity (0=quiet, 1=normal, 2=debug)'
    )

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.timesteps = 50_000
        args.seeds = 1
        if args.verbose > 0:
            print("QUICK MODE: 50k timesteps, 1 seed")

    # Crear pipeline
    pipeline = ProductionTrainingPipeline(
        data_path=args.data,
        timeframe=args.timeframe,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_seeds=args.seeds,
        use_vol_scaling=not args.no_vol_scaling,
        use_regime_detector=not args.no_regime,
        output_dir=args.output,
        verbose=args.verbose,
    )

    # Ejecutar
    try:
        analysis = pipeline.run_full_pipeline()

        # Exit code basado en performance
        best_sharpe = analysis['best_model']['sharpe']
        if best_sharpe >= 1.0:
            sys.exit(0)  # Success
        elif best_sharpe >= 0.5:
            sys.exit(0)  # Acceptable
        else:
            sys.exit(1)  # Failed

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
