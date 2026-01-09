"""
USD/COP RL Trading System - Training Pipeline V19
==================================================

Script principal de entrenamiento que integra todos los componentes V19.

CARACTERÍSTICAS:
- Curriculum learning (3 fases)
- Costos realistas SET-FX
- Multi-seed training
- Purged walk-forward CV
- Stress testing en crisis
- Early stopping por Sharpe

EJECUCIÓN:
    python train_v19.py --config config/training_config_v19.yaml
    python train_v19.py --quick  # Modo rápido para debug

Author: Claude Code
Version: 1.0.0
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

# Stable-Baselines3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Componentes V19
from config.training_config_v19 import TrainingConfigV19, load_config
from rewards import get_reward_function, SymmetricCurriculumReward
from callbacks import (
    SharpeEvalCallback,
    EntropySchedulerCallback,
    ActionDistributionCallback,
    CostCurriculumCallback,
)
from environment_v19 import (
    TradingEnvironmentV19,
    create_training_env,
    create_validation_env,
)
from validation import (
    PurgedKFoldCV,
    TimeSeriesSplit,
    TradingMetrics,
    calculate_all_metrics,
    MultiSeedTrainer,
    BootstrapConfidenceInterval,
    StressTester,
    CrisisPeriodsValidator,
)


class TrainingPipelineV19:
    """
    Pipeline de entrenamiento V19 completo.

    Orquesta:
    1. Carga y preprocesamiento de datos
    2. Configuración de environment y reward
    3. Entrenamiento con curriculum learning
    4. Validación robusta
    5. Stress testing
    6. Guardado de modelos y reportes
    """

    def __init__(
        self,
        config: TrainingConfigV19,
        data_path: str,
        output_dir: str = "./outputs",
        verbose: int = 1,
    ):
        self.config = config
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # Crear directorios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        # Estado
        self.df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.reward_function = None
        self.best_model = None
        self.training_history = {}

    def load_data(self) -> pd.DataFrame:
        """Cargar y preprocesar datos."""
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("LOADING DATA")
            print("=" * 60)

        # Cargar
        if self.data_path.endswith('.parquet'):
            df = pd.read_parquet(self.data_path)
        else:
            df = pd.read_csv(self.data_path)

        if self.verbose > 0:
            print(f"Loaded {len(df):,} rows from {self.data_path}")

        # Verificar columnas requeridas
        required = ['close', 'open', 'high', 'low']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

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
            print(f"Date range: {df.iloc[0].get('timestamp', 'N/A')} to {df.iloc[-1].get('timestamp', 'N/A')}")

        return df

    def split_data(self) -> Dict[str, pd.DataFrame]:
        """Dividir datos en train/val/test con embargo."""
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("SPLITTING DATA")
            print("=" * 60)

        splitter = TimeSeriesSplit(
            train_pct=self.config.validation.train_pct,
            val_pct=self.config.validation.val_pct,
            test_pct=self.config.validation.test_pct,
            embargo_bars=self.config.validation.embargo_bars,
        )

        splits = splitter.get_dataframes(self.df)

        self.train_df = splits['train']
        self.val_df = splits['val']
        self.test_df = splits['test']

        if self.verbose > 0:
            print(f"Train: {len(self.train_df):,} rows ({self.config.validation.train_pct*100:.0f}%)")
            print(f"Val:   {len(self.val_df):,} rows ({self.config.validation.val_pct*100:.0f}%)")
            print(f"Test:  {len(self.test_df):,} rows ({self.config.validation.test_pct*100:.0f}%)")
            print(f"Embargo: {self.config.validation.embargo_bars} bars between splits")

        return splits

    def create_environments(self) -> Tuple[Any, Any]:
        """Crear environments de training y validación."""
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("CREATING ENVIRONMENTS")
            print("=" * 60)

        # Determinar tipo de reward (default: alpha para V19)
        reward_type = getattr(self.config, 'reward_type', 'alpha')

        # Crear reward function según tipo
        if reward_type == 'symmetric':
            # SymmetricCurriculumReward usa RewardConfig internamente
            # Solo pasamos total_timesteps, config usa defaults del objeto self.config.reward
            self.reward_function = get_reward_function(
                reward_type='symmetric',
                total_timesteps=self.config.total_timesteps,
                config=self.config.reward,  # Pasar el objeto RewardConfig
            )
        elif reward_type in ('alpha', 'alpha_v2'):
            # Alpha-based reward (no penaliza HOLD, basada en skill sobre mercado)
            # Convertir cost_bps a decimal (25 bps -> 0.0025)
            final_cost = getattr(self.config.reward, 'realistic_min_cost_bps', 25.0) / 10000
            self.reward_function = get_reward_function(
                reward_type=reward_type,
                final_cost=final_cost,
                total_steps=self.config.total_timesteps,
                phase_boundaries=self.config.reward.phase_boundaries,
                drawdown_threshold=self.config.acceptance.max_drawdown,
                verbose=self.verbose,
            )
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}. Use 'symmetric', 'alpha', or 'alpha_v2'")

        # Training environment
        train_env = create_training_env(
            df=self.train_df,
            config=self.config,
            reward_function=self.reward_function,
        )

        # Wrap para SB3
        train_env = Monitor(train_env)
        train_env = DummyVecEnv([lambda: train_env])

        # Validation environment (costos completos)
        val_env = create_validation_env(
            df=self.val_df,
            config=self.config,
            reward_function=self.reward_function,
        )

        if self.verbose > 0:
            print(f"Train env observation space: {train_env.observation_space.shape}")
            print(f"Train env action space: {train_env.action_space}")
            print(f"Reward function: {reward_type}")
            print(f"Timeframe: {self.config.environment.timeframe}")

        return train_env, val_env

    def create_callbacks(self, train_env, val_env) -> CallbackList:
        """Crear callbacks de training."""
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("SETTING UP CALLBACKS")
            print("=" * 60)

        callbacks = []

        # 1. Sharpe-based evaluation y early stopping
        sharpe_callback = SharpeEvalCallback(
            eval_env=val_env,
            eval_freq=self.config.callbacks.eval_freq,
            n_eval_episodes=self.config.callbacks.n_eval_episodes,
            patience=self.config.callbacks.patience,
            min_delta=self.config.callbacks.min_delta,
            min_sharpe=self.config.callbacks.min_sharpe,
            best_model_save_path=str(self.output_dir / "models"),
            verbose=self.verbose,
        )
        callbacks.append(sharpe_callback)

        # 2. Entropy scheduler
        entropy_callback = EntropySchedulerCallback(
            init_ent=self.config.callbacks.init_entropy,
            final_ent=self.config.callbacks.final_entropy,
            schedule_type='warmup_cosine',
            warmup_fraction=0.2,
            verbose=self.verbose,
        )
        callbacks.append(entropy_callback)

        # 3. Action distribution monitor
        action_callback = ActionDistributionCallback(
            log_freq=self.config.callbacks.action_log_freq,
            collapse_threshold=self.config.callbacks.collapse_threshold,
            hold_warning_threshold=70.0,
            verbose=self.verbose,
        )
        callbacks.append(action_callback)

        # 4. Cost curriculum
        cost_callback = CostCurriculumCallback(
            env=train_env,
            warmup_steps=self.config.callbacks.cost_warmup_steps,
            rampup_steps=self.config.callbacks.cost_rampup_steps,
            final_cost=self.config.callbacks.cost_final,
            crisis_multiplier=self.config.callbacks.cost_crisis_multiplier,
            verbose=self.verbose,
        )
        callbacks.append(cost_callback)

        if self.verbose > 0:
            print(f"Callbacks configured:")
            print(f"  - SharpeEvalCallback (patience={self.config.callbacks.patience})")
            print(f"  - EntropySchedulerCallback ({self.config.callbacks.init_entropy} -> {self.config.callbacks.final_entropy})")
            print(f"  - ActionDistributionCallback")
            print(f"  - CostCurriculumCallback")

        return CallbackList(callbacks)

    def create_model(self, env) -> Any:
        """Crear modelo según configuración."""
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("CREATING MODEL")
            print("=" * 60)

        # Configuración de red - usar net_arch directamente
        net_arch = self.config.network.net_arch
        policy_kwargs = {
            'net_arch': net_arch,
        }

        if self.config.algorithm == 'PPO':
            ppo_cfg = self.config.ppo
            # Para PPO, usar dict con pi y vf si es necesario
            policy_kwargs['net_arch'] = dict(pi=net_arch, vf=net_arch)
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=ppo_cfg.learning_rate,
                n_steps=ppo_cfg.n_steps,
                batch_size=ppo_cfg.batch_size,
                n_epochs=ppo_cfg.n_epochs,
                gamma=ppo_cfg.gamma,
                gae_lambda=ppo_cfg.gae_lambda,
                clip_range=ppo_cfg.clip_range,
                ent_coef=self.config.callbacks.init_entropy,
                vf_coef=ppo_cfg.vf_coef,
                max_grad_norm=ppo_cfg.max_grad_norm,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                tensorboard_log=str(self.output_dir / "logs"),
            )
        elif self.config.algorithm == 'SAC':
            sac_cfg = self.config.sac
            model = SAC(
                'MlpPolicy',
                env,
                learning_rate=sac_cfg.learning_rate,
                buffer_size=sac_cfg.buffer_size,
                learning_starts=sac_cfg.learning_starts,
                batch_size=sac_cfg.batch_size,
                tau=sac_cfg.tau,
                gamma=sac_cfg.gamma,
                ent_coef=sac_cfg.ent_coef,
                policy_kwargs=policy_kwargs,
                verbose=self.verbose,
                tensorboard_log=str(self.output_dir / "logs"),
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.config.algorithm}")

        if self.verbose > 0:
            print(f"Algorithm: {self.config.algorithm}")
            print(f"Total timesteps: {self.config.total_timesteps:,}")
            print(f"Network architecture: {net_arch}")

        return model

    def train(self) -> Any:
        """Ejecutar training completo."""
        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("STARTING TRAINING")
            print("=" * 60)
            print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Crear environments
        train_env, val_env = self.create_environments()

        # Crear callbacks
        callbacks = self.create_callbacks(train_env, val_env)

        # Crear modelo
        model = self.create_model(train_env)

        # Training
        try:
            model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")

        self.best_model = model

        if self.verbose > 0:
            print(f"\nTraining completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return model

    def evaluate(self, model: Any = None) -> TradingMetrics:
        """Evaluar modelo en test set."""
        if model is None:
            model = self.best_model

        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("EVALUATING ON TEST SET")
            print("=" * 60)

        # Crear environment de test
        test_env = create_validation_env(
            df=self.test_df,
            config=self.config,
            reward_function=self.reward_function,
        )

        # Ejecutar episodio
        obs, _ = test_env.reset()
        done = False
        returns = []
        actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)

            returns.append(info.get('step_return', 0))
            actions.append(float(action[0] if hasattr(action, '__len__') else action))
            done = terminated or truncated

        # Calcular métricas
        metrics = calculate_all_metrics(
            returns=np.array(returns),
            actions=np.array(actions),
            bars_per_day=self.config.environment.bars_per_day,
        )

        if self.verbose > 0:
            print(f"\nTest Set Results:")
            print(f"  Total Return: {metrics.total_return:.2f}%")
            print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
            print(f"  Sortino Ratio: {metrics.sortino_ratio:.3f}")
            print(f"  Max Drawdown: {metrics.max_drawdown:.2f}%")
            print(f"  Win Rate: {metrics.win_rate:.1%}")
            print(f"  Profit Factor: {metrics.profit_factor:.2f}")
            print(f"  Trades: {metrics.n_trades}")
            print(f"\nAction Distribution:")
            print(f"  LONG: {metrics.pct_long:.1f}%")
            print(f"  SHORT: {metrics.pct_short:.1f}%")
            print(f"  HOLD: {metrics.pct_hold:.1f}%")

        return metrics

    def stress_test(self, model: Any = None) -> Dict:
        """Ejecutar stress testing en períodos de crisis."""
        if model is None:
            model = self.best_model

        if self.verbose > 0:
            print("\n" + "=" * 60)
            print("STRESS TESTING")
            print("=" * 60)

        def env_factory(df):
            return create_validation_env(
                df=df,
                config=self.config,
                reward_function=self.reward_function,
            )

        tester = StressTester(verbose=self.verbose)
        report = tester.run(model, env_factory, self.df)

        if self.verbose > 0:
            report.print_report()

        return report.to_dict()

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Ejecutar pipeline completo."""
        start_time = datetime.now()

        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("USD/COP RL TRADING SYSTEM - TRAINING PIPELINE V19")
            print("=" * 70)
            print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Cargar datos
        self.load_data()

        # 2. Dividir datos
        self.split_data()

        # 3. Training
        model = self.train()

        # 4. Evaluación en test
        test_metrics = self.evaluate(model)

        # 5. Stress testing
        stress_results = self.stress_test(model)

        # 6. Verificar criterios de aceptación
        passed, failures = test_metrics.passes_acceptance(
            min_sharpe=self.config.acceptance.min_sharpe,
            max_drawdown=self.config.acceptance.max_drawdown,
            min_calmar=self.config.acceptance.min_calmar,
            min_profit_factor=self.config.acceptance.min_profit_factor,
            min_win_rate=self.config.acceptance.min_win_rate,
            max_hold_pct=self.config.acceptance.max_hold_pct,
        )

        # 7. Guardar modelo final y reporte
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds() / 60

        final_report = {
            'config': {
                'algorithm': self.config.algorithm,
                'total_timesteps': self.config.total_timesteps,
                'timeframe': self.config.environment.timeframe,
                'reward_type': getattr(self.config, 'reward_type', 'symmetric'),
                'bars_per_day': self.config.environment.bars_per_day,
            },
            'test_metrics': test_metrics.to_dict(),
            'stress_test': stress_results,
            'acceptance': {
                'passed': passed,
                'failures': failures,
            },
            'training_time_minutes': duration,
            'timestamp': end_time.isoformat(),
        }

        # Guardar reporte
        report_path = self.output_dir / "reports" / f"training_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)

        # Guardar modelo final
        if passed:
            model_path = self.output_dir / "models" / f"final_model_sharpe_{test_metrics.sharpe_ratio:.3f}"
            model.save(str(model_path))

            if self.verbose > 0:
                print(f"\nModel saved to: {model_path}")

        # Resumen final
        if self.verbose > 0:
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"Duration: {duration:.1f} minutes")
            print(f"Acceptance: {'PASSED' if passed else 'FAILED'}")

            if failures:
                print(f"\nFailures:")
                for f in failures:
                    print(f"  - {f}")

            print(f"\nReports saved to: {self.output_dir / 'reports'}")
            print("=" * 70)

        return final_report


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description='USD/COP RL Trading System - Training V19')

    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Path to data file (CSV or Parquet)',
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to config file (YAML or JSON)',
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./outputs',
        help='Output directory',
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['PPO', 'SAC'],
        default='SAC',
        help='RL algorithm to use',
    )

    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=200_000,
        help='Total training timesteps',
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode for debugging (reduced timesteps)',
    )

    parser.add_argument(
        '--verbose', '-v',
        type=int,
        default=1,
        help='Verbosity level (0=quiet, 1=normal, 2=debug)',
    )

    args = parser.parse_args()

    # Cargar o crear configuración
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfigV19()

    # Aplicar overrides de CLI
    config.algorithm = args.algorithm
    config.total_timesteps = args.timesteps

    if args.quick:
        config.total_timesteps = 10_000
        config.callbacks.eval_freq = 2_000
        config.callbacks.patience = 3

    # Ejecutar pipeline
    pipeline = TrainingPipelineV19(
        config=config,
        data_path=args.data,
        output_dir=args.output,
        verbose=args.verbose,
    )

    results = pipeline.run_full_pipeline()

    # Exit code basado en si pasó los criterios
    sys.exit(0 if results['acceptance']['passed'] else 1)


if __name__ == '__main__':
    main()
