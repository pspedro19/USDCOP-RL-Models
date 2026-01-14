#!/usr/bin/env python3
"""
USD/COP RL Trading System V19 - Quick Run Script
==================================================

Script simplificado para lanzar el entrenamiento.

MODOS DE USO:
-------------

1. Entrenamiento rápido (debug):
   python run_training.py --quick

2. Entrenamiento completo:
   python run_training.py --data path/to/data.parquet

3. Entrenamiento con configuración personalizada:
   python run_training.py --data data.parquet --config config.yaml --timesteps 500000

4. Multi-seed training:
   python run_training.py --data data.parquet --multi-seed 5

EJEMPLO MÍNIMO:
---------------
   python run_training.py --data ../../data/pipeline/07_output/datasets_5min/RL_DS5_MULTIFREQ_28F.parquet

Author: Claude Code
Version: 1.0.0
"""

import os
import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Agregar config al path
config_path = Path(__file__).parent / "config"
sys.path.insert(0, str(config_path))


def main():
    """Punto de entrada principal."""
    import argparse

    parser = argparse.ArgumentParser(
        description='USD/COP RL Trading System V19',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EJEMPLOS:
  python run_training.py --quick
  python run_training.py --data data.parquet
  python run_training.py --data data.parquet --algorithm PPO --timesteps 500000
  python run_training.py --data data.parquet --multi-seed 5
        """
    )

    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Path to training data (CSV or Parquet)',
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to config file (YAML or JSON)',
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./outputs',
        help='Output directory for models and reports',
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['PPO', 'SAC'],
        default='SAC',
        help='RL algorithm (default: SAC)',
    )

    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=200_000,
        help='Total training timesteps (default: 200000)',
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode for debugging (10k steps)',
    )

    parser.add_argument(
        '--multi-seed',
        type=int,
        default=1,
        help='Number of seeds for multi-seed training',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )

    parser.add_argument(
        '--verbose', '-v',
        type=int,
        default=1,
        help='Verbosity level (0=quiet, 1=normal, 2=debug)',
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        choices=['5min', '15min', '1h'],
        default='15min',
        help='Timeframe for training (default: 15min - RECOMMENDED)',
    )

    parser.add_argument(
        '--reward',
        type=str,
        choices=['symmetric', 'alpha', 'alpha_v2'],
        default='alpha',
        help='Reward function: symmetric (SymmetricCurriculumReward), alpha (AlphaCurriculumReward), alpha_v2 (15min optimized)',
    )

    args = parser.parse_args()

    # Validar argumentos
    if not args.quick and not args.data:
        parser.error("--data is required unless using --quick mode")

    # En modo quick, usar datos de demo si no se especifican
    if args.quick:
        args.timesteps = 10_000
        if not args.data:
            # Buscar archivo de datos existente
            possible_paths = [
                "../../data/pipeline/07_output/datasets_5min/RL_DS5_MULTIFREQ_28F.parquet",
                "../../data/pipeline/07_output/datasets_5min/RL_DS5_MULTIFREQ_28F.csv",
                "./demo_data.csv",
            ]
            for p in possible_paths:
                if os.path.exists(p):
                    args.data = p
                    break

            if not args.data:
                print("ERROR: No data file found. Please specify --data path")
                print("Creating demo data for testing...")
                args.data = create_demo_data()

    # Importar después de configurar paths
    from training_config_v19 import TrainingConfigV19, load_config
    from train_v19 import TrainingPipelineV19

    # Cargar configuración
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfigV19()

    # Aplicar overrides
    config.algorithm = args.algorithm
    config.total_timesteps = args.timesteps

    # Configurar timeframe
    config.environment.timeframe = args.timeframe
    if args.timeframe == '5min':
        config.environment.bars_per_day = 60
        config.environment.episode_length = 1200
    elif args.timeframe == '15min':
        config.environment.bars_per_day = 20
        config.environment.episode_length = 400
    elif args.timeframe == '1h':
        config.environment.bars_per_day = 5
        config.environment.episode_length = 100

    # Configurar reward function
    config.reward_type = args.reward  # Se usará en train_v19.py

    if args.quick:
        config.callbacks.eval_freq = 2_000
        config.callbacks.patience = 3
        if args.timeframe == '5min':
            config.environment.episode_length = 500
        elif args.timeframe == '15min':
            config.environment.episode_length = 200
        else:
            config.environment.episode_length = 50

    # Set seed
    import numpy as np
    np.random.seed(args.seed)

    print("\n" + "=" * 70)
    print("USD/COP RL TRADING SYSTEM V19")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Algorithm: {config.algorithm}")
    print(f"  Reward: {args.reward}")
    print(f"  Timesteps: {config.total_timesteps:,}")
    print(f"  Bars/day: {config.environment.bars_per_day}")
    print(f"  Episode length: {config.environment.episode_length}")
    print(f"  Output: {args.output}")
    print(f"  Mode: {'Quick (debug)' if args.quick else 'Full training'}")

    if args.multi_seed > 1:
        print(f"  Multi-seed: {args.multi_seed} seeds")
        run_multi_seed_training(args, config)
    else:
        run_single_training(args, config)


def run_single_training(args, config):
    """Ejecutar entrenamiento con un solo seed."""
    from train_v19 import TrainingPipelineV19

    pipeline = TrainingPipelineV19(
        config=config,
        data_path=args.data,
        output_dir=args.output,
        verbose=args.verbose,
    )

    results = pipeline.run_full_pipeline()

    # Exit code
    sys.exit(0 if results['acceptance']['passed'] else 1)


def run_multi_seed_training(args, config):
    """Ejecutar entrenamiento multi-seed."""
    from train_v19 import TrainingPipelineV19
    from validation.robustness import MultiSeedTrainer, RobustnessReport

    import numpy as np
    import json
    from pathlib import Path

    all_results = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.multi_seed):
        seed = args.seed + i
        print(f"\n{'='*70}")
        print(f"TRAINING SEED {i+1}/{args.multi_seed} (seed={seed})")
        print(f"{'='*70}")

        np.random.seed(seed)

        pipeline = TrainingPipelineV19(
            config=config,
            data_path=args.data,
            output_dir=str(output_dir / f"seed_{seed}"),
            verbose=args.verbose,
        )

        try:
            results = pipeline.run_full_pipeline()
            results['seed'] = seed
            all_results.append(results)
        except Exception as e:
            print(f"Seed {seed} failed: {e}")
            continue

    # Resumen multi-seed
    if all_results:
        print_multi_seed_summary(all_results, output_dir)


def print_multi_seed_summary(results, output_dir):
    """Imprimir resumen de entrenamiento multi-seed."""
    import numpy as np

    print("\n" + "=" * 70)
    print("MULTI-SEED TRAINING SUMMARY")
    print("=" * 70)

    sharpes = [r['test_metrics']['sharpe_ratio'] for r in results]
    max_dds = [r['test_metrics']['max_drawdown'] for r in results]
    passed = [r['acceptance']['passed'] for r in results]

    print(f"\nSeeds tested: {len(results)}")
    print(f"Seeds passed: {sum(passed)}")

    print(f"\nSharpe Ratio:")
    print(f"  Mean: {np.mean(sharpes):.3f}")
    print(f"  Std:  {np.std(sharpes):.3f}")
    print(f"  Min:  {np.min(sharpes):.3f}")
    print(f"  Max:  {np.max(sharpes):.3f}")

    print(f"\nMax Drawdown:")
    print(f"  Mean: {np.mean(max_dds):.2f}%")
    print(f"  Std:  {np.std(max_dds):.2f}%")
    print(f"  Max:  {np.max(max_dds):.2f}%")

    # Mejor seed
    best_idx = np.argmax(sharpes)
    best_seed = results[best_idx]['seed']
    print(f"\nBest seed: {best_seed} (Sharpe={sharpes[best_idx]:.3f})")

    # Consensus
    consensus = sum(passed) / len(passed)
    print(f"\nConsensus (>60% required): {consensus:.0%} - {'PASSED' if consensus >= 0.6 else 'FAILED'}")

    print("=" * 70)


def create_demo_data():
    """Crear datos de demo para testing."""
    import numpy as np
    import pandas as pd

    print("Creating demo data with 10000 rows...")

    np.random.seed(42)
    n = 10000

    # Simular OHLCV
    price = 4000 + np.cumsum(np.random.randn(n) * 10)
    noise = np.random.randn(n) * 5

    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='5min'),
        'open': price + noise,
        'high': price + abs(noise) + 5,
        'low': price - abs(noise) - 5,
        'close': price,
        'volume': np.random.randint(100, 1000, n),
    })

    # Agregar features básicos
    df['close_return'] = df['close'].pct_change()
    df['volatility_pct'] = df['close_return'].rolling(20).std() * 100
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = 50 + np.random.randn(n) * 10  # Simplificado

    df = df.dropna()

    path = "./demo_data.csv"
    df.to_csv(path, index=False)
    print(f"Demo data saved to {path}")

    return path


if __name__ == '__main__':
    main()
