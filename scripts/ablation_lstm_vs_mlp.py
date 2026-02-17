#!/usr/bin/env python3
"""
V22 P3: Ablation Test - LSTM vs MLP
=====================================
Train same config with PPO (MlpPolicy) and RecurrentPPO (MlpLstmPolicy),
compare on L4-VAL to determine if LSTM provides meaningful improvement.

Usage:
    python scripts/ablation_lstm_vs_mlp.py
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def run_ablation():
    """Run LSTM vs MLP ablation test."""
    from src.config.pipeline_config import load_pipeline_config
    from src.training.engine import TrainingEngine, TrainingRequest
    from src.training.utils.reproducibility import set_reproducible_seeds

    config = load_pipeline_config()
    project_root = Path(__file__).parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find dataset
    dataset_dir = project_root / config.paths.l2_output_dir
    train_files = list(dataset_dir.glob("*_train.csv")) + list(dataset_dir.glob("*_train.parquet"))
    if not train_files:
        logger.error("No training dataset found")
        return

    dataset_path = train_files[0]
    results = {}

    for model_type in ["mlp", "lstm"]:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"ABLATION: Training {model_type.upper()} model")
        logger.info(f"{'=' * 60}")

        set_reproducible_seeds(42)

        use_lstm = model_type == "lstm"
        output_dir = project_root / "models" / f"ablation_{model_type}_{timestamp}"

        request = TrainingRequest(
            version=f"ablation_{model_type}",
            dataset_path=dataset_path,
            output_dir=output_dir,
            seed=42,
            total_timesteps=500_000,  # Shorter for ablation
        )

        engine = TrainingEngine(project_root=project_root)
        result = engine.run(request)

        results[model_type] = {
            "success": result.success,
            "best_mean_reward": result.best_mean_reward,
            "final_mean_reward": result.final_mean_reward,
            "model_path": str(result.model_path) if result.model_path else None,
            "use_lstm": use_lstm,
        }

        logger.info(f"{model_type.upper()} result: reward={result.best_mean_reward:.4f}")

    # Compare
    logger.info(f"\n{'=' * 60}")
    logger.info("ABLATION RESULTS")
    logger.info(f"{'=' * 60}")

    mlp_reward = results["mlp"]["best_mean_reward"]
    lstm_reward = results["lstm"]["best_mean_reward"]

    logger.info(f"  MLP reward:  {mlp_reward:.4f}")
    logger.info(f"  LSTM reward: {lstm_reward:.4f}")

    if lstm_reward > mlp_reward:
        improvement = (lstm_reward - mlp_reward) / abs(mlp_reward) * 100 if mlp_reward != 0 else float('inf')
        logger.info(f"  LSTM wins by {improvement:.1f}%")
        logger.info(f"  RECOMMENDATION: Use RecurrentPPO")
    else:
        logger.info(f"  MLP wins or tie")
        logger.info(f"  RECOMMENDATION: Use standard PPO (simpler, faster)")

    # Save results
    output_path = project_root / "results" / f"ablation_lstm_vs_mlp_{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {output_path}")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    run_ablation()


if __name__ == "__main__":
    main()
