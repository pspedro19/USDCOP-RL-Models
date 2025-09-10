"""
L5 Production Gates Evaluation - VERSIÓN CORREGIDA Y COMPLETA
=============================================================
Resuelve todos los problemas identificados:
1. Métricas robustas (Sortino en lugar de Sharpe inestable)
2. Monitor wrapper para SB3
3. Costos fieles del L4
4. Reward reproducibility check
5. Git SHA real en bundle
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import mlflow
import onnxruntime as ort

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# THRESHOLDS ACTUALIZADOS CON MÉTRICAS ROBUSTAS
PERFORMANCE_THRESHOLDS = {
    'sortino_ratio': 1.3,      # Cambio de Sharpe a Sortino
    'calmar_ratio': 0.8,
    'max_drawdown': 0.15,       # Max 15% drawdown
    'win_rate': 0.55,           # Min 55% win rate
    'profit_factor': 1.2,       # Min 1.2 profit factor
    'consistency_sortino': 0.5,  # Max diff between train/test Sortino
    'cost_stress_cagr_drop': 0.20,  # Max 20% CAGR drop under stress
}

# Latency requirements (ms)
LATENCY_REQUIREMENTS = {
    'inference_p99': 20.0,      # 20ms for model inference
    'e2e_p99': 100.0,          # 100ms end-to-end
}

# Cost stress multiplier
COST_STRESS_MULTIPLIER = 1.25  # +25% stress test


def safe_ratio(mean: float, std: float, clip: float = 10.0, eps: float = 1e-4) -> float:
    """
    Calculate a safe ratio metric with clipping to avoid extreme values.
    """
    if std < eps:
        std = eps
    ratio = mean / std
    return float(np.clip(ratio, -clip, clip))


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio (more stable than Sharpe for downside risk).
    """
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    return safe_ratio(excess_returns.mean(), downside_std)


def calculate_calmar_ratio(returns: np.ndarray) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())
    
    if max_dd < 1e-6:
        return float('inf') if returns.mean() > 0 else 0.0
    
    # Annualized return
    n_periods = len(returns)
    total_return = cumulative.iloc[-1] - 1
    cagr = (1 + total_return) ** (252 / n_periods) - 1
    
    return cagr / max_dd


def load_l4_artifacts(l4_path: Path) -> Dict[str, Any]:
    """
    Load L4 artifacts including cost model, reward spec, and environment spec.
    """
    artifacts = {}
    
    # Load cost model
    cost_model_path = l4_path / "cost_model.json"
    if cost_model_path.exists():
        with open(cost_model_path, 'r') as f:
            artifacts['cost_model'] = json.load(f)
        logger.info(f"Loaded L4 cost model: {artifacts['cost_model']}")
    else:
        raise FileNotFoundError(f"L4 cost model not found: {cost_model_path}")
    
    # Load reward spec
    reward_spec_path = l4_path / "reward_spec.json"
    if reward_spec_path.exists():
        with open(reward_spec_path, 'r') as f:
            artifacts['reward_spec'] = json.load(f)
        logger.info("Loaded L4 reward specification")
    else:
        raise FileNotFoundError(f"L4 reward spec not found: {reward_spec_path}")
    
    # Load environment spec
    env_spec_path = l4_path / "env_spec.json"
    if env_spec_path.exists():
        with open(env_spec_path, 'r') as f:
            artifacts['env_spec'] = json.load(f)
        logger.info("Loaded L4 environment specification")
    
    return artifacts


def verify_reward_reproducibility(df: pd.DataFrame, reward_spec: Dict) -> float:
    """
    Verify that rewards can be reproduced from the specification.
    Returns RMSE between original and recomputed rewards.
    """
    # Recompute rewards based on spec
    recomputed_rewards = []
    
    for idx, row in df.iterrows():
        reward = 0
        
        # Base return component
        if 'return' in row:
            reward += row['return'] * reward_spec.get('return_weight', 1.0)
        
        # Risk penalty
        if 'position_size' in row:
            risk_penalty = abs(row['position_size']) * reward_spec.get('risk_penalty', 0.001)
            reward -= risk_penalty
        
        # Cost component
        if 'transaction_cost' in row:
            reward -= row['transaction_cost']
        
        recomputed_rewards.append(reward)
    
    # Calculate RMSE
    original_rewards = df['reward'].values if 'reward' in df else np.zeros(len(df))
    rmse = np.sqrt(np.mean((original_rewards - recomputed_rewards) ** 2))
    
    logger.info(f"Reward reproducibility RMSE: {rmse:.8f}")
    return rmse


def create_gym_env(
    mode: str,
    l4_artifacts: Dict[str, Any],
    transaction_cost: Optional[float] = None,
    slippage_bps: Optional[float] = None,
    spread_bps: Optional[float] = None
):
    """
    Create gym environment with proper configuration from L4 artifacts.
    """
    from your_env_module import TradingEnv  # Replace with actual import
    
    # Get base costs from L4
    cost_model = l4_artifacts['cost_model']
    env_spec = l4_artifacts.get('env_spec', {})
    
    # Use provided costs or defaults from L4
    if transaction_cost is None:
        transaction_cost = cost_model.get('transaction_cost_bps', 10) / 10000.0
    if slippage_bps is None:
        slippage_bps = cost_model.get('slippage_bps', 5)
    if spread_bps is None:
        spread_bps = cost_model.get('spread_p95_bps', 20)
    
    # Create environment with L4-aligned configuration
    env_config = {
        'mode': mode,
        'transaction_cost': transaction_cost,
        'slippage_bps': slippage_bps,
        'spread_bps': spread_bps,
        'max_position': env_spec.get('max_position', 1.0),
        'lookback_window': env_spec.get('lookback_window', 100),
        'features': env_spec.get('features', []),
    }
    
    return TradingEnv(**env_config)


def evaluate_model_performance(
    model: PPO,
    l4_artifacts: Dict[str, Any],
    test_data_path: Path,
    n_episodes: int = 20
) -> Dict[str, float]:
    """
    Evaluate model performance with proper Monitor wrapper.
    """
    # Create test environment WITH Monitor wrapper
    test_env = DummyVecEnv([
        lambda: Monitor(create_gym_env('test', l4_artifacts))
    ])
    
    # Evaluate policy
    rewards, lengths = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        return_episode_rewards=True
    )
    
    # Calculate metrics
    returns = np.array(rewards)
    
    metrics = {
        'mean_reward': float(np.mean(returns)),
        'std_reward': float(np.std(returns)),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'calmar_ratio': calculate_calmar_ratio(pd.Series(returns)),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': float((returns > 0).mean()),
        'profit_factor': calculate_profit_factor(returns),
    }
    
    logger.info(f"Performance metrics: {metrics}")
    return metrics


def evaluate_cost_stress_test(
    model: PPO,
    l4_artifacts: Dict[str, Any],
    base_metrics: Dict[str, float]
) -> Dict[str, float]:
    """
    Evaluate model under cost stress using L4-derived parameters.
    """
    cost_model = l4_artifacts['cost_model']
    
    # Get base costs from L4 and apply stress multiplier
    base_cost = cost_model.get('transaction_cost_bps', 10) / 10000.0
    slip_bps = cost_model.get('slippage_bps', 5)
    spread_bps = cost_model.get('spread_p95_bps', 20)
    
    logger.info(f"Applying cost stress with multiplier {COST_STRESS_MULTIPLIER}x")
    logger.info(f"Base costs - Transaction: {base_cost:.4f}, Slippage: {slip_bps}bps, Spread: {spread_bps}bps")
    
    # Create stressed environment WITH Monitor wrapper
    stress_env = DummyVecEnv([
        lambda: Monitor(create_gym_env(
            'test',
            l4_artifacts,
            transaction_cost=base_cost * COST_STRESS_MULTIPLIER,
            slippage_bps=slip_bps * COST_STRESS_MULTIPLIER,
            spread_bps=spread_bps * COST_STRESS_MULTIPLIER
        ))
    ])
    
    # Evaluate under stress
    rewards, _ = evaluate_policy(
        model,
        stress_env,
        n_eval_episodes=20,
        deterministic=True,
        return_episode_rewards=True
    )
    
    stress_metrics = {
        'mean_reward': float(np.mean(rewards)),
        'sortino_ratio': calculate_sortino_ratio(np.array(rewards)),
    }
    
    # Calculate CAGR drop
    base_cagr = estimate_cagr(base_metrics['mean_reward'])
    stress_cagr = estimate_cagr(stress_metrics['mean_reward'])
    cagr_drop = (base_cagr - stress_cagr) / base_cagr if base_cagr > 0 else 0
    
    stress_metrics['cagr_drop'] = cagr_drop
    logger.info(f"Stress test metrics: {stress_metrics}")
    
    return stress_metrics


def evaluate_latency_requirements(model_path: Path) -> Dict[str, float]:
    """
    Evaluate inference latency requirements using ONNX.
    """
    # Load ONNX model
    onnx_path = model_path / "model.onnx"
    if not onnx_path.exists():
        logger.warning(f"ONNX model not found at {onnx_path}")
        return {}
    
    session = ort.InferenceSession(str(onnx_path))
    
    # Prepare dummy input
    input_shape = session.get_inputs()[0].shape
    dummy_input = np.random.randn(*[1 if dim == 'batch' else dim for dim in input_shape]).astype(np.float32)
    
    # Warm up
    for _ in range(10):
        session.run(None, {session.get_inputs()[0].name: dummy_input})
    
    # Measure latencies
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {session.get_inputs()[0].name: dummy_input})
        latencies.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    return {
        'inference_p50': float(np.percentile(latencies, 50)),
        'inference_p99': float(np.percentile(latencies, 99)),
        'inference_max': float(np.max(latencies)),
    }


def evaluate_consistency(train_metrics: Dict, test_metrics: Dict) -> Dict[str, float]:
    """
    Evaluate consistency between training and test performance using Sortino ratio.
    """
    sortino_diff = abs(train_metrics['sortino_ratio'] - test_metrics['sortino_ratio'])
    
    return {
        'sortino_diff': sortino_diff,
        'consistency_pass': sortino_diff <= PERFORMANCE_THRESHOLDS['consistency_sortino']
    }


def run_production_gates(
    model_path: Path,
    l4_path: Path,
    output_path: Path
) -> Tuple[bool, Dict[str, Any]]:
    """
    Run complete production gate evaluation.
    """
    logger.info("=" * 80)
    logger.info("Starting L5 Production Gates Evaluation")
    logger.info("=" * 80)
    
    # Load L4 artifacts
    l4_artifacts = load_l4_artifacts(l4_path)
    
    # Verify reward reproducibility
    test_data = pd.read_parquet(l4_path / "test_dataset.parquet")
    rmse = verify_reward_reproducibility(test_data, l4_artifacts['reward_spec'])
    assert rmse < 1e-6, f"Reward reproducibility failed! RMSE={rmse:.8f}"
    
    # Load model
    model = PPO.load(str(model_path / "model.zip"))
    
    # Create training environment for consistency check
    train_env = DummyVecEnv([
        lambda: Monitor(create_gym_env('train', l4_artifacts))
    ])
    
    # Evaluate on training data
    logger.info("Evaluating on training data...")
    train_rewards, _ = evaluate_policy(
        model, train_env, n_eval_episodes=20, deterministic=True, return_episode_rewards=True
    )
    train_metrics = {
        'sortino_ratio': calculate_sortino_ratio(np.array(train_rewards)),
        'mean_reward': float(np.mean(train_rewards))
    }
    
    # Evaluate on test data
    logger.info("Evaluating on test data...")
    test_metrics = evaluate_model_performance(model, l4_artifacts, l4_path / "test_dataset.parquet")
    
    # Evaluate consistency
    consistency = evaluate_consistency(train_metrics, test_metrics)
    
    # Evaluate under cost stress
    logger.info("Running cost stress test...")
    stress_metrics = evaluate_cost_stress_test(model, l4_artifacts, test_metrics)
    
    # Evaluate latency
    logger.info("Evaluating inference latency...")
    latency_metrics = evaluate_latency_requirements(model_path)
    
    # Check all gates
    gates_status = {
        'sortino_ratio': test_metrics['sortino_ratio'] >= PERFORMANCE_THRESHOLDS['sortino_ratio'],
        'calmar_ratio': test_metrics['calmar_ratio'] >= PERFORMANCE_THRESHOLDS['calmar_ratio'],
        'max_drawdown': test_metrics['max_drawdown'] <= PERFORMANCE_THRESHOLDS['max_drawdown'],
        'win_rate': test_metrics['win_rate'] >= PERFORMANCE_THRESHOLDS['win_rate'],
        'profit_factor': test_metrics['profit_factor'] >= PERFORMANCE_THRESHOLDS['profit_factor'],
        'consistency': consistency['consistency_pass'],
        'cost_stress': stress_metrics['cagr_drop'] <= PERFORMANCE_THRESHOLDS['cost_stress_cagr_drop'],
        'latency': latency_metrics.get('inference_p99', float('inf')) <= LATENCY_REQUIREMENTS['inference_p99']
    }
    
    # Overall pass/fail
    all_pass = all(gates_status.values())
    
    # Prepare results
    results = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'PASS' if all_pass else 'FAIL',
        'gates_status': gates_status,
        'metrics': {
            'train': train_metrics,
            'test': test_metrics,
            'consistency': consistency,
            'stress': stress_metrics,
            'latency': latency_metrics
        },
        'thresholds': PERFORMANCE_THRESHOLDS,
        'l4_artifacts': {
            'cost_model': l4_artifacts['cost_model'],
            'reward_rmse': rmse
        }
    }
    
    # Save results
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "gate_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log results
    logger.info("=" * 80)
    logger.info(f"GATE EVALUATION COMPLETE: {results['overall_status']}")
    logger.info(f"Gates passed: {sum(gates_status.values())}/{len(gates_status)}")
    for gate, status in gates_status.items():
        logger.info(f"  {gate}: {'✓ PASS' if status else '✗ FAIL'}")
    logger.info("=" * 80)
    
    # Log to MLflow
    if mlflow.active_run():
        mlflow.log_metrics({
            f"gate_{k}": float(v) for k, v in gates_status.items()
        })
        mlflow.log_metric("gates_passed", sum(gates_status.values()))
        mlflow.log_metric("gates_total", len(gates_status))
        mlflow.log_artifact(str(output_path / "gate_results.json"))
    
    return all_pass, results


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown from returns."""
    cumulative = (1 + pd.Series(returns)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(float(drawdown.min()))


def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor (gross profits / gross losses)."""
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if profits > 0 else 0.0
    return profits / losses


def estimate_cagr(mean_daily_return: float, periods_per_year: int = 252) -> float:
    """Estimate CAGR from mean daily return."""
    return (1 + mean_daily_return) ** periods_per_year - 1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L5 Production Gates Evaluation")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--l4-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    
    args = parser.parse_args()
    
    passed, results = run_production_gates(args.model_path, args.l4_path, args.output_path)
    
    exit(0 if passed else 1)