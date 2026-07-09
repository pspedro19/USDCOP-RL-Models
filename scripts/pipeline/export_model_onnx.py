#!/usr/bin/env python3
"""
ONNX Model Export Script for USDCOP Trading Models
===================================================

Exports trained PyTorch/Stable-Baselines3 models to ONNX format
for optimized production inference.

Benefits:
- 5-10x faster inference (50ms → 5ms)
- No PyTorch dependency in production
- Cross-platform compatibility
- Optimized for CPU/GPU inference

Usage:
    python scripts/mlops/export_model_onnx.py --model-path models/ppo_v3.zip --output models/onnx/

Author: USDCOP Trading Team
Version: 1.0.0
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

import torch
import torch.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class PolicyNetworkWrapper(nn.Module):
    """
    Wrapper for Stable-Baselines3 policy networks to export to ONNX.
    Handles the extraction of action probabilities and values.
    """

    def __init__(self, policy_net: nn.Module, action_space_size: int = 3):
        super().__init__()
        self.policy_net = policy_net
        self.action_space_size = action_space_size

    def forward(self, observation: torch.Tensor) -> tuple:
        """
        Forward pass that returns action probabilities and confidence.

        Args:
            observation: Input tensor of shape (batch_size, observation_dim)

        Returns:
            action_probs: Probability distribution over actions
            confidence: Maximum probability (confidence in prediction)
        """
        # Get latent features
        features = self.policy_net.mlp_extractor(observation)

        # Get action logits
        if hasattr(self.policy_net, 'action_net'):
            action_logits = self.policy_net.action_net(features[0])
        else:
            action_logits = self.policy_net.policy_net(features[0])

        # Convert to probabilities
        action_probs = torch.softmax(action_logits, dim=-1)

        # Get confidence (max probability)
        confidence = torch.max(action_probs, dim=-1, keepdim=True)[0]

        return action_probs, confidence


def load_sb3_model(model_path: str, algorithm: str = "PPO"):
    """
    Load a Stable-Baselines3 model.

    Args:
        model_path: Path to the .zip model file
        algorithm: Algorithm type (PPO, SAC, A2C)

    Returns:
        Loaded model
    """
    try:
        if algorithm.upper() == "PPO":
            from stable_baselines3 import PPO
            return PPO.load(model_path)
        elif algorithm.upper() == "SAC":
            from stable_baselines3 import SAC
            return SAC.load(model_path)
        elif algorithm.upper() == "A2C":
            from stable_baselines3 import A2C
            return A2C.load(model_path)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def export_to_onnx(
    model,
    observation_dim: int,
    output_path: str,
    opset_version: int = 14,
    dynamic_batch: bool = True,
    optimize: bool = True
) -> Dict[str, Any]:
    """
    Export a model to ONNX format.

    Args:
        model: Stable-Baselines3 model or PyTorch model
        observation_dim: Dimension of observation space
        output_path: Path to save the ONNX model
        opset_version: ONNX opset version
        dynamic_batch: Enable dynamic batch size
        optimize: Apply ONNX optimizations

    Returns:
        Dictionary with export metadata
    """
    logger.info(f"Exporting model to ONNX...")
    logger.info(f"  Observation dimension: {observation_dim}")
    logger.info(f"  Output path: {output_path}")

    # Extract policy network
    if hasattr(model, 'policy'):
        policy = model.policy
    else:
        policy = model

    # Create wrapper for clean output
    try:
        wrapped_model = PolicyNetworkWrapper(policy)
        wrapped_model.eval()
    except Exception as e:
        logger.warning(f"Could not wrap model, using direct export: {e}")
        wrapped_model = policy
        wrapped_model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, observation_dim, dtype=torch.float32)

    # Define input/output names
    input_names = ['observation']
    output_names = ['action_probs', 'confidence']

    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'observation': {0: 'batch_size'},
            'action_probs': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Export to ONNX
    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
        logger.info(f"✅ Model exported to {output_path}")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        raise

    # Optimize ONNX model
    if optimize:
        try:
            import onnx
            from onnxruntime.transformers import optimizer

            optimized_path = output_path.replace('.onnx', '_optimized.onnx')

            # Load and optimize
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)

            # Apply optimizations
            optimized_model = optimizer.optimize_model(
                output_path,
                model_type='bert',  # Generic optimization
                num_heads=0,
                hidden_size=0
            )
            optimized_model.save_model_to_file(optimized_path)

            logger.info(f"✅ Optimized model saved to {optimized_path}")
        except ImportError:
            logger.warning("onnxruntime.transformers not available, skipping optimization")
        except Exception as e:
            logger.warning(f"Optimization failed (using unoptimized): {e}")

    # Verify export
    try:
        import onnx
        import onnxruntime as ort

        # Load and check
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        # Test inference
        session = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        test_input = np.random.randn(1, observation_dim).astype(np.float32)
        outputs = session.run(None, {'observation': test_input})

        logger.info(f"✅ Verification passed")
        logger.info(f"  Output shapes: {[o.shape for o in outputs]}")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise

    # Create metadata
    metadata = {
        "export_timestamp": datetime.now().isoformat(),
        "observation_dim": observation_dim,
        "opset_version": opset_version,
        "input_names": input_names,
        "output_names": output_names,
        "dynamic_batch": dynamic_batch,
        "file_size_mb": os.path.getsize(output_path) / (1024 * 1024),
        "onnx_path": output_path,
    }

    # Save metadata
    metadata_path = output_path.replace('.onnx', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Metadata saved to {metadata_path}")

    return metadata


def export_ensemble(
    models_config: list,
    output_dir: str,
    observation_dim: int
) -> Dict[str, Any]:
    """
    Export multiple models for ensemble inference.

    Args:
        models_config: List of dicts with 'path', 'algorithm', 'name'
        output_dir: Directory to save ONNX models
        observation_dim: Observation dimension

    Returns:
        Ensemble metadata
    """
    logger.info(f"Exporting ensemble of {len(models_config)} models...")

    os.makedirs(output_dir, exist_ok=True)

    ensemble_metadata = {
        "export_timestamp": datetime.now().isoformat(),
        "models": [],
        "observation_dim": observation_dim,
    }

    for config in models_config:
        model_name = config.get('name', Path(config['path']).stem)
        algorithm = config.get('algorithm', 'PPO')

        logger.info(f"\nExporting {model_name} ({algorithm})...")

        try:
            # Load model
            model = load_sb3_model(config['path'], algorithm)

            # Export
            output_path = os.path.join(output_dir, f"{model_name}.onnx")
            metadata = export_to_onnx(model, observation_dim, output_path)

            ensemble_metadata["models"].append({
                "name": model_name,
                "algorithm": algorithm,
                "onnx_path": output_path,
                "metadata": metadata
            })

        except Exception as e:
            logger.error(f"Failed to export {model_name}: {e}")
            continue

    # Save ensemble metadata
    ensemble_path = os.path.join(output_dir, "ensemble_metadata.json")
    with open(ensemble_path, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)

    logger.info(f"\n✅ Ensemble exported: {len(ensemble_metadata['models'])} models")
    logger.info(f"✅ Ensemble metadata: {ensemble_path}")

    return ensemble_metadata


def main():
    parser = argparse.ArgumentParser(
        description="Export trading models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single model
  python export_model_onnx.py --model-path models/ppo_v3.zip --output models/onnx/ppo_v3.onnx

  # Export with custom observation dimension
  python export_model_onnx.py --model-path models/ppo_v3.zip --obs-dim 45 --output models/onnx/

  # Export ensemble
  python export_model_onnx.py --ensemble --config models/ensemble_config.json
        """
    )

    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to the model file (.zip for SB3)'
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        default='PPO',
        choices=['PPO', 'SAC', 'A2C'],
        help='Model algorithm type'
    )
    parser.add_argument(
        '--obs-dim',
        type=int,
        default=45,
        help='Observation space dimension'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/onnx/model.onnx',
        help='Output path for ONNX model'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=14,
        help='ONNX opset version'
    )
    parser.add_argument(
        '--no-optimize',
        action='store_true',
        help='Skip ONNX optimization'
    )
    parser.add_argument(
        '--ensemble',
        action='store_true',
        help='Export ensemble of models'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to ensemble config JSON'
    )

    args = parser.parse_args()

    if args.ensemble:
        if not args.config:
            logger.error("--config required for ensemble export")
            sys.exit(1)

        with open(args.config) as f:
            config = json.load(f)

        export_ensemble(
            config['models'],
            config.get('output_dir', 'models/onnx'),
            config.get('observation_dim', args.obs_dim)
        )
    else:
        if not args.model_path:
            logger.error("--model-path required")
            sys.exit(1)

        model = load_sb3_model(args.model_path, args.algorithm)
        export_to_onnx(
            model,
            args.obs_dim,
            args.output,
            opset_version=args.opset,
            optimize=not args.no_optimize
        )

    logger.info("\n🚀 Export complete!")


if __name__ == "__main__":
    main()
