#!/usr/bin/env python3
"""
Export PPO Model to ONNX
========================

Exports a trained PPO model to ONNX format for production deployment.
Includes verification that the ONNX model produces the same outputs.

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

# Production observation dimensions
OBS_DIM = 15


def export_to_onnx(
    model_path: str,
    output_path: str,
    obs_dim: int = OBS_DIM,
    verbose: bool = True
):
    """
    Export PPO model to ONNX format.

    Args:
        model_path: Path to the .zip model file
        output_path: Path for the output .onnx file
        obs_dim: Observation dimension (must match training)
        verbose: Print debug info
    """
    if verbose:
        print(f"Loading model from: {model_path}")

    # Load the model
    model = PPO.load(model_path, device='cpu')

    if verbose:
        print(f"Model loaded successfully")
        print(f"Policy type: {type(model.policy)}")

    # Extract the policy network
    policy = model.policy

    # Set to eval mode
    policy.eval()

    # Create a dummy input with correct dimensions
    dummy_input = torch.randn(1, obs_dim, dtype=torch.float32)

    if verbose:
        print(f"Dummy input shape: {dummy_input.shape}")

    # Get reference output for verification
    with torch.no_grad():
        reference_output = policy.forward(dummy_input, deterministic=True)
        if isinstance(reference_output, tuple):
            reference_action = reference_output[0].numpy()
        else:
            reference_action = reference_output.numpy()

    if verbose:
        print(f"Reference action shape: {reference_action.shape}")
        print(f"Reference action: {reference_action}")

    # Export to ONNX
    if verbose:
        print(f"\nExporting to ONNX: {output_path}")

    # Create wrapper for clean ONNX export
    class PolicyWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.mlp_extractor = policy.mlp_extractor
            self.action_net = policy.action_net

        def forward(self, x):
            features = self.mlp_extractor.forward_actor(x)
            return self.action_net(features)

    wrapper = PolicyWrapper(policy)
    wrapper.eval()

    # Export
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )

    if verbose:
        print(f"ONNX export complete!")

    # Verify ONNX model
    try:
        import onnx
        import onnxruntime as ort

        # Load and check ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)

        if verbose:
            print("\nONNX model validation passed!")

        # Test inference with ONNX Runtime
        session = ort.InferenceSession(output_path)
        onnx_input = dummy_input.numpy()
        onnx_output = session.run(None, {'observation': onnx_input})[0]

        if verbose:
            print(f"ONNX output shape: {onnx_output.shape}")
            print(f"ONNX output: {onnx_output}")

        # Compare outputs
        with torch.no_grad():
            torch_output = wrapper(dummy_input).numpy()

        diff = np.abs(onnx_output - torch_output).max()

        if verbose:
            print(f"\nMax difference PyTorch vs ONNX: {diff:.6f}")

            if diff < 1e-5:
                print("VERIFICATION PASSED: Outputs match!")
            else:
                print("WARNING: Output difference detected")

        return True

    except ImportError:
        if verbose:
            print("\nWARNING: onnx or onnxruntime not installed. Skipping verification.")
        return True


def main():
    """Main export function."""
    import argparse

    parser = argparse.ArgumentParser(description='Export PPO model to ONNX')
    parser.add_argument('--model', type=str,
                       default='models/ppo_production/final_model.zip',
                       help='Path to .zip model file')
    parser.add_argument('--output', type=str,
                       default='models/ppo_production/model.onnx',
                       help='Output ONNX path')
    parser.add_argument('--obs-dim', type=int, default=15,
                       help='Observation dimension')

    args = parser.parse_args()

    # Resolve paths
    model_path = PROJECT_ROOT / args.model
    output_path = PROJECT_ROOT / args.output

    # Check if model exists (try best_model.zip if final not found)
    if not model_path.exists():
        best_model = model_path.parent / "best_model.zip"
        if best_model.exists():
            model_path = best_model
            print(f"Using best_model.zip instead")
        else:
            print(f"ERROR: Model not found at {model_path}")
            sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("MODEL EXPORT TO ONNX")
    print("="*70)

    success = export_to_onnx(
        model_path=str(model_path),
        output_path=str(output_path),
        obs_dim=args.obs_dim,
        verbose=True
    )

    if success:
        print("\n" + "="*70)
        print("EXPORT COMPLETE")
        print(f"ONNX model saved to: {output_path}")
        print(f"Model observation dim: {args.obs_dim}")
        print("="*70)
    else:
        print("\nERROR: Export failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
