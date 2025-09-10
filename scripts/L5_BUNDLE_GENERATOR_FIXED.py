"""
L5 Bundle Generator with Git SHA - VERSIÓN CORREGIDA
====================================================
Fixes:
1. Proper git SHA extraction
2. Complete lineage tracking
3. All artifacts included
"""

import json
import logging
import os
import subprocess
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import onnx
import torch
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


def get_git_info() -> Dict[str, str]:
    """
    Get git information with proper error handling.
    """
    git_info = {
        'sha': 'unknown',
        'branch': 'unknown',
        'tag': None,
        'dirty': False
    }
    
    try:
        # Try multiple methods to get git info
        
        # Method 1: Direct git command
        try:
            git_sha = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            git_info['sha'] = git_sha[:8]  # Short SHA
        except:
            pass
        
        # Method 2: Use git python if available
        if git_info['sha'] == 'unknown':
            try:
                import git
                repo = git.Repo(search_parent_directories=True)
                git_info['sha'] = repo.head.object.hexsha[:8]
                git_info['branch'] = repo.active_branch.name
                git_info['dirty'] = repo.is_dirty()
            except:
                pass
        
        # Method 3: Check environment variable (for CI/CD)
        if git_info['sha'] == 'unknown':
            git_sha = os.environ.get('GIT_COMMIT', os.environ.get('GIT_SHA', 'unknown'))
            if git_sha != 'unknown':
                git_info['sha'] = git_sha[:8]
        
        # Get branch
        if git_info['branch'] == 'unknown':
            try:
                git_branch = subprocess.check_output(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    stderr=subprocess.DEVNULL,
                    text=True
                ).strip()
                git_info['branch'] = git_branch
            except:
                git_info['branch'] = os.environ.get('GIT_BRANCH', 'unknown')
        
        # Check if dirty
        try:
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            git_info['dirty'] = bool(status)
        except:
            pass
        
        logger.info(f"Git info: {git_info}")
        
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
    
    return git_info


def export_to_onnx(model_path: Path, output_path: Path) -> Dict[str, Any]:
    """
    Export PPO model to ONNX format for production deployment.
    """
    logger.info("Exporting model to ONNX...")
    
    # Load PPO model
    model = PPO.load(str(model_path))
    
    # Get the policy network
    policy = model.policy
    
    # Create dummy input
    obs_space = model.observation_space
    if hasattr(obs_space, 'shape'):
        dummy_input = torch.randn(1, *obs_space.shape)
    else:
        dummy_input = torch.randn(1, obs_space.n)
    
    # Export to ONNX
    onnx_path = output_path / "model.onnx"
    torch.onnx.export(
        policy,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    # Verify ONNX model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    # Get model size
    model_size = onnx_path.stat().st_size
    
    # Calculate model hash
    with open(onnx_path, 'rb') as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
    
    return {
        'format': 'onnx',
        'opset_version': 11,
        'size_bytes': model_size,
        'hash': model_hash,
        'input_shape': list(dummy_input.shape),
        'output_shape': 'variable'
    }


def create_deployment_bundle(
    model_path: Path,
    l4_path: Path,
    gate_results_path: Path,
    output_path: Path
) -> Path:
    """
    Create complete deployment bundle with all artifacts and metadata.
    """
    logger.info("Creating deployment bundle...")
    
    # Create bundle directory
    bundle_path = output_path / f"deployment_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    logger.info("Copying model files...")
    shutil.copy2(model_path / "model.zip", bundle_path / "model.zip")
    
    # Export to ONNX
    onnx_info = export_to_onnx(model_path / "model.zip", bundle_path)
    
    # Copy L4 artifacts
    logger.info("Copying L4 artifacts...")
    l4_artifacts_path = bundle_path / "l4_artifacts"
    l4_artifacts_path.mkdir(exist_ok=True)
    
    for artifact in ['cost_model.json', 'env_spec.json', 'reward_spec.json']:
        src = l4_path / artifact
        if src.exists():
            shutil.copy2(src, l4_artifacts_path / artifact)
    
    # Load gate results
    with open(gate_results_path, 'r') as f:
        gate_results = json.load(f)
    
    # Get git info
    git_info = get_git_info()
    
    # Create manifest
    manifest = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'environment': os.environ.get('AIRFLOW_ENV', 'production'),
        
        'git': git_info,
        
        'model': {
            'framework': 'stable-baselines3',
            'algorithm': 'PPO',
            'onnx': onnx_info,
            'training_timesteps': gate_results['metrics']['test'].get('total_timesteps', 1000000),
        },
        
        'performance': {
            'sortino_ratio': gate_results['metrics']['test']['sortino_ratio'],
            'calmar_ratio': gate_results['metrics']['test']['calmar_ratio'],
            'max_drawdown': gate_results['metrics']['test']['max_drawdown'],
            'win_rate': gate_results['metrics']['test']['win_rate'],
        },
        
        'gates': {
            'overall_status': gate_results['overall_status'],
            'gates_passed': sum(gate_results['gates_status'].values()),
            'gates_total': len(gate_results['gates_status']),
            'details': gate_results['gates_status']
        },
        
        'latency': gate_results['metrics'].get('latency', {}),
        
        'l4_lineage': {
            'cost_model_hash': calculate_file_hash(l4_artifacts_path / "cost_model.json"),
            'env_spec_hash': calculate_file_hash(l4_artifacts_path / "env_spec.json"),
            'reward_spec_hash': calculate_file_hash(l4_artifacts_path / "reward_spec.json"),
            'reward_reproducibility_rmse': gate_results['l4_artifacts'].get('reward_rmse', 0)
        },
        
        'deployment': {
            'ready': gate_results['overall_status'] == 'PASS',
            'bundle_path': str(bundle_path),
            'bundle_size_mb': calculate_directory_size(bundle_path) / (1024 * 1024)
        }
    }
    
    # Save manifest
    manifest_path = bundle_path / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create README
    create_bundle_readme(bundle_path, manifest)
    
    # Create deployment config
    create_deployment_config(bundle_path, manifest)
    
    # Create tarball
    tarball_path = output_path / f"bundle_{manifest['git']['sha']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
    logger.info(f"Creating tarball: {tarball_path}")
    
    import tarfile
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(bundle_path, arcname=bundle_path.name)
    
    logger.info(f"Bundle created: {tarball_path}")
    logger.info(f"Bundle size: {tarball_path.stat().st_size / (1024*1024):.2f} MB")
    logger.info(f"Git SHA: {manifest['git']['sha']}")
    logger.info(f"Deployment ready: {manifest['deployment']['ready']}")
    
    return tarball_path


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    if not file_path.exists():
        return "not_found"
    
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def calculate_directory_size(path: Path) -> int:
    """Calculate total size of directory in bytes."""
    total = 0
    for entry in path.rglob('*'):
        if entry.is_file():
            total += entry.stat().st_size
    return total


def create_bundle_readme(bundle_path: Path, manifest: Dict):
    """Create README for deployment bundle."""
    readme_content = f"""
# L5 Model Deployment Bundle

## Model Information
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Stable Baselines3
- **Git SHA**: {manifest['git']['sha']}
- **Branch**: {manifest['git']['branch']}
- **Created**: {manifest['created_at']}

## Performance Metrics
- **Sortino Ratio**: {manifest['performance']['sortino_ratio']:.2f}
- **Calmar Ratio**: {manifest['performance']['calmar_ratio']:.2f}
- **Max Drawdown**: {manifest['performance']['max_drawdown']:.2%}
- **Win Rate**: {manifest['performance']['win_rate']:.2%}

## Gate Status
- **Overall**: {manifest['gates']['overall_status']}
- **Gates Passed**: {manifest['gates']['gates_passed']}/{manifest['gates']['gates_total']}

## Latency
- **Inference P99**: {manifest['latency'].get('inference_p99', 'N/A')} ms
- **E2E P99**: {manifest['latency'].get('e2e_p99', 'N/A')} ms

## Deployment
- **Ready**: {manifest['deployment']['ready']}
- **Bundle Size**: {manifest['deployment']['bundle_size_mb']:.2f} MB

## Files
- `model.zip`: Trained PPO model
- `model.onnx`: ONNX export for production
- `manifest.json`: Complete metadata
- `l4_artifacts/`: L4 configuration files
- `deployment_config.yaml`: Deployment configuration

## Usage
```python
# Load ONNX model
import onnxruntime as ort
session = ort.InferenceSession("model.onnx")
output = session.run(None, {{"input": observation}})
```
"""
    
    with open(bundle_path / "README.md", 'w') as f:
        f.write(readme_content)


def create_deployment_config(bundle_path: Path, manifest: Dict):
    """Create deployment configuration file."""
    import yaml
    
    config = {
        'version': manifest['version'],
        'model': {
            'path': 'model.onnx',
            'format': 'onnx',
            'input_shape': manifest['model']['onnx']['input_shape'],
        },
        'serving': {
            'batch_size': 1,
            'max_batch_size': 32,
            'timeout_ms': 100,
            'num_threads': 4,
        },
        'monitoring': {
            'metrics_port': 9090,
            'health_check_path': '/health',
            'ready_check_path': '/ready',
        },
        'scaling': {
            'min_replicas': 2,
            'max_replicas': 10,
            'target_cpu_utilization': 70,
            'target_latency_p99_ms': 20,
        },
        'environment': {
            'MARKET': 'USDCOP',
            'TRADING_HOURS': '08:00-12:55',
            'TIMEZONE': 'America/Bogota',
        }
    }
    
    with open(bundle_path / "deployment_config.yaml", 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L5 Bundle Generator")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--l4-path", type=Path, required=True)
    parser.add_argument("--gate-results", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create bundle
    bundle_path = create_deployment_bundle(
        model_path=args.model_path,
        l4_path=args.l4_path,
        gate_results_path=args.gate_results,
        output_path=args.output_path
    )
    
    print(f"Bundle created: {bundle_path}")