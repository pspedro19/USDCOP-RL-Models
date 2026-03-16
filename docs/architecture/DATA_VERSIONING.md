# Data Versioning with DVC

## Overview

This document describes the data versioning strategy for the USDCOP trading system using DVC (Data Version Control).

## Why DVC?

- **Reproducibility**: Recreate any dataset version used for training
- **Collaboration**: Share large datasets without Git limitations
- **Lineage**: Track data transformations and model dependencies
- **Integrity**: Verify data hasn't been corrupted or modified

## Quick Start

```bash
# Install DVC
pip install "dvc[s3]"

# Initialize (already done)
dvc init

# Pull latest data
dvc pull

# Check status
dvc status
```

## Tracked Files

### Critical Files (Always Versioned)

| File | Description | Impact if Wrong |
|------|-------------|-----------------|
| `config/norm_stats.json` | Normalization statistics | Model produces wrong predictions |
| `data/training/*.parquet` | Training datasets | Model trained on different data |
| `models/ppo_primary/model.onnx` | Inference model | Different model deployed |

### Pipeline Outputs

| Directory | Description |
|-----------|-------------|
| `data/pipeline/07_output/` | Final datasets for training |
| `models/ppo_primary/` | Trained model artifacts |
| `results/backtest/` | Backtest results and metrics |

## Commands Reference

### Daily Operations

```bash
# Check what's changed
dvc status

# Pull latest data (e.g., on new machine)
dvc pull

# Push new data after changes
dvc push

# Show differences between versions
dvc diff HEAD~1
```

### Pipeline Operations

```bash
# Run full pipeline
dvc repro

# Run specific stage
dvc repro train_model

# Show pipeline DAG
dvc dag

# Show stage dependencies
dvc dag --md
```

### Version Management

```bash
# List tracked files
dvc list .

# Get specific version
git checkout v1.0.0
dvc checkout

# Compare metrics between versions
dvc metrics diff HEAD~5
```

## Remote Storage Configuration

### Local (Default)

```bash
# Already configured during setup
dvc remote add -d myremote .dvc/remote-storage
```

### AWS S3

```bash
dvc remote modify myremote url s3://your-bucket/dvc-store
dvc remote modify myremote region us-east-1

# Optional: Use AWS profiles
dvc remote modify myremote profile myprofile
```

### Google Cloud Storage

```bash
dvc remote modify myremote url gs://your-bucket/dvc-store
```

### Azure Blob Storage

```bash
dvc remote modify myremote url azure://container/path
```

## Pipeline Stages

```
prepare_data → calculate_norm_stats → train_model → export_onnx → backtest
```

### Stage Details

1. **prepare_data**: Builds 5-minute datasets from raw data
2. **calculate_norm_stats**: Computes normalization statistics
3. **train_model**: Trains PPO model
4. **export_onnx**: Exports to ONNX format
5. **backtest**: Validates on historical data

## Best Practices

### Before Training

```bash
# Always pull latest data
dvc pull

# Verify data integrity
dvc status

# Check norm_stats matches expected
cat config/norm_stats.json | jq '.rsi_9'
```

### After Training

```bash
# Add new model to DVC
dvc add models/ppo_primary/

# Commit both git and DVC changes
git add models/ppo_primary.dvc
git commit -m "Train model v2.1: improved Sharpe"

# Push data to remote
dvc push
```

### Deployment

```bash
# On deployment server
git pull
dvc pull

# Verify model exists
ls -la models/ppo_primary/model.onnx
```

## Data Integrity Verification

```bash
# Check MD5 hashes
dvc status --show-json | jq '.[]'

# Verify specific file
dvc check config/norm_stats.json
```

## Troubleshooting

### "File not found" after pull

```bash
# Force checkout
dvc checkout --force

# Or re-pull
dvc pull --force
```

### "Remote storage not accessible"

```bash
# Check remote configuration
dvc remote list -v

# Test connection
dvc push --dry-run
```

### "Hash mismatch"

```bash
# File was modified locally
# Option 1: Discard local changes
dvc checkout config/norm_stats.json

# Option 2: Update DVC tracking
dvc add config/norm_stats.json
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
jobs:
  train:
    steps:
      - uses: actions/checkout@v3

      - name: Setup DVC
        uses: iterative/setup-dvc@v1

      - name: Pull data
        run: dvc pull
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}

      - name: Run pipeline
        run: dvc repro
```

## Model Versioning Convention

```
models/ppo_primary/
├── v2.0.0/           # Major version (breaking changes)
│   ├── model.onnx
│   ├── norm_stats.json (snapshot)
│   └── metrics.json
├── v2.1.0/           # Minor version (new features)
└── v2.1.1/           # Patch version (bug fixes)
```

## Related Documentation

- [DVC Documentation](https://dvc.org/doc)
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [SLA.md](./SLA.md) - Service level agreements
