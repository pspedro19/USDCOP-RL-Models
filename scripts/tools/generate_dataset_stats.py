import pandas as pd
import numpy as np
import os

def generate_stats():
    """
    Generates statistics for a dataset and saves them to a text file.

    This script reads a dataset from a CSV file, calculates descriptive
    statistics, and writes a summary report to a text file. The input and
    output file paths are determined by the AUDIT_DIR environment variable.
    """
    audit_dir = os.environ.get('AUDIT_DIR')
    if not audit_dir:
        print("Error: AUDIT_DIR environment variable is not set.")
        return

    dataset_path = os.path.join(audit_dir, 'dataset', 'RL_DS3_MACRO_CORE.csv')
    stats_path = os.path.join(audit_dir, 'dataset', 'DATASET_STATS.txt')

    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return

    # Generate statistics
    with open(stats_path, 'w') as f:
        f.write('=' * 80 + '\n')
        f.write('DATASET STATISTICS: RL_DS3_MACRO_CORE.csv\n')
        f.write('=' * 80 + '\n\n')

        f.write(f'Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n')
        f.write(f'Date Range: {df["timestamp"].min()} to {df["timestamp"].max()}\n')
        f.write(f'Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n')

        f.write('Columns:\n')
        for col in df.columns:
            dtype = df[col].dtype
            nulls = df[col].isna().sum()
            f.write(f'  - {col}: {dtype} (nulls: {nulls})\n')

        f.write('\n' + '=' * 80 + '\n')
        f.write('FEATURE STATISTICS (13 model features)\n')
        f.write('=' * 80 + '\n\n')

        features = ['log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'rsi_9', 'atr_pct', 'adx_14',
                    'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z', 'brent_change_1d',
                    'rate_spread', 'usdmxn_ret_1h']

        for feat in features:
            if feat in df.columns:
                s = df[feat]
                f.write(f'{feat}:\n')
                f.write(f'  mean={s.mean():.6f}, std={s.std():.6f}\n')
                f.write(f'  min={s.min():.6f}, max={s.max():.6f}\n')
                f.write(f'  nulls={s.isna().sum()}, zeros={(s==0).sum()}\n\n')

    print(f'Dataset statistics saved to {stats_path}')

if __name__ == "__main__":
    generate_stats()
