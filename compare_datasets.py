import pandas as pd
import numpy as np

# Load both datasets
archive_path = r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models - copia\data\archive\PASS\OUTPUT_RL\RL_DS3_MACRO_CORE.csv"
pipeline_path = r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\data\pipeline\07_output\datasets_5min\RL_DS3_MACRO_CORE.csv"

print("Loading datasets...")
archive_df = pd.read_csv(archive_path)
pipeline_df = pd.read_csv(pipeline_path)

print("\n" + "="*80)
print("1. COLUMN NAMES AND COUNTS COMPARISON")
print("="*80)

archive_cols = set(archive_df.columns)
pipeline_cols = set(pipeline_df.columns)

print(f"\nArchive columns: {len(archive_cols)}")
print(f"Pipeline columns: {len(pipeline_cols)}")
print(f"Identical columns: {archive_cols == pipeline_cols}")

if archive_cols != pipeline_cols:
    print(f"\nColumns only in archive: {archive_cols - pipeline_cols}")
    print(f"Columns only in pipeline: {pipeline_cols - archive_cols}")

print(f"\nColumn order identical: {list(archive_df.columns) == list(pipeline_df.columns)}")

print("\n" + "="*80)
print("2. DATE RANGES COMPARISON")
print("="*80)

archive_df['timestamp'] = pd.to_datetime(archive_df['timestamp'])
pipeline_df['timestamp'] = pd.to_datetime(pipeline_df['timestamp'])

print(f"\nArchive dataset:")
print(f"  Start date: {archive_df['timestamp'].min()}")
print(f"  End date:   {archive_df['timestamp'].max()}")
print(f"  Total rows: {len(archive_df)}")

print(f"\nPipeline dataset:")
print(f"  Start date: {pipeline_df['timestamp'].min()}")
print(f"  End date:   {pipeline_df['timestamp'].max()}")
print(f"  Total rows: {len(pipeline_df)}")

print("\n" + "="*80)
print("3. OVERLAPPING DATES AND ROW DIFFERENCES")
print("="*80)

archive_ts = set(archive_df['timestamp'])
pipeline_ts = set(pipeline_df['timestamp'])
overlap_ts = archive_ts & pipeline_ts

print(f"\nRows only in archive: {len(archive_ts - pipeline_ts)}")
print(f"Rows only in pipeline: {len(pipeline_ts - archive_ts)}")
print(f"Overlapping rows: {len(overlap_ts)}")

# Show sample of non-overlapping timestamps
if len(archive_ts - pipeline_ts) > 0:
    print("\nFirst 10 timestamps only in archive:")
    archive_only = sorted(list(archive_ts - pipeline_ts))[:10]
    for ts in archive_only:
        print(f"  {ts}")

if len(pipeline_ts - archive_ts) > 0:
    print("\nFirst 10 timestamps only in pipeline:")
    pipeline_only = sorted(list(pipeline_ts - archive_ts))[:10]
    for ts in pipeline_only:
        print(f"  {ts}")

print("\n" + "="*80)
print("4. FEATURE VALUE COMPARISON FOR OVERLAPPING DATES")
print("="*80)

# Merge on timestamp for comparison
merged_df = archive_df.merge(pipeline_df, on='timestamp', suffixes=('_archive', '_pipeline'), how='inner')
print(f"\nMerged {len(merged_df)} overlapping rows for comparison")

# Key features to compare
key_features = [
    'dxy_z', 'vix_z', 'embi_z', 'rate_spread',
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
    'rsi_9', 'atr_pct', 'adx_14',
    'usdmxn_ret_1h'
]

print("\nDifferences found in overlapping rows:")
print("-" * 80)

differences_found = {}
for feature in key_features:
    archive_col = f'{feature}_archive'
    pipeline_col = f'{feature}_pipeline'

    if archive_col in merged_df.columns and pipeline_col in merged_df.columns:
        # Compare values (accounting for floating point precision)
        diff_mask = ~np.isclose(merged_df[archive_col], merged_df[pipeline_col], rtol=1e-5, atol=1e-8)
        num_diffs = diff_mask.sum()

        if num_diffs > 0:
            differences_found[feature] = num_diffs
            pct_diff = (num_diffs / len(merged_df)) * 100

            print(f"\n{feature}:")
            print(f"  Rows with different values: {num_diffs} ({pct_diff:.2f}%)")

            # Show statistics of differences
            archive_vals = merged_df.loc[diff_mask, archive_col]
            pipeline_vals = merged_df.loc[diff_mask, pipeline_col]

            abs_diffs = np.abs(archive_vals - pipeline_vals)
            print(f"  Mean absolute difference: {abs_diffs.mean():.6f}")
            print(f"  Max absolute difference: {abs_diffs.max():.6f}")

            # Show first 5 examples
            print(f"\n  First 5 examples of differences:")
            for i, idx in enumerate(merged_df[diff_mask].head(5).index):
                ts = merged_df.loc[idx, 'timestamp']
                arch_val = merged_df.loc[idx, archive_col]
                pipe_val = merged_df.loc[idx, pipeline_col]
                print(f"    {ts}: Archive={arch_val:.6f}, Pipeline={pipe_val:.6f}, Diff={arch_val-pipe_val:.6f}")

if not differences_found:
    print("\nNo significant differences found in key features for overlapping timestamps!")

print("\n" + "="*80)
print("5. STATISTICAL COMPARISON (FULL DATASETS)")
print("="*80)

print("\nStatistical comparison for each feature:")
print("-" * 80)

for feature in key_features:
    if feature in archive_df.columns and feature in pipeline_df.columns:
        print(f"\n{feature}:")
        print(f"{'Metric':<15} {'Archive':<20} {'Pipeline':<20} {'Difference':<20}")
        print("-" * 75)

        arch_mean = archive_df[feature].mean()
        pipe_mean = pipeline_df[feature].mean()
        print(f"{'Mean':<15} {arch_mean:<20.6f} {pipe_mean:<20.6f} {arch_mean-pipe_mean:<20.6f}")

        arch_std = archive_df[feature].std()
        pipe_std = pipeline_df[feature].std()
        print(f"{'Std Dev':<15} {arch_std:<20.6f} {pipe_std:<20.6f} {arch_std-pipe_std:<20.6f}")

        arch_min = archive_df[feature].min()
        pipe_min = pipeline_df[feature].min()
        print(f"{'Min':<15} {arch_min:<20.6f} {pipe_min:<20.6f} {arch_min-pipe_min:<20.6f}")

        arch_max = archive_df[feature].max()
        pipe_max = pipeline_df[feature].max()
        print(f"{'Max':<15} {arch_max:<20.6f} {pipe_max:<20.6f} {arch_max-pipe_max:<20.6f}")

        arch_med = archive_df[feature].median()
        pipe_med = pipeline_df[feature].median()
        print(f"{'Median':<15} {arch_med:<20.6f} {pipe_med:<20.6f} {arch_med-pipe_med:<20.6f}")

print("\n" + "="*80)
print("6. SUMMARY OF FINDINGS")
print("="*80)

print(f"\nDataset row counts:")
print(f"  Archive: {len(archive_df)} rows")
print(f"  Pipeline: {len(pipeline_df)} rows")
print(f"  Difference: {len(archive_df) - len(pipeline_df)} rows")

print(f"\nOverlapping data:")
print(f"  Common timestamps: {len(overlap_ts)}")
print(f"  Features with differences: {len(differences_found)}")

if differences_found:
    print(f"\nFeatures with value differences:")
    for feature, count in differences_found.items():
        pct = (count / len(merged_df)) * 100
        print(f"  {feature}: {count} rows ({pct:.2f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
