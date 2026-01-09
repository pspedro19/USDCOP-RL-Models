import pandas as pd
import numpy as np

# Load both datasets
archive_path = r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models - copia\data\archive\PASS\OUTPUT_RL\RL_DS3_MACRO_CORE.csv"
pipeline_path = r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models\data\pipeline\07_output\datasets_5min\RL_DS3_MACRO_CORE.csv"

archive_df = pd.read_csv(archive_path, parse_dates=['timestamp'])
pipeline_df = pd.read_csv(pipeline_path, parse_dates=['timestamp'])

print("="*80)
print("DETAILED DIFFERENCE ANALYSIS")
print("="*80)

# Analyze the capping pattern in archive
print("\n1. ARCHIVE DATASET - CAPPING ANALYSIS")
print("-" * 80)

macro_features = ['dxy_z', 'vix_z', 'embi_z', 'rate_spread']

for feature in macro_features:
    capped_low = (archive_df[feature] == -4.0).sum()
    capped_high = (archive_df[feature] == 4.0).sum()
    total = len(archive_df)

    pct_low = (capped_low / total) * 100
    pct_high = (capped_high / total) * 100
    pct_capped = ((capped_low + capped_high) / total) * 100

    print(f"\n{feature}:")
    print(f"  Values at -4.0: {capped_low:,} ({pct_low:.2f}%)")
    print(f"  Values at +4.0: {capped_high:,} ({pct_high:.2f}%)")
    print(f"  Total capped:   {capped_low + capped_high:,} ({pct_capped:.2f}%)")

# Analyze the distribution in pipeline
print("\n\n2. PIPELINE DATASET - DISTRIBUTION ANALYSIS")
print("-" * 80)

for feature in macro_features:
    beyond_4 = (np.abs(pipeline_df[feature]) > 4.0).sum()
    beyond_3 = (np.abs(pipeline_df[feature]) > 3.0).sum()
    beyond_2 = (np.abs(pipeline_df[feature]) > 2.0).sum()
    total = len(pipeline_df)

    print(f"\n{feature}:")
    print(f"  |z| > 2.0: {beyond_2:,} ({(beyond_2/total)*100:.2f}%)")
    print(f"  |z| > 3.0: {beyond_3:,} ({(beyond_3/total)*100:.2f}%)")
    print(f"  |z| > 4.0: {beyond_4:,} ({(beyond_4/total)*100:.2f}%)")
    print(f"  Min: {pipeline_df[feature].min():.4f}")
    print(f"  Max: {pipeline_df[feature].max():.4f}")

# Find specific high-impact difference examples
print("\n\n3. TOP 10 LARGEST ABSOLUTE DIFFERENCES")
print("-" * 80)

merged_df = archive_df.merge(pipeline_df, on='timestamp', suffixes=('_arch', '_pipe'))

for feature in macro_features:
    arch_col = f'{feature}_arch'
    pipe_col = f'{feature}_pipe'

    merged_df[f'{feature}_diff'] = np.abs(merged_df[arch_col] - merged_df[pipe_col])

    print(f"\n{feature}:")
    top_10 = merged_df.nlargest(10, f'{feature}_diff')[['timestamp', arch_col, pipe_col, f'{feature}_diff']]

    for idx, row in top_10.iterrows():
        print(f"  {row['timestamp']}: Archive={row[arch_col]:7.3f}, Pipeline={row[pipe_col]:7.3f}, Diff={row[f'{feature}_diff']:7.3f}")

# Correlation analysis
print("\n\n4. CORRELATION BETWEEN ARCHIVE AND PIPELINE VALUES")
print("-" * 80)

for feature in macro_features:
    arch_col = f'{feature}_arch'
    pipe_col = f'{feature}_pipe'

    correlation = merged_df[arch_col].corr(merged_df[pipe_col])

    print(f"\n{feature}:")
    print(f"  Correlation: {correlation:.6f}")

    # Calculate correlation excluding capped values in archive
    uncapped_mask = (merged_df[arch_col] > -4.0) & (merged_df[arch_col] < 4.0)
    if uncapped_mask.sum() > 0:
        correlation_uncapped = merged_df.loc[uncapped_mask, arch_col].corr(
            merged_df.loc[uncapped_mask, pipe_col]
        )
        print(f"  Correlation (excluding capped): {correlation_uncapped:.6f}")
        print(f"  Uncapped rows: {uncapped_mask.sum():,} ({(uncapped_mask.sum()/len(merged_df))*100:.2f}%)")

# Time-based analysis
print("\n\n5. DIFFERENCES OVER TIME")
print("-" * 80)

merged_df['year'] = merged_df['timestamp'].dt.year

print("\nAverage absolute differences by year:")
print(f"{'Year':<8} {'dxy_z':<12} {'vix_z':<12} {'embi_z':<12} {'rate_spread':<12}")
print("-" * 60)

for year in sorted(merged_df['year'].unique()):
    year_data = merged_df[merged_df['year'] == year]

    dxy_diff = year_data['dxy_z_diff'].mean()
    vix_diff = year_data['vix_z_diff'].mean()
    embi_diff = year_data['embi_z_diff'].mean()
    rate_diff = year_data['rate_spread_diff'].mean()

    print(f"{year:<8} {dxy_diff:<12.4f} {vix_diff:<12.4f} {embi_diff:<12.4f} {rate_diff:<12.4f}")

# Identify periods of maximum divergence
print("\n\n6. PERIODS OF MAXIMUM DIVERGENCE")
print("-" * 80)

merged_df['total_diff'] = (
    merged_df['dxy_z_diff'] +
    merged_df['vix_z_diff'] +
    merged_df['embi_z_diff'] +
    merged_df['rate_spread_diff']
)

print("\nTop 20 timestamps with largest combined differences:")
top_divergence = merged_df.nlargest(20, 'total_diff')[
    ['timestamp', 'dxy_z_diff', 'vix_z_diff', 'embi_z_diff', 'rate_spread_diff', 'total_diff']
]

for idx, row in top_divergence.iterrows():
    print(f"\n  {row['timestamp']}")
    print(f"    dxy_z: {row['dxy_z_diff']:.3f}, vix_z: {row['vix_z_diff']:.3f}, " +
          f"embi_z: {row['embi_z_diff']:.3f}, rate_spread: {row['rate_spread_diff']:.3f}")
    print(f"    TOTAL: {row['total_diff']:.3f}")

# USDMXN specific analysis
print("\n\n7. USDMXN_RET_1H SPECIFIC ANALYSIS")
print("-" * 80)

usdmxn_diff_mask = ~np.isclose(
    merged_df['usdmxn_ret_1h_arch'],
    merged_df['usdmxn_ret_1h_pipe'],
    rtol=1e-5, atol=1e-8
)

print(f"\nTotal rows with different usdmxn_ret_1h: {usdmxn_diff_mask.sum()}")

if usdmxn_diff_mask.sum() > 0:
    usdmxn_diff_data = merged_df[usdmxn_diff_mask].copy()
    usdmxn_diff_data['usdmxn_diff'] = np.abs(
        usdmxn_diff_data['usdmxn_ret_1h_arch'] - usdmxn_diff_data['usdmxn_ret_1h_pipe']
    )

    print(f"\nDate range of differences:")
    print(f"  First: {usdmxn_diff_data['timestamp'].min()}")
    print(f"  Last:  {usdmxn_diff_data['timestamp'].max()}")

    print(f"\nDifference statistics:")
    print(f"  Mean abs diff: {usdmxn_diff_data['usdmxn_diff'].mean():.6f}")
    print(f"  Max abs diff:  {usdmxn_diff_data['usdmxn_diff'].max():.6f}")

    print(f"\nTop 10 largest USDMXN differences:")
    top_usdmxn = usdmxn_diff_data.nlargest(10, 'usdmxn_diff')[
        ['timestamp', 'usdmxn_ret_1h_arch', 'usdmxn_ret_1h_pipe', 'usdmxn_diff']
    ]

    for idx, row in top_usdmxn.iterrows():
        print(f"  {row['timestamp']}: Archive={row['usdmxn_ret_1h_arch']:9.6f}, " +
              f"Pipeline={row['usdmxn_ret_1h_pipe']:9.6f}, Diff={row['usdmxn_diff']:9.6f}")

    # Check if differences cluster by date
    usdmxn_diff_data['date'] = usdmxn_diff_data['timestamp'].dt.date
    date_counts = usdmxn_diff_data['date'].value_counts().head(10)

    print(f"\nTop 10 dates with most USDMXN differences:")
    for date, count in date_counts.items():
        print(f"  {date}: {count} rows")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
