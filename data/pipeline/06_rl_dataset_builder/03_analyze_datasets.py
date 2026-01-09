"""
SCRIPT 04 - Analisis Estadistico de Datasets RL
================================================
Genera reporte detallado de cada columna en los 5 datasets:
- Nulls y porcentaje
- Outliers (IQR method)
- Estadisticas descriptivas
- Recomendaciones por columna
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# =============================================================================
# CONFIGURACION
# =============================================================================
BASE_PATH = Path(__file__).parent  # 06_rl_dataset_builder/
PIPELINE_PATH = BASE_PATH.parent  # pipeline/

# Input/Output (nueva estructura)
INPUT_PATH = PIPELINE_PATH / "07_output" / "datasets_5min"
OUTPUT_PATH = PIPELINE_PATH / "07_output" / "analysis"

# Fallback a ubicacion anterior
if not INPUT_PATH.exists():
    INPUT_PATH = PIPELINE_PATH / "04_rl_datasets" / "datasets_5min"
    OUTPUT_PATH = PIPELINE_PATH / "04_rl_datasets" / "analysis"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

DATASETS = [
    "RL_DS1_MINIMAL.csv",
    "RL_DS2_TECHNICAL_MTF.csv",
    "RL_DS3_MACRO_CORE.csv",
    "RL_DS4_COST_AWARE.csv",
    "RL_DS5_REGIME.csv",
    "RL_DS6_CARRY_TRADE.csv",
    "RL_DS7_COMMODITY_BASKET.csv",
    "RL_DS8_RISK_SENTIMENT.csv",
    "RL_DS9_FED_WATCH.csv",
    "RL_DS10_FLOWS_FUNDAMENTALS.csv",
]

def detect_outliers_iqr(series, k=1.5):
    """Detecta outliers usando metodo IQR"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    outliers = (series < lower) | (series > upper)
    return outliers.sum(), lower, upper

def detect_extreme_outliers(series, k=3.0):
    """Detecta outliers extremos (k=3)"""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    outliers = (series < lower) | (series > upper)
    return outliers.sum()

def analyze_column(series, col_name):
    """Analiza una columna y retorna diccionario de estadisticas"""
    total = len(series)

    # Nulls
    nulls = series.isna().sum()
    null_pct = (nulls / total) * 100

    # Non-null series for stats
    valid = series.dropna()
    valid_count = len(valid)

    if valid_count == 0:
        return {
            'column': col_name,
            'total_rows': total,
            'nulls': nulls,
            'null_pct': 100.0,
            'valid_count': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'p25': np.nan,
            'median': np.nan,
            'p75': np.nan,
            'max': np.nan,
            'outliers_iqr': 0,
            'outliers_extreme': 0,
            'inf_count': 0,
            'zeros': 0,
            'status': 'EMPTY'
        }

    # Infinitos
    inf_count = np.isinf(valid).sum() if np.issubdtype(valid.dtype, np.floating) else 0

    # Zeros
    zeros = (valid == 0).sum()
    zero_pct = (zeros / valid_count) * 100

    # Stats (exclude inf)
    clean = valid.replace([np.inf, -np.inf], np.nan).dropna()

    if len(clean) == 0:
        return {
            'column': col_name,
            'total_rows': total,
            'nulls': nulls,
            'null_pct': null_pct,
            'valid_count': valid_count,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'p25': np.nan,
            'median': np.nan,
            'p75': np.nan,
            'max': np.nan,
            'outliers_iqr': 0,
            'outliers_extreme': 0,
            'inf_count': inf_count,
            'zeros': zeros,
            'status': 'ALL_INF'
        }

    # Basic stats
    mean = clean.mean()
    std = clean.std()
    min_val = clean.min()
    max_val = clean.max()
    p25 = clean.quantile(0.25)
    median = clean.quantile(0.50)
    p75 = clean.quantile(0.75)

    # Outliers
    outliers_iqr, lower, upper = detect_outliers_iqr(clean)
    outliers_extreme = detect_extreme_outliers(clean)
    outlier_pct = (outliers_iqr / len(clean)) * 100

    # Status
    status = 'OK'
    issues = []

    if null_pct > 50:
        issues.append('HIGH_NULLS')
    elif null_pct > 20:
        issues.append('MODERATE_NULLS')

    if inf_count > 0:
        issues.append('HAS_INF')

    if outlier_pct > 20:
        issues.append('HIGH_OUTLIERS')
    elif outliers_extreme > 100:
        issues.append('EXTREME_OUTLIERS')

    if zero_pct > 95:
        issues.append('MOSTLY_ZEROS')

    if std == 0:
        issues.append('NO_VARIANCE')

    if issues:
        status = ' | '.join(issues)

    return {
        'column': col_name,
        'total_rows': total,
        'nulls': nulls,
        'null_pct': round(null_pct, 2),
        'valid_count': valid_count,
        'mean': round(mean, 6),
        'std': round(std, 6),
        'min': round(min_val, 6),
        'p25': round(p25, 6),
        'median': round(median, 6),
        'p75': round(p75, 6),
        'max': round(max_val, 6),
        'outliers_iqr': outliers_iqr,
        'outliers_pct': round(outlier_pct, 2),
        'outliers_extreme': outliers_extreme,
        'inf_count': inf_count,
        'zeros': zeros,
        'zeros_pct': round(zero_pct, 2),
        'status': status
    }

def analyze_dataset(filepath):
    """Analiza un dataset completo"""
    df = pd.read_csv(filepath)

    results = []
    for col in df.columns:
        if col == 'timestamp':
            continue

        # Check if numeric
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        stats = analyze_column(df[col], col)
        results.append(stats)

    return df, pd.DataFrame(results)

# =============================================================================
# MAIN
# =============================================================================
print("=" * 80)
print("ANALISIS ESTADISTICO DE DATASETS RL")
print("=" * 80)
print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

all_summaries = {}
all_recommendations = {}

for ds_name in DATASETS:
    filepath = INPUT_PATH / ds_name
    if not filepath.exists():
        print(f"[!] No existe: {ds_name}")
        continue

    print("-" * 80)
    print(f"DATASET: {ds_name}")
    print("-" * 80)

    df, stats_df = analyze_dataset(filepath)
    all_summaries[ds_name] = stats_df

    # Basic info
    print(f"  Filas: {len(df):,}")
    print(f"  Columnas: {len(df.columns)}")

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"  Rango: {df['timestamp'].min().date()} a {df['timestamp'].max().date()}")

    # Count by status
    ok_cols = stats_df[stats_df['status'] == 'OK']
    problem_cols = stats_df[stats_df['status'] != 'OK']

    print(f"\n  Columnas OK: {len(ok_cols)}")
    print(f"  Columnas con problemas: {len(problem_cols)}")

    # Show problems
    if len(problem_cols) > 0:
        print(f"\n  COLUMNAS PROBLEMATICAS:")
        for _, row in problem_cols.iterrows():
            print(f"    - {row['column']}: {row['status']}")
            if row['inf_count'] > 0:
                print(f"        Infinitos: {row['inf_count']:,}")
            if 'MOSTLY_ZEROS' in row['status']:
                print(f"        Zeros: {row['zeros']:,} ({row['zeros_pct']:.1f}%)")
            if row['null_pct'] > 20:
                print(f"        Nulls: {row['nulls']:,} ({row['null_pct']:.1f}%)")
            if row['outliers_extreme'] > 100:
                print(f"        Outliers extremos: {row['outliers_extreme']:,}")

    # Save detailed stats
    output_file = OUTPUT_PATH / f"STATS_{ds_name.replace('.csv', '')}.csv"
    stats_df.to_csv(output_file, index=False)
    print(f"\n  [OK] Guardado: {output_file.name}")

    # Generate recommendations
    recommendations = []
    for _, row in stats_df.iterrows():
        rec = {'column': row['column'], 'action': 'KEEP', 'reason': ''}

        if row['status'] == 'EMPTY' or row['status'] == 'ALL_INF':
            rec['action'] = 'REMOVE'
            rec['reason'] = row['status']
        elif 'MOSTLY_ZEROS' in row['status']:
            rec['action'] = 'REMOVE'
            rec['reason'] = f"Zeros: {row['zeros_pct']:.1f}%"
        elif row['inf_count'] > 0:
            rec['action'] = 'FIX_OR_REMOVE'
            rec['reason'] = f"Infinitos: {row['inf_count']}"
        elif 'HIGH_NULLS' in row['status']:
            rec['action'] = 'REVIEW'
            rec['reason'] = f"Nulls: {row['null_pct']:.1f}%"
        elif 'EXTREME_OUTLIERS' in row['status'] and row['outliers_extreme'] > 500:
            rec['action'] = 'REVIEW'
            rec['reason'] = f"Outliers extremos: {row['outliers_extreme']}"
        elif 'NO_VARIANCE' in row['status']:
            rec['action'] = 'REMOVE'
            rec['reason'] = 'Sin varianza'

        recommendations.append(rec)

    rec_df = pd.DataFrame(recommendations)
    all_recommendations[ds_name] = rec_df

    # Summary counts
    keep_count = len(rec_df[rec_df['action'] == 'KEEP'])
    remove_count = len(rec_df[rec_df['action'] == 'REMOVE'])
    fix_count = len(rec_df[rec_df['action'] == 'FIX_OR_REMOVE'])
    review_count = len(rec_df[rec_df['action'] == 'REVIEW'])

    print(f"\n  RECOMENDACIONES:")
    print(f"    KEEP: {keep_count}")
    print(f"    REMOVE: {remove_count}")
    print(f"    FIX_OR_REMOVE: {fix_count}")
    print(f"    REVIEW: {review_count}")

    if remove_count > 0 or fix_count > 0:
        print(f"\n    Columnas a remover/arreglar:")
        for _, row in rec_df[rec_df['action'].isin(['REMOVE', 'FIX_OR_REMOVE'])].iterrows():
            print(f"      - {row['column']}: {row['reason']}")

# =============================================================================
# REPORTE CONSOLIDADO
# =============================================================================
print("\n" + "=" * 80)
print("REPORTE CONSOLIDADO")
print("=" * 80)

# Create consolidated report
report_lines = []
report_lines.append("REPORTE DE ANALISIS - DATASETS RL")
report_lines.append("=" * 60)
report_lines.append(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
report_lines.append("")

for ds_name in DATASETS:
    if ds_name not in all_summaries:
        continue

    stats_df = all_summaries[ds_name]
    rec_df = all_recommendations[ds_name]

    report_lines.append("-" * 60)
    report_lines.append(f"DATASET: {ds_name}")
    report_lines.append("-" * 60)

    # Summary
    ok_cols = len(stats_df[stats_df['status'] == 'OK'])
    total_cols = len(stats_df)
    report_lines.append(f"Columnas totales: {total_cols}")
    report_lines.append(f"Columnas OK: {ok_cols} ({ok_cols/total_cols*100:.1f}%)")

    # Columns to keep
    keep_cols = rec_df[rec_df['action'] == 'KEEP']['column'].tolist()
    report_lines.append(f"\nColumnas recomendadas ({len(keep_cols)}):")
    for col in keep_cols:
        report_lines.append(f"  - {col}")

    # Columns to remove
    remove_cols = rec_df[rec_df['action'].isin(['REMOVE', 'FIX_OR_REMOVE'])]
    if len(remove_cols) > 0:
        report_lines.append(f"\nColumnas a remover ({len(remove_cols)}):")
        for _, row in remove_cols.iterrows():
            report_lines.append(f"  - {row['column']}: {row['reason']}")

    report_lines.append("")

# Comparative table
report_lines.append("=" * 60)
report_lines.append("TABLA COMPARATIVA")
report_lines.append("=" * 60)
report_lines.append(f"{'Dataset':<30} {'Total':<8} {'OK':<8} {'Remove':<8} {'Score':<8}")
report_lines.append("-" * 60)

for ds_name in DATASETS:
    if ds_name not in all_summaries:
        continue

    stats_df = all_summaries[ds_name]
    rec_df = all_recommendations[ds_name]

    total = len(stats_df)
    ok = len(rec_df[rec_df['action'] == 'KEEP'])
    remove = len(rec_df[rec_df['action'].isin(['REMOVE', 'FIX_OR_REMOVE'])])
    score = ok / total * 100

    report_lines.append(f"{ds_name:<30} {total:<8} {ok:<8} {remove:<8} {score:.1f}%")

# Save report
report_path = OUTPUT_PATH / "ANALYSIS_REPORT.txt"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\nReporte guardado: {report_path}")

# Print final summary
print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)

print(f"\n{'Dataset':<35} {'Cols':<8} {'OK':<8} {'Problems':<10} {'Score':<8}")
print("-" * 70)

for ds_name in DATASETS:
    if ds_name not in all_summaries:
        continue

    stats_df = all_summaries[ds_name]
    rec_df = all_recommendations[ds_name]

    total = len(stats_df)
    ok = len(rec_df[rec_df['action'] == 'KEEP'])
    problems = len(rec_df[rec_df['action'] != 'KEEP'])
    score = ok / total * 100

    print(f"{ds_name:<35} {total:<8} {ok:<8} {problems:<10} {score:.1f}%")

print("\nArchivos generados en ANALYSIS/:")
for f in sorted(OUTPUT_PATH.iterdir()):
    print(f"  - {f.name}")
