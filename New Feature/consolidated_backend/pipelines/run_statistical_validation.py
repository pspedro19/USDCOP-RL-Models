# pipeline_limpio_regresion/run_statistical_validation.py
"""
STATISTICAL VALIDATION PIPELINE - Scientific Rigor Tests

Implements 4 additional tests recommended by doctoral experts:

1. WALK-FORWARD VALIDATION
   - Multiple train/test splits over time
   - Avoids single-split bias
   - Provides distribution of DA estimates

2. PESARAN-TIMMERMANN TEST
   - Tests if DA is significantly > 50%
   - H0: No predictive ability (DA = 50%)
   - H1: Predictive ability exists (DA > 50%)

3. DIEBOLD-MARIANO TEST
   - Tests if one model is significantly better than another
   - Compares forecast errors between models
   - Accounts for autocorrelation in errors

4. BOOTSTRAP CONFIDENCE INTERVALS
   - Non-parametric uncertainty quantification
   - 95% CI for DA estimates
   - Robust to distributional assumptions

Author: Pipeline USD/COP ML
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import json
import warnings
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.linear_model import Ridge, BayesianRidge, ARDRegression
from sklearn.preprocessing import StandardScaler

from src.features.common import prepare_features, create_targets

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

HORIZONS = [1, 5, 10, 15, 20, 25, 30]
RANDOM_STATE = 42

# Walk-Forward parameters
N_SPLITS = 5  # Number of walk-forward splits
MIN_TRAIN_SIZE = 500  # Minimum training samples

# Bootstrap parameters
N_BOOTSTRAP = 1000  # Number of bootstrap samples
CONFIDENCE_LEVEL = 0.95

BASE_DIR = Path(__file__).parent
DATA_PATH = Path(r"C:\Users\pedro\OneDrive\Documents\data\RL_COMBINED_ML_FEATURES_FIXED.csv")
OUTPUT_DIR = BASE_DIR / "results" / "statistical_validation" / datetime.now().strftime('%Y%m%d_%H%M%S')


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

def pesaran_timmermann_test(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Pesaran-Timmermann Test for Directional Accuracy.

    Tests whether the proportion of correct sign predictions is
    significantly different from what would be expected by chance.

    H0: No predictive ability (signs are independent)
    H1: Predictive ability exists

    Reference: Pesaran & Timmermann (1992) "A Simple Nonparametric Test
               of Predictive Performance", Journal of Business & Economic Statistics

    Returns:
        dict with test statistic, p-value, and interpretation
    """
    n = len(y_true)

    # Actual signs
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)

    # Proportion of correct predictions
    P_hat = np.mean(sign_true == sign_pred)

    # Proportion of positive actual outcomes
    P_y = np.mean(sign_true > 0)

    # Proportion of positive predictions
    P_x = np.mean(sign_pred > 0)

    # Expected proportion under independence (H0)
    P_star = P_y * P_x + (1 - P_y) * (1 - P_x)

    # Variance under H0
    var_P_star = (1/n) * (
        (2 * P_y - 1)**2 * P_x * (1 - P_x) +
        (2 * P_x - 1)**2 * P_y * (1 - P_y) +
        4 * P_y * P_x * (1 - P_y) * (1 - P_x)
    )

    # Variance of P_hat
    var_P_hat = (1/n) * P_star * (1 - P_star)

    # Total variance
    var_total = var_P_hat + var_P_star

    # PT statistic (asymptotically N(0,1) under H0)
    if var_total > 0:
        PT_stat = (P_hat - P_star) / np.sqrt(var_total)
    else:
        PT_stat = 0

    # P-value (one-sided, testing if DA > random)
    p_value = 1 - stats.norm.cdf(PT_stat)

    # Interpretation
    if p_value < 0.01:
        significance = "***"
        interpretation = "Highly significant (p < 0.01)"
    elif p_value < 0.05:
        significance = "**"
        interpretation = "Significant (p < 0.05)"
    elif p_value < 0.10:
        significance = "*"
        interpretation = "Marginally significant (p < 0.10)"
    else:
        significance = ""
        interpretation = "Not significant (p >= 0.10)"

    return {
        'test_name': 'Pesaran-Timmermann',
        'statistic': PT_stat,
        'p_value': p_value,
        'DA_observed': P_hat * 100,
        'DA_expected_H0': P_star * 100,
        'significance': significance,
        'interpretation': interpretation,
        'reject_H0': p_value < 0.05
    }


def diebold_mariano_test(errors1: np.ndarray, errors2: np.ndarray,
                          h: int = 1, power: int = 2) -> Dict:
    """
    Diebold-Mariano Test for Comparing Forecast Accuracy.

    Tests whether two forecasts have significantly different accuracy.

    H0: Equal predictive accuracy
    H1: Different predictive accuracy

    Reference: Diebold & Mariano (1995) "Comparing Predictive Accuracy",
               Journal of Business & Economic Statistics

    Args:
        errors1: Forecast errors from model 1
        errors2: Forecast errors from model 2
        h: Forecast horizon (for HAC variance)
        power: 1 for MAE, 2 for MSE

    Returns:
        dict with test statistic, p-value, and interpretation
    """
    # Loss differential
    d = np.abs(errors1)**power - np.abs(errors2)**power

    n = len(d)
    mean_d = np.mean(d)

    # HAC variance (Newey-West with h-1 lags)
    gamma_0 = np.var(d, ddof=1)

    # Autocovariances
    gamma_sum = 0
    for k in range(1, h):
        gamma_k = np.mean((d[k:] - mean_d) * (d[:-k] - mean_d))
        weight = 1 - k / h  # Bartlett kernel
        gamma_sum += 2 * weight * gamma_k

    var_d = (gamma_0 + gamma_sum) / n

    # DM statistic
    if var_d > 0:
        DM_stat = mean_d / np.sqrt(var_d)
    else:
        DM_stat = 0

    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(DM_stat)))

    # Interpretation
    if p_value < 0.01:
        significance = "***"
    elif p_value < 0.05:
        significance = "**"
    elif p_value < 0.10:
        significance = "*"
    else:
        significance = ""

    # Which model is better?
    if mean_d < 0:
        better_model = "Model 1"
    elif mean_d > 0:
        better_model = "Model 2"
    else:
        better_model = "Equal"

    return {
        'test_name': 'Diebold-Mariano',
        'statistic': DM_stat,
        'p_value': p_value,
        'mean_loss_diff': mean_d,
        'better_model': better_model,
        'significance': significance,
        'reject_H0': p_value < 0.05
    }


def bootstrap_confidence_interval(y_true: np.ndarray, y_pred: np.ndarray,
                                   n_bootstrap: int = 1000,
                                   confidence: float = 0.95) -> Dict:
    """
    Bootstrap Confidence Interval for Direction Accuracy.

    Non-parametric method to estimate uncertainty in DA.

    Args:
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        dict with point estimate, CI, and standard error
    """
    n = len(y_true)

    # Point estimate
    da_point = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100

    # Bootstrap resampling
    np.random.seed(RANDOM_STATE)
    bootstrap_das = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(n, size=n, replace=True)
        y_true_boot = y_true[idx]
        y_pred_boot = y_pred[idx]

        # Calculate DA for this bootstrap sample
        da_boot = np.mean(np.sign(y_pred_boot) == np.sign(y_true_boot)) * 100
        bootstrap_das.append(da_boot)

    bootstrap_das = np.array(bootstrap_das)

    # Confidence interval (percentile method)
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_das, alpha/2 * 100)
    ci_upper = np.percentile(bootstrap_das, (1 - alpha/2) * 100)

    # Standard error
    se = np.std(bootstrap_das)

    # Bias
    bias = np.mean(bootstrap_das) - da_point

    return {
        'point_estimate': da_point,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'se': se,
        'bias': bias,
        'confidence_level': confidence,
        'n_bootstrap': n_bootstrap,
        'ci_contains_50': ci_lower <= 50 <= ci_upper
    }


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validation(X: np.ndarray, y: np.ndarray,
                            model_class, model_params: Dict,
                            n_splits: int = 5,
                            min_train_size: int = 500,
                            gap: int = 30,
                            use_scaler: bool = True) -> Dict:
    """
    Walk-Forward Validation (Expanding Window).

    Trains on expanding window, tests on next block.
    More robust than single train/test split.

    Args:
        X: Features
        y: Target
        model_class: Model class to instantiate
        model_params: Parameters for model
        n_splits: Number of train/test splits
        min_train_size: Minimum training samples
        gap: Gap between train and test (prevent leakage)
        use_scaler: Whether to scale features

    Returns:
        dict with DA per split, average, std, and predictions
    """
    n = len(X)

    # Calculate split points
    test_size = (n - min_train_size - gap) // n_splits

    results = {
        'das': [],
        'var_ratios': [],
        'n_train': [],
        'n_test': [],
        'all_predictions': [],
        'all_actuals': []
    }

    for split in range(n_splits):
        # Expanding window
        train_end = min_train_size + split * test_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, n)

        if test_end <= test_start:
            continue

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        # Scale if needed
        if use_scaler:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Train model
        model = model_class(**model_params)
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        da = np.mean(np.sign(y_pred) == np.sign(y_test)) * 100
        var_ratio = np.var(y_pred) / (np.var(y_test) + 1e-8)

        results['das'].append(da)
        results['var_ratios'].append(var_ratio)
        results['n_train'].append(len(X_train))
        results['n_test'].append(len(X_test))
        results['all_predictions'].extend(y_pred.tolist())
        results['all_actuals'].extend(y_test.tolist())

    # Aggregate statistics
    results['da_mean'] = np.mean(results['das'])
    results['da_std'] = np.std(results['das'])
    results['da_min'] = np.min(results['das'])
    results['da_max'] = np.max(results['das'])
    results['var_ratio_mean'] = np.mean(results['var_ratios'])
    results['n_splits_actual'] = len(results['das'])

    return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """Load and prepare data."""
    print(f"Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


# =============================================================================
# MAIN VALIDATION
# =============================================================================

def run_validation(df: pd.DataFrame, feature_cols: List[str],
                   targets: Dict[int, pd.Series]):
    """Run complete statistical validation."""

    print("\n" + "="*130)
    print("STATISTICAL VALIDATION - SCIENTIFIC RIGOR TESTS")
    print("="*130)

    all_results = {}

    # Models to validate
    MODELS = {
        'ridge': (Ridge, {'alpha': 10.0, 'random_state': RANDOM_STATE}),
        'bayesian_ridge': (BayesianRidge, {'max_iter': 300}),
        'ard': (ARDRegression, {'max_iter': 500, 'tol': 1e-4})
    }

    for h in HORIZONS:
        print(f"\n{'='*80}")
        print(f"HORIZON {h} DAYS")
        print(f"{'='*80}")

        y = targets[h]

        # Prepare features
        X_df = df[feature_cols].copy()
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.ffill().bfill().fillna(0)

        valid_idx = ~y.isna()
        X_all = X_df.values[valid_idx]
        y_all = y.values[valid_idx]
        X_all = np.nan_to_num(X_all, nan=0.0)

        all_results[h] = {}

        for model_name, (model_class, model_params) in MODELS.items():
            print(f"\n--- {model_name.upper()} ---")

            # =================================================================
            # 1. WALK-FORWARD VALIDATION
            # =================================================================
            print("\n1. Walk-Forward Validation:")
            wf_results = walk_forward_validation(
                X_all, y_all,
                model_class, model_params,
                n_splits=N_SPLITS,
                min_train_size=MIN_TRAIN_SIZE,
                gap=max(HORIZONS),
                use_scaler=True
            )

            print(f"   Splits: {wf_results['n_splits_actual']}")
            print(f"   DA per split: {[f'{d:.1f}%' for d in wf_results['das']]}")
            print(f"   DA Mean: {wf_results['da_mean']:.2f}% ± {wf_results['da_std']:.2f}%")
            print(f"   DA Range: [{wf_results['da_min']:.1f}%, {wf_results['da_max']:.1f}%]")

            # =================================================================
            # 2. PESARAN-TIMMERMANN TEST
            # =================================================================
            print("\n2. Pesaran-Timmermann Test (H0: DA = 50%):")

            if len(wf_results['all_predictions']) > 0:
                pt_results = pesaran_timmermann_test(
                    np.array(wf_results['all_actuals']),
                    np.array(wf_results['all_predictions'])
                )

                print(f"   Observed DA: {pt_results['DA_observed']:.2f}%")
                print(f"   Expected DA (H0): {pt_results['DA_expected_H0']:.2f}%")
                print(f"   PT Statistic: {pt_results['statistic']:.3f}")
                print(f"   P-value: {pt_results['p_value']:.4f} {pt_results['significance']}")
                print(f"   Conclusion: {pt_results['interpretation']}")
            else:
                pt_results = None

            # =================================================================
            # 3. BOOTSTRAP CONFIDENCE INTERVALS
            # =================================================================
            print("\n3. Bootstrap Confidence Intervals:")

            if len(wf_results['all_predictions']) > 0:
                boot_results = bootstrap_confidence_interval(
                    np.array(wf_results['all_actuals']),
                    np.array(wf_results['all_predictions']),
                    n_bootstrap=N_BOOTSTRAP,
                    confidence=CONFIDENCE_LEVEL
                )

                print(f"   Point Estimate: {boot_results['point_estimate']:.2f}%")
                print(f"   95% CI: [{boot_results['ci_lower']:.2f}%, {boot_results['ci_upper']:.2f}%]")
                print(f"   Standard Error: {boot_results['se']:.2f}%")
                print(f"   CI contains 50%: {'Yes (!)' if boot_results['ci_contains_50'] else 'No (GOOD)'}")
            else:
                boot_results = None

            # Store results
            all_results[h][model_name] = {
                'walk_forward': wf_results,
                'pesaran_timmermann': pt_results,
                'bootstrap': boot_results
            }

    # =========================================================================
    # 4. DIEBOLD-MARIANO TEST (Compare models)
    # =========================================================================
    print("\n" + "="*130)
    print("DIEBOLD-MARIANO TEST - MODEL COMPARISON")
    print("="*130)

    dm_results = {}

    for h in HORIZONS:
        print(f"\nHorizon {h}:")
        dm_results[h] = {}

        # Get predictions for each model
        model_preds = {}
        model_actuals = None

        for model_name in MODELS.keys():
            wf = all_results[h][model_name]['walk_forward']
            if len(wf['all_predictions']) > 0:
                model_preds[model_name] = np.array(wf['all_predictions'])
                model_actuals = np.array(wf['all_actuals'])

        if model_actuals is None:
            continue

        # Compare pairs
        comparisons = [
            ('ridge', 'bayesian_ridge'),
            ('ridge', 'ard'),
            ('bayesian_ridge', 'ard')
        ]

        for m1, m2 in comparisons:
            if m1 in model_preds and m2 in model_preds:
                errors1 = model_preds[m1] - model_actuals
                errors2 = model_preds[m2] - model_actuals

                dm = diebold_mariano_test(errors1, errors2, h=h)
                dm_results[h][(m1, m2)] = dm

                print(f"   {m1} vs {m2}: DM={dm['statistic']:.3f}, p={dm['p_value']:.4f} {dm['significance']}")
                if dm['reject_H0']:
                    print(f"      -> {dm['better_model']} is significantly better")

    return all_results, dm_results


def print_summary(all_results: Dict, dm_results: Dict):
    """Print comprehensive summary."""

    print("\n" + "="*130)
    print("SUMMARY - PESARAN-TIMMERMANN TEST RESULTS")
    print("(Testing if DA is significantly > 50%)")
    print("="*130)

    print(f"\n{'MODEL':<18}", end="")
    for h in HORIZONS:
        print(f"{'H='+str(h):>16}", end="")
    print()
    print("-"*130)

    for model_name in ['ridge', 'bayesian_ridge', 'ard']:
        print(f"{model_name:<18}", end="")

        for h in HORIZONS:
            if h in all_results and model_name in all_results[h]:
                pt = all_results[h][model_name]['pesaran_timmermann']
                if pt:
                    sig = pt['significance']
                    da = pt['DA_observed']
                    p = pt['p_value']
                    if p < 0.05:
                        print(f"{da:>7.1f}%{sig:<3} OK ", end="")
                    else:
                        print(f"{da:>7.1f}%{sig:<3}    ", end="")
                else:
                    print(f"{'N/A':>16}", end="")
            else:
                print(f"{'N/A':>16}", end="")
        print()

    # Bootstrap CI summary
    print("\n" + "="*130)
    print("SUMMARY - BOOTSTRAP 95% CONFIDENCE INTERVALS")
    print("="*130)

    print(f"\n{'MODEL':<18}", end="")
    for h in HORIZONS:
        print(f"{'H='+str(h):>18}", end="")
    print()
    print("-"*144)

    for model_name in ['ridge', 'bayesian_ridge', 'ard']:
        print(f"{model_name:<18}", end="")

        for h in HORIZONS:
            if h in all_results and model_name in all_results[h]:
                boot = all_results[h][model_name]['bootstrap']
                if boot:
                    ci_str = f"[{boot['ci_lower']:.1f}, {boot['ci_upper']:.1f}]"
                    print(f"{ci_str:>18}", end="")
                else:
                    print(f"{'N/A':>18}", end="")
            else:
                print(f"{'N/A':>18}", end="")
        print()

    # Walk-forward summary
    print("\n" + "="*130)
    print("SUMMARY - WALK-FORWARD VALIDATION (Mean DA ± Std)")
    print("="*130)

    print(f"\n{'MODEL':<18}", end="")
    for h in HORIZONS:
        print(f"{'H='+str(h):>16}", end="")
    print()
    print("-"*130)

    for model_name in ['ridge', 'bayesian_ridge', 'ard']:
        print(f"{model_name:<18}", end="")

        for h in HORIZONS:
            if h in all_results and model_name in all_results[h]:
                wf = all_results[h][model_name]['walk_forward']
                if wf:
                    mean_da = wf['da_mean']
                    std_da = wf['da_std']
                    print(f"{mean_da:>7.1f}±{std_da:<5.1f}  ", end="")
                else:
                    print(f"{'N/A':>16}", end="")
            else:
                print(f"{'N/A':>16}", end="")
        print()

    # Overall conclusions
    print("\n" + "="*130)
    print("STATISTICAL CONCLUSIONS")
    print("="*130)

    significant_count = 0
    total_tests = 0

    for h in HORIZONS:
        for model_name in ['ridge', 'bayesian_ridge', 'ard']:
            if h in all_results and model_name in all_results[h]:
                pt = all_results[h][model_name]['pesaran_timmermann']
                if pt:
                    total_tests += 1
                    if pt['reject_H0']:
                        significant_count += 1

    print(f"\n1. PESARAN-TIMMERMANN TEST:")
    print(f"   Tests with DA significantly > 50%: {significant_count}/{total_tests}")
    print(f"   Percentage: {100*significant_count/total_tests:.1f}%")

    # Check if CIs exclude 50%
    ci_excludes_50 = 0
    ci_total = 0

    for h in HORIZONS:
        for model_name in ['ridge', 'bayesian_ridge', 'ard']:
            if h in all_results and model_name in all_results[h]:
                boot = all_results[h][model_name]['bootstrap']
                if boot:
                    ci_total += 1
                    if not boot['ci_contains_50']:
                        ci_excludes_50 += 1

    print(f"\n2. BOOTSTRAP CONFIDENCE INTERVALS:")
    print(f"   95% CIs that exclude 50%: {ci_excludes_50}/{ci_total}")
    print(f"   Percentage: {100*ci_excludes_50/ci_total:.1f}%")

    # Diebold-Mariano summary
    dm_significant = 0
    dm_total = 0

    for h in HORIZONS:
        if h in dm_results:
            for pair, dm in dm_results[h].items():
                dm_total += 1
                if dm['reject_H0']:
                    dm_significant += 1

    print(f"\n3. DIEBOLD-MARIANO TEST (Model Comparison):")
    print(f"   Significant differences found: {dm_significant}/{dm_total}")
    print(f"   Conclusion: {'Models are statistically different' if dm_significant > 0 else 'No significant differences between models'}")


def save_results(all_results: Dict, dm_results: Dict, output_dir: Path):
    """Save validation results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare serializable results
    summary = {
        'generated_at': datetime.now().isoformat(),
        'n_splits': N_SPLITS,
        'n_bootstrap': N_BOOTSTRAP,
        'confidence_level': CONFIDENCE_LEVEL,
        'horizons': HORIZONS,
        'results': {}
    }

    for h in HORIZONS:
        summary['results'][h] = {}
        for model_name in ['ridge', 'bayesian_ridge', 'ard']:
            if h in all_results and model_name in all_results[h]:
                r = all_results[h][model_name]
                summary['results'][h][model_name] = {
                    'walk_forward': {
                        'da_mean': r['walk_forward']['da_mean'],
                        'da_std': r['walk_forward']['da_std'],
                        'da_min': r['walk_forward']['da_min'],
                        'da_max': r['walk_forward']['da_max'],
                        'das_per_split': r['walk_forward']['das']
                    },
                    'pesaran_timmermann': {
                        'statistic': float(r['pesaran_timmermann']['statistic']) if r['pesaran_timmermann'] else None,
                        'p_value': float(r['pesaran_timmermann']['p_value']) if r['pesaran_timmermann'] else None,
                        'significant': bool(r['pesaran_timmermann']['reject_H0']) if r['pesaran_timmermann'] else None
                    },
                    'bootstrap': {
                        'point_estimate': float(r['bootstrap']['point_estimate']) if r['bootstrap'] else None,
                        'ci_lower': float(r['bootstrap']['ci_lower']) if r['bootstrap'] else None,
                        'ci_upper': float(r['bootstrap']['ci_upper']) if r['bootstrap'] else None,
                        'ci_excludes_50': bool(not r['bootstrap']['ci_contains_50']) if r['bootstrap'] else None
                    }
                }

    with open(output_dir / 'validation_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*130)
    print("STATISTICAL VALIDATION PIPELINE")
    print("="*130)
    print("\nTests implemented:")
    print("  1. Walk-Forward Validation (expanding window)")
    print("  2. Pesaran-Timmermann Test (DA significance)")
    print("  3. Diebold-Mariano Test (model comparison)")
    print("  4. Bootstrap Confidence Intervals")
    print("="*130)

    # Load data
    df = load_data()
    df, feature_cols = prepare_features(df)
    targets = create_targets(df)

    # Run validation
    all_results, dm_results = run_validation(df, feature_cols, targets)

    # Print summary
    print_summary(all_results, dm_results)

    # Save results
    save_results(all_results, dm_results, OUTPUT_DIR)

    print("\n" + "="*130)
    print("VALIDATION COMPLETE")
    print("="*130)


if __name__ == "__main__":
    main()
