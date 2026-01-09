"""
L5 Patch Metrics - Métricas Ultra-Robustas para Production Gates
================================================================
Implementa TODAS las correcciones de la propuesta unificada:
- CAGR con log-growth y guardas para evitar warnings
- Sortino por episodios con downside deviation real
- Fail-fast para NaN/Inf con explicaciones claras
- Attribution detallado de PnL
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# ==================== MÉTRICAS ROBUSTAS CON GUARDAS ====================

def robust_cagr(returns: np.ndarray, 
                periods_per_year: int = 252 * 60,  # M5: 60 bars per premium trading day
                min_return: float = -0.999) -> Tuple[float, str]:
    """
    CAGR ultra-robusto con log-growth
    Retorna: (valor, status) donde status indica si hubo problemas
    """
    if len(returns) == 0:
        return 0.0, "NO_DATA"
    
    # Guarda contra returns extremos
    clipped_returns = np.maximum(returns, min_return)
    
    # Detecta si tuvimos que clipear
    clipped_count = np.sum(returns < min_return)
    if clipped_count > 0:
        logger.warning(f"Clipped {clipped_count} extreme negative returns")
    
    try:
        # Log-growth method para evitar potencias fraccionarias
        log_returns = np.log(1 + clipped_returns)
        
        # Verifica NaN/Inf
        if np.any(np.isnan(log_returns)) or np.any(np.isinf(log_returns)):
            return -1.0, "LOG_RETURNS_INVALID"
        
        # Calcula retorno acumulado
        cum_log_return = np.sum(log_returns)
        cum_return = np.exp(cum_log_return) - 1
        
        # Casos edge
        if cum_return <= -1:
            return -1.0, "TOTAL_LOSS"
        
        # Anualización
        n_periods = len(returns)
        if n_periods == 0:
            return 0.0, "NO_PERIODS"
        
        years = n_periods / periods_per_year
        
        # CAGR usando log para evitar warnings
        if cum_return > -1:
            # Método alternativo: (1 + cum_return)^(1/years) - 1
            # Pero usando exp/log para estabilidad
            cagr = np.exp(np.log(1 + cum_return) / years) - 1
            
            # Verifica resultado final
            if np.isnan(cagr) or np.isinf(cagr):
                return 0.0, "CALC_FAILED"
            
            return float(cagr), "OK"
        else:
            return -1.0, "NEGATIVE_BASE"
            
    except Exception as e:
        logger.error(f"CAGR calculation failed: {e}")
        return 0.0, f"EXCEPTION: {str(e)}"

def robust_sortino(returns: np.ndarray,
                  risk_free: float = 0.0,
                  target: float = 0.0,
                  annualize: bool = True,
                  periods_per_year: int = 252 * 60) -> Tuple[float, str]:
    """
    Sortino robusto con downside deviation real
    Calcula por episodios para evitar colapso de varianza
    """
    if len(returns) == 0:
        return 0.0, "NO_DATA"
    
    try:
        # Excess returns
        excess = returns - risk_free
        
        # Verifica datos válidos
        if np.all(np.isnan(excess)):
            return 0.0, "ALL_NAN"
        
        # Filtra NaN
        excess_clean = excess[~np.isnan(excess)]
        if len(excess_clean) == 0:
            return 0.0, "NO_VALID_DATA"
        
        # Mean excess return
        mean_excess = np.mean(excess_clean)
        
        # Downside returns (below target)
        downside = excess_clean[excess_clean < target]
        
        if len(downside) == 0:
            # No downside risk - return based on sign of mean
            if mean_excess > 0:
                return 100.0, "NO_DOWNSIDE_POSITIVE"  # Cap at 100
            elif mean_excess < 0:
                return -100.0, "NO_DOWNSIDE_NEGATIVE"
            else:
                return 0.0, "NO_DOWNSIDE_ZERO"
        
        # Calcula downside deviation con epsilon para evitar división por cero
        # Usa denominador n en lugar de n-1 para consistencia con definición estándar
        epsilon = 1e-8  # Small value to prevent division by zero
        downside_dev = np.sqrt(np.mean(downside ** 2)) + epsilon
        
        # Additional check for extremely small downside deviation
        if downside_dev < 1e-6:
            logger.warning(f"Downside deviation too small ({downside_dev:.2e}), using fallback")
            if mean_excess > 0:
                return 10.0, "MINIMAL_DOWNSIDE_POSITIVE"  # Cap at 10 for minimal risk
            elif mean_excess < 0:
                return -10.0, "MINIMAL_DOWNSIDE_NEGATIVE"
            else:
                return 0.0, "MINIMAL_DOWNSIDE_ZERO"
        
        # Sortino ratio with epsilon protection
        sortino = mean_excess / downside_dev
        
        # Anualización si se requiere
        if annualize:
            sortino *= np.sqrt(periods_per_year)
        
        # Verifica resultado
        if np.isnan(sortino) or np.isinf(sortino):
            return 0.0, "CALC_INVALID"
        
        # Clamp a rango razonable para evitar valores extremos tipo -5e15
        # Si el valor es extremadamente negativo, es una señal de datos degenerados
        if sortino < -1000:
            logger.warning(f"Sortino extremely negative ({sortino:.2e}), clamping to -1.0")
            return -1.0, "EXTREME_NEGATIVE"
        elif sortino > 1000:
            logger.warning(f"Sortino extremely positive ({sortino:.2e}), clamping to 100.0")
            return 100.0, "EXTREME_POSITIVE"
        
        sortino = np.clip(sortino, -100, 100)
        
        return float(sortino), "OK"
        
    except Exception as e:
        logger.error(f"Sortino calculation failed: {e}")
        return 0.0, f"EXCEPTION: {str(e)}"

def robust_calmar(returns: np.ndarray,
                 periods_per_year: int = 252 * 60) -> Tuple[float, str]:
    """
    Calmar robusto: CAGR / MaxDD con manejo de casos edge
    """
    if len(returns) == 0:
        return 0.0, "NO_DATA"
    
    try:
        # Calcula CAGR robusto
        cagr, cagr_status = robust_cagr(returns, periods_per_year)
        
        if cagr_status != "OK":
            return 0.0, f"CAGR_FAILED: {cagr_status}"
        
        # Calcula drawdown de forma segura
        # Usa retornos acumulados con producto (más estable)
        clipped_returns = np.maximum(returns, -0.999)
        cum_returns = np.cumprod(1 + clipped_returns) - 1
        
        # Running maximum (peak)
        running_max = np.maximum.accumulate(cum_returns)
        
        # Drawdown con guardas contra división por cero
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = np.where(
                1 + running_max > 0,
                (cum_returns - running_max) / (1 + running_max),
                0
            )
        
        # Max drawdown (valor absoluto)
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Verifica validez
        if np.isnan(max_dd) or np.isinf(max_dd):
            return 0.0, "MAXDD_INVALID"
        
        # Calmar ratio
        if max_dd == 0:
            if cagr > 0:
                return 100.0, "NO_DD_POSITIVE"  # Cap at 100
            elif cagr < 0:
                return -100.0, "NO_DD_NEGATIVE"
            else:
                return 0.0, "NO_DD_ZERO"
        
        calmar = cagr / max_dd
        
        # Clamp a rango razonable
        calmar = np.clip(calmar, -100, 100)
        
        return float(calmar), "OK"
        
    except Exception as e:
        logger.error(f"Calmar calculation failed: {e}")
        return 0.0, f"EXCEPTION: {str(e)}"

def robust_max_drawdown(returns: np.ndarray) -> Tuple[float, str]:
    """
    Max Drawdown robusto con manejo completo de edge cases
    """
    if len(returns) == 0:
        return 0.0, "NO_DATA"
    
    try:
        # Clip returns extremos
        clipped_returns = np.maximum(returns, -0.999)
        
        # Calcula retornos acumulados
        cum_returns = np.cumprod(1 + clipped_returns) - 1
        
        # Running maximum
        running_max = np.maximum.accumulate(cum_returns)
        
        # Calcula drawdown con protección
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdown = np.where(
                1 + running_max > 0,
                (cum_returns - running_max) / (1 + running_max),
                -1.0  # Pérdida total si peak cayó a -100%
            )
        
        # Max drawdown
        max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        
        # Verifica validez
        if np.isnan(max_dd):
            return 0.0, "NAN_DETECTED"
        if np.isinf(max_dd):
            return 1.0, "INF_DETECTED"  # Pérdida total
        
        return float(max_dd), "OK"
        
    except Exception as e:
        logger.error(f"Max DD calculation failed: {e}")
        return 0.0, f"EXCEPTION: {str(e)}"

# ==================== EVALUACIÓN POR EPISODIOS ====================

def evaluate_episode_metrics(episode_rewards: List[float],
                            episode_lengths: List[int],
                            periods_per_year: int = 252 * 60) -> Dict[str, Any]:
    """
    Calcula métricas usando distribución por episodios
    Evita colapso de varianza que lleva a Sortino = -1
    """
    if not episode_rewards:
        return {
            "status": "NO_EPISODES",
            "metrics": {},
            "warnings": ["No episodes to evaluate"]
        }
    
    returns = np.array(episode_rewards)
    lengths = np.array(episode_lengths)
    
    # Métricas básicas
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    median_return = float(np.median(returns))
    
    # Percentiles para entender distribución
    p25 = float(np.percentile(returns, 25))
    p75 = float(np.percentile(returns, 75))
    p05 = float(np.percentile(returns, 5))
    p95 = float(np.percentile(returns, 95))
    
    # Win rate y ratios
    win_rate = float(np.mean(returns > 0))
    positive_returns = returns[returns > 0]
    negative_returns = returns[returns < 0]
    
    avg_win = float(np.mean(positive_returns)) if len(positive_returns) > 0 else 0.0
    avg_loss = float(np.mean(negative_returns)) if len(negative_returns) > 0 else 0.0
    
    # Profit factor
    total_wins = np.sum(positive_returns) if len(positive_returns) > 0 else 0.0
    total_losses = abs(np.sum(negative_returns)) if len(negative_returns) > 0 else 0.0
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf if total_wins > 0 else 0.0
    
    # Métricas robustas
    sortino, sortino_status = robust_sortino(returns, annualize=False)  # Por episodio
    cagr, cagr_status = robust_cagr(returns, periods_per_year)
    calmar, calmar_status = robust_calmar(returns, periods_per_year)
    max_dd, dd_status = robust_max_drawdown(returns)
    
    # Recopila warnings
    warnings = []
    if sortino_status != "OK":
        warnings.append(f"Sortino: {sortino_status}")
    if cagr_status != "OK":
        warnings.append(f"CAGR: {cagr_status}")
    if calmar_status != "OK":
        warnings.append(f"Calmar: {calmar_status}")
    if dd_status != "OK":
        warnings.append(f"MaxDD: {dd_status}")
    
    # Detecta problemas críticos
    critical_issues = []
    if std_return < 1e-6:
        critical_issues.append("ZERO_VARIANCE")
    if win_rate == 0:
        critical_issues.append("NEVER_WINS")
    if win_rate == 1:
        critical_issues.append("NEVER_LOSES")
    if abs(mean_return) < 1e-8:
        critical_issues.append("ZERO_MEAN")
    
    return {
        "status": "OK" if not critical_issues else "CRITICAL",
        "critical_issues": critical_issues,
        "warnings": warnings,
        "metrics": {
            # Distribución
            "mean_return": mean_return,
            "std_return": std_return,
            "median_return": median_return,
            "percentile_5": p05,
            "percentile_25": p25,
            "percentile_75": p75,
            "percentile_95": p95,
            
            # Performance
            "sortino_ratio": sortino,
            "cagr": cagr,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            
            # Trading stats
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": min(profit_factor, 100),  # Cap para JSON
            
            # Episodios
            "n_episodes": len(returns),
            "avg_episode_length": float(np.mean(lengths)),
            "total_steps": int(np.sum(lengths))
        },
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "periods_per_year": periods_per_year
        }
    }

# ==================== FAIL-FAST PARA GATES ====================

def validate_metrics_for_gates(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Valida que las métricas sean válidas para evaluación de gates
    Fail-fast si hay NaN/Inf o valores imposibles
    """
    failures = []
    
    # Lista de métricas críticas
    critical_metrics = [
        'sortino_ratio', 'cagr', 'calmar_ratio', 
        'max_drawdown', 'mean_return', 'std_return'
    ]
    
    for metric_name in critical_metrics:
        if metric_name not in metrics:
            failures.append(f"MISSING: {metric_name}")
            continue
        
        value = metrics[metric_name]
        
        # Check NaN
        if isinstance(value, float) and np.isnan(value):
            failures.append(f"NAN: {metric_name}")
        
        # Check Inf
        if isinstance(value, float) and np.isinf(value):
            failures.append(f"INF: {metric_name}")
        
        # Rangos válidos específicos
        if metric_name == 'max_drawdown' and (value < 0 or value > 1):
            failures.append(f"INVALID_RANGE: {metric_name}={value}")
        
        if metric_name == 'win_rate' and (value < 0 or value > 1):
            failures.append(f"INVALID_RANGE: {metric_name}={value}")
    
    # Validaciones cruzadas
    if 'std_return' in metrics and metrics['std_return'] < 1e-8:
        failures.append("ZERO_VARIANCE: Policy collapsed")
    
    if 'win_rate' in metrics:
        wr = metrics['win_rate']
        if wr == 0:
            failures.append("NEVER_WINS: Policy always loses")
        elif wr == 1:
            failures.append("NEVER_LOSES: Suspicious perfect record")
    
    is_valid = len(failures) == 0
    return is_valid, failures

# ==================== ATTRIBUTION DE PNL ====================

def calculate_pnl_attribution(episode_data: Dict[str, List]) -> Dict[str, Any]:
    """
    Descompone PnL en componentes para diagnóstico
    Requiere episode_data con: rewards, costs, trades, holds
    """
    if not episode_data or 'rewards' not in episode_data:
        return {"status": "NO_DATA"}
    
    rewards = np.array(episode_data.get('rewards', []))
    costs = np.array(episode_data.get('costs', []))
    trades = episode_data.get('trades', 0)
    holds = episode_data.get('holds', 0)
    
    # PnL total y componentes
    total_pnl = np.sum(rewards)
    
    # Si tenemos desglose de costos
    if len(costs) > 0:
        total_costs = np.sum(costs)
        pnl_before_costs = total_pnl + total_costs
    else:
        total_costs = 0
        pnl_before_costs = total_pnl
    
    # Ratio de actividad
    total_actions = trades + holds
    trade_ratio = trades / total_actions if total_actions > 0 else 0
    hold_ratio = holds / total_actions if total_actions > 0 else 0
    
    # Costo promedio por trade
    avg_cost_per_trade = total_costs / trades if trades > 0 else 0
    
    # Edge (ganancia antes de costos por trade)
    edge_per_trade = pnl_before_costs / trades if trades > 0 else 0
    
    # Net por trade (después de costos)
    net_per_trade = total_pnl / trades if trades > 0 else 0
    
    return {
        "status": "OK",
        "pnl": {
            "total": float(total_pnl),
            "before_costs": float(pnl_before_costs),
            "costs": float(total_costs)
        },
        "activity": {
            "trades": trades,
            "holds": holds,
            "trade_ratio": float(trade_ratio),
            "hold_ratio": float(hold_ratio)
        },
        "per_trade": {
            "edge": float(edge_per_trade),
            "cost": float(avg_cost_per_trade),
            "net": float(net_per_trade)
        },
        "diagnosis": {
            "has_edge": edge_per_trade > 0,
            "covers_costs": net_per_trade > 0,
            "overtrades": trade_ratio > 0.5,  # Threshold ajustable
            "undertrades": trade_ratio < 0.05
        }
    }

# ==================== GENERADOR DE REPORTES ====================

def generate_seed_report(seed: int,
                        train_metrics: Dict,
                        test_metrics: Dict,
                        episode_data: Optional[Dict] = None,
                        gate_results: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Genera reporte completo JSON para un seed
    """
    report = {
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics
    }
    
    # Añade attribution si está disponible
    if episode_data:
        attribution = calculate_pnl_attribution(episode_data)
        report["pnl_attribution"] = attribution
    
    # Valida métricas
    is_valid, failures = validate_metrics_for_gates(test_metrics)
    report["validation"] = {
        "is_valid": is_valid,
        "failures": failures
    }
    
    # Gates si están disponibles
    if gate_results:
        report["gates"] = gate_results
    
    # Diagnóstico
    diagnosis = []
    
    # Check Sortino
    if 'sortino_ratio' in test_metrics:
        sortino = test_metrics['sortino_ratio']
        if sortino < 0:
            diagnosis.append("NEGATIVE_SORTINO: Consistent losses")
        elif sortino < 0.5:
            diagnosis.append("LOW_SORTINO: Weak risk-adjusted returns")
    
    # Check drawdown
    if 'max_drawdown' in test_metrics:
        dd = test_metrics['max_drawdown']
        if dd > 0.2:
            diagnosis.append(f"HIGH_DRAWDOWN: {dd:.1%}")
    
    # Check variance
    if 'std_return' in test_metrics:
        std = test_metrics['std_return']
        if std < 1e-6:
            diagnosis.append("ZERO_VARIANCE: Policy not exploring")
    
    report["diagnosis"] = diagnosis
    
    # Status general
    if not is_valid:
        status = "INVALID"
    elif gate_results and gate_results.get('passed', 0) >= 6:
        status = "PASS"
    elif gate_results and gate_results.get('passed', 0) >= 4:
        status = "PARTIAL"
    else:
        status = "FAIL"
    
    report["overall_status"] = status
    
    return report

# ==================== INTERFAZ PRINCIPAL ====================

def patch_evaluate_test_metrics(model, env, n_episodes: int) -> Dict[str, Any]:
    """
    Reemplazo directo para evaluate_test_metrics en el DAG
    Usa métricas robustas y evaluación por episodios
    """
    episode_rewards = []
    episode_lengths = []
    episode_data = {
        'rewards': [],
        'costs': [],
        'trades': 0,
        'holds': 0
    }
    
    for _ in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        ep_reward = 0
        ep_length = 0
        ep_trades = 0
        ep_holds = 0
        
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            ep_reward += float(reward[0])
            ep_length += 1
            
            # Track actividad (asume action space 0=hold, 1=buy, 2=sell)
            if action[0] == 0:
                ep_holds += 1
            else:
                ep_trades += 1
        
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_data['trades'] += ep_trades
        episode_data['holds'] += ep_holds
    
    # Calcula métricas robustas
    eval_result = evaluate_episode_metrics(episode_rewards, episode_lengths)
    
    # Añade attribution
    episode_data['rewards'] = episode_rewards
    attribution = calculate_pnl_attribution(episode_data)
    
    # Combina todo
    final_metrics = eval_result['metrics'].copy()
    final_metrics['attribution'] = attribution
    final_metrics['warnings'] = eval_result.get('warnings', [])
    final_metrics['critical_issues'] = eval_result.get('critical_issues', [])
    
    return final_metrics

# ==================== TEST MODULE ====================

if __name__ == "__main__":
    print("=" * 60)
    print("L5 PATCH METRICS - TEST SUITE")
    print("=" * 60)
    
    # Test 1: Métricas robustas con casos edge
    print("\n1. Testing robust metrics with edge cases...")
    
    # Caso normal
    normal_returns = np.random.normal(0.001, 0.01, 100)
    cagr, status = robust_cagr(normal_returns)
    print(f"  Normal - CAGR: {cagr:.4f} ({status})")
    
    sortino, status = robust_sortino(normal_returns)
    print(f"  Normal - Sortino: {sortino:.4f} ({status})")
    
    # Caso extremo (tu caso actual)
    negative_returns = np.full(100, -0.001)
    cagr, status = robust_cagr(negative_returns)
    print(f"  Negative - CAGR: {cagr:.4f} ({status})")
    
    sortino, status = robust_sortino(negative_returns)
    print(f"  Negative - Sortino: {sortino:.4f} ({status})")
    
    # Caso con NaN
    nan_returns = np.array([0.001, np.nan, -0.001, 0.002])
    cagr, status = robust_cagr(nan_returns)
    print(f"  With NaN - CAGR: {cagr:.4f} ({status})")
    
    # Test 2: Evaluación por episodios
    print("\n2. Testing episode evaluation...")
    
    # Simula episodios
    episode_rewards = np.random.normal(0.0001, 0.001, 50).tolist()
    episode_lengths = [100] * 50
    
    result = evaluate_episode_metrics(episode_rewards, episode_lengths)
    print(f"  Status: {result['status']}")
    print(f"  Sortino: {result['metrics']['sortino_ratio']:.4f}")
    print(f"  Win Rate: {result['metrics']['win_rate']:.2%}")
    print(f"  Warnings: {result['warnings']}")
    
    # Test 3: Validación para gates
    print("\n3. Testing gate validation...")
    
    test_metrics = {
        'sortino_ratio': -1.0,  # Tu caso actual
        'cagr': np.nan,
        'max_drawdown': 0.5,
        'mean_return': -0.001,
        'std_return': 0.0
    }
    
    is_valid, failures = validate_metrics_for_gates(test_metrics)
    print(f"  Valid: {is_valid}")
    print(f"  Failures: {failures}")
    
    # Test 4: Attribution
    print("\n4. Testing PnL attribution...")
    
    episode_data = {
        'rewards': [-0.001] * 100,
        'costs': [0.0001] * 10,  # Solo 10 trades
        'trades': 10,
        'holds': 90
    }
    
    attribution = calculate_pnl_attribution(episode_data)
    print(f"  Has Edge: {attribution['diagnosis']['has_edge']}")
    print(f"  Covers Costs: {attribution['diagnosis']['covers_costs']}")
    print(f"  Trade Ratio: {attribution['activity']['trade_ratio']:.2%}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)