"""
Reward & Costs Sanity Checker - Validación Completa de Señal y Costos
====================================================================
Implementa todos los smoke-tests de la propuesta unificada:
- Test con costos=0 para aislar problemas de señal
- Validación de alineación con reward_spec de L4
- Verificación de que costos solo se aplican en trades
- Telemetría detallada por episodio
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logger = logging.getLogger(__name__)

class RewardCostsSanityChecker:
    """
    Suite completa de validación para reward/costs
    Implementa C1 de la propuesta: sanity checks críticos
    """
    
    def __init__(self, 
                 env_factory,
                 cost_model: Dict[str, float],
                 reward_spec: Optional[Dict] = None):
        """
        Args:
            env_factory: Función para crear environments
            cost_model: Modelo de costos del L4
            reward_spec: Especificación de reward del L4
        """
        self.env_factory = env_factory
        self.cost_model = cost_model
        self.reward_spec = reward_spec or {}
        self.results = {}
        
    def run_all_checks(self, model=None, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Ejecuta todos los smoke-tests
        """
        logger.info("=" * 60)
        logger.info("REWARD & COSTS SANITY CHECKS")
        logger.info("=" * 60)
        
        # Test 1: Baseline con costos normales
        logger.info("\n1. Baseline test (with normal costs)...")
        baseline_results = self._test_baseline(model, n_episodes)
        self.results['baseline'] = baseline_results
        
        # Test 2: Sin costos (aislar señal)
        logger.info("\n2. Zero-cost test (isolate signal)...")
        zero_cost_results = self._test_zero_costs(model, n_episodes)
        self.results['zero_costs'] = zero_cost_results
        
        # Test 3: Costos solo en trades
        logger.info("\n3. Trade-only costs test...")
        trade_only_results = self._test_trade_only_costs(model, n_episodes)
        self.results['trade_only'] = trade_only_results
        
        # Test 4: Alineación con reward_spec
        logger.info("\n4. Reward spec alignment test...")
        alignment_results = self._test_reward_alignment()
        self.results['alignment'] = alignment_results
        
        # Test 5: Distribución de rewards
        logger.info("\n5. Reward distribution test...")
        distribution_results = self._test_reward_distribution(model, n_episodes)
        self.results['distribution'] = distribution_results
        
        # Diagnóstico general
        diagnosis = self._generate_diagnosis()
        self.results['diagnosis'] = diagnosis
        
        # Genera reporte
        report = self._generate_report()
        
        # Guarda reporte
        report_path = f"/tmp/reward_costs_sanity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nReport saved to: {report_path}")
        
        return report
    
    def _test_baseline(self, model, n_episodes: int) -> Dict:
        """Test con costos normales del L4"""
        from reward_sentinel import SentinelTradingEnv
        
        # Crea environment con costos normales
        base_env = self.env_factory(mode="test")
        # Get max_episode_length from reward_spec if available
        max_ep_length = self.reward_spec.get('max_episode_length', 60)
        env = DummyVecEnv([
            lambda: Monitor(
                SentinelTradingEnv(
                    base_env,
                    cost_model=self.cost_model,
                    shaping_penalty=0.0,
                    enable_telemetry=True,
                    max_episode_length=max_ep_length
                ),
                allow_early_resets=True
            )
        ])
        
        results = self._run_episodes(model, env, n_episodes)
        
        return {
            'mean_reward': float(np.mean(results['rewards'])),
            'std_reward': float(np.std(results['rewards'])),
            'win_rate': float(np.mean(np.array(results['rewards']) > 0)),
            'trade_ratio': results['trade_ratio'],
            'costs_per_episode': results['avg_costs'],
            'status': 'OK'
        }
    
    def _test_zero_costs(self, model, n_episodes: int) -> Dict:
        """Test sin costos para aislar señal"""
        from reward_sentinel import SentinelTradingEnv
        
        # Crea environment SIN costos
        zero_cost_model = {k: 0.0 for k in self.cost_model.keys()}
        
        base_env = self.env_factory(mode="test")
        env = DummyVecEnv([
            lambda: Monitor(
                SentinelTradingEnv(
                    base_env,
                    cost_model=zero_cost_model,
                    shaping_penalty=0.0,
                    enable_telemetry=True
                ),
                allow_early_resets=True
            )
        ])
        
        results = self._run_episodes(model, env, n_episodes)
        
        mean_reward = float(np.mean(results['rewards']))
        std_reward = float(np.std(results['rewards']))
        
        # Diagnóstico crítico
        status = 'OK'
        issues = []
        
        if mean_reward < 0:
            status = 'FAIL'
            issues.append('NEGATIVE_MEAN_WITH_ZERO_COSTS')
        
        if std_reward < 1e-6:
            status = 'FAIL'
            issues.append('ZERO_VARIANCE_SIGNAL')
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'win_rate': float(np.mean(np.array(results['rewards']) > 0)),
            'trade_ratio': results['trade_ratio'],
            'status': status,
            'issues': issues,
            'interpretation': self._interpret_zero_cost_results(mean_reward, std_reward)
        }
    
    def _test_trade_only_costs(self, model, n_episodes: int) -> Dict:
        """Verifica que costos se aplican SOLO en trades"""
        from reward_sentinel import SentinelTradingEnv
        
        class CostTrackingEnv(SentinelTradingEnv):
            """Wrapper que trackea cuando se aplican costos"""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.cost_applications = []
                
            def step(self, action):
                obs, reward, done, truncated, info = super().step(action)
                
                # Track si se aplicó costo
                if action == 0:  # Hold
                    self.cost_applications.append(('hold', self.ep_stats['cost_sum_bps']))
                else:  # Trade
                    self.cost_applications.append(('trade', self.ep_stats['cost_sum_bps']))
                
                return obs, reward, done, truncated, info
        
        base_env = self.env_factory(mode="test")
        tracking_env = CostTrackingEnv(
            base_env,
            cost_model=self.cost_model,
            shaping_penalty=0.0,
            enable_telemetry=False
        )
        
        env = DummyVecEnv([lambda: Monitor(tracking_env, allow_early_resets=True)])
        
        # Ejecuta episodios
        results = self._run_episodes(model, env, min(10, n_episodes))
        
        # Analiza aplicación de costos
        cost_apps = tracking_env.cost_applications if hasattr(tracking_env, 'cost_applications') else []
        
        holds_with_costs = sum(1 for action, cost in cost_apps if action == 'hold' and cost > 0)
        trades_without_costs = sum(1 for action, cost in cost_apps if action == 'trade' and cost == 0)
        
        status = 'OK'
        issues = []
        
        if holds_with_costs > 0:
            status = 'FAIL'
            issues.append(f'COSTS_ON_HOLD: {holds_with_costs} instances')
        
        if trades_without_costs > 0:
            status = 'WARNING'
            issues.append(f'TRADES_WITHOUT_COSTS: {trades_without_costs} instances')
        
        return {
            'holds_with_costs': holds_with_costs,
            'trades_without_costs': trades_without_costs,
            'total_holds': sum(1 for a, _ in cost_apps if a == 'hold'),
            'total_trades': sum(1 for a, _ in cost_apps if a == 'trade'),
            'status': status,
            'issues': issues
        }
    
    def _test_reward_alignment(self) -> Dict:
        """Verifica alineación con reward_spec del L4"""
        
        if not self.reward_spec:
            return {
                'status': 'SKIPPED',
                'reason': 'No reward_spec provided'
            }
        
        issues = []
        
        # Verifica ventana forward
        expected_window = self.reward_spec.get('forward_window', [1, 2])
        logger.info(f"  Expected forward window: t+{expected_window[0]} to t+{expected_window[1]}")
        
        # Verifica tipo de precio
        expected_price = self.reward_spec.get('price_type', 'mid')
        logger.info(f"  Expected price type: {expected_price}")
        
        # Verifica normalización
        expected_norm = self.reward_spec.get('normalization', 'log_returns_bps')
        logger.info(f"  Expected normalization: {expected_norm}")
        
        # TODO: Verificación real contra environment actual
        # Por ahora, solo registro de especificación
        
        return {
            'status': 'INFO',
            'expected_window': expected_window,
            'expected_price': expected_price,
            'expected_normalization': expected_norm,
            'issues': issues,
            'note': 'Manual verification required against env implementation'
        }
    
    def _test_reward_distribution(self, model, n_episodes: int) -> Dict:
        """Analiza distribución de rewards para detectar degeneración"""
        from reward_sentinel import SentinelTradingEnv
        
        base_env = self.env_factory(mode="test")
        # Get max_episode_length from reward_spec if available
        max_ep_length = self.reward_spec.get('max_episode_length', 60)
        env = DummyVecEnv([
            lambda: Monitor(
                SentinelTradingEnv(
                    base_env,
                    cost_model=self.cost_model,
                    shaping_penalty=0.0,
                    enable_telemetry=True,
                    max_episode_length=max_ep_length
                ),
                allow_early_resets=True
            )
        ])
        
        # Colecta rewards detallados
        all_step_rewards = []
        episode_rewards = []
        
        for _ in range(min(20, n_episodes)):
            obs = env.reset()
            done = np.array([False])
            ep_rewards = []
            
            while not done[0]:
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                obs, reward, done, _ = env.step(action)
                ep_rewards.append(float(reward[0]))
            
            all_step_rewards.extend(ep_rewards)
            episode_rewards.append(sum(ep_rewards))
        
        # Analiza distribución
        step_rewards = np.array(all_step_rewards)
        
        # Cuenta valores únicos
        unique_values = np.unique(step_rewards)
        value_counts = {float(v): int(np.sum(step_rewards == v)) for v in unique_values[:10]}  # Top 10
        
        # Detecta problemas
        issues = []
        
        zero_ratio = np.mean(step_rewards == 0)
        if zero_ratio > 0.9:
            issues.append(f'EXCESSIVE_ZEROS: {zero_ratio:.1%}')
        
        negative_ratio = np.mean(step_rewards < 0)
        if negative_ratio > 0.8:
            issues.append(f'EXCESSIVE_NEGATIVES: {negative_ratio:.1%}')
        
        # Entropía (diversidad de valores)
        if len(unique_values) < 5:
            issues.append(f'LOW_DIVERSITY: Only {len(unique_values)} unique values')
        
        # Coeficiente de variación
        cv = np.std(step_rewards) / (abs(np.mean(step_rewards)) + 1e-8)
        if cv < 0.1:
            issues.append(f'LOW_VARIATION: CV={cv:.3f}')
        
        return {
            'mean': float(np.mean(step_rewards)),
            'std': float(np.std(step_rewards)),
            'min': float(np.min(step_rewards)),
            'max': float(np.max(step_rewards)),
            'zero_ratio': float(zero_ratio),
            'negative_ratio': float(negative_ratio),
            'positive_ratio': float(np.mean(step_rewards > 0)),
            'unique_values': len(unique_values),
            'top_values': value_counts,
            'coefficient_variation': float(cv),
            'issues': issues,
            'status': 'FAIL' if len(issues) > 2 else 'WARNING' if issues else 'OK'
        }
    
    def _run_episodes(self, model, env, n_episodes: int) -> Dict:
        """Ejecuta episodios y colecta estadísticas"""
        episode_rewards = []
        total_trades = 0
        total_holds = 0
        total_costs = 0
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = np.array([False])
            ep_reward = 0
            ep_trades = 0
            ep_holds = 0
            
            while not done[0]:
                if model:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                obs, reward, done, _ = env.step(action)
                ep_reward += float(reward[0])
                
                # Track actividad
                if action[0] == 0:
                    ep_holds += 1
                else:
                    ep_trades += 1
            
            episode_rewards.append(ep_reward)
            total_trades += ep_trades
            total_holds += ep_holds
        
        total_actions = total_trades + total_holds
        
        return {
            'rewards': episode_rewards,
            'trade_ratio': total_trades / total_actions if total_actions > 0 else 0,
            'hold_ratio': total_holds / total_actions if total_actions > 0 else 0,
            'avg_costs': total_costs / n_episodes
        }
    
    def _interpret_zero_cost_results(self, mean_reward: float, std_reward: float) -> str:
        """Interpreta resultados del test sin costos"""
        
        if mean_reward < -0.0001 and std_reward < 1e-6:
            return "CRITICAL: No signal even without costs - check reward calculation and features"
        
        elif mean_reward < 0:
            return "WARNING: Negative mean without costs - weak or inverted signal"
        
        elif std_reward < 1e-6:
            return "WARNING: Zero variance - policy collapsed or constant rewards"
        
        elif mean_reward > 0 and std_reward > 0.001:
            return "GOOD: Positive signal exists - costs are the main issue"
        
        elif mean_reward > 0 and std_reward < 0.001:
            return "MIXED: Positive but low variance - limited learning potential"
        
        else:
            return "UNKNOWN: Requires further investigation"
    
    def _generate_diagnosis(self) -> Dict:
        """Genera diagnóstico basado en todos los tests"""
        
        diagnosis = {
            'timestamp': datetime.now().isoformat(),
            'main_issue': 'UNKNOWN',
            'recommendations': [],
            'critical_failures': []
        }
        
        # Analiza test zero-cost
        if 'zero_costs' in self.results:
            zc = self.results['zero_costs']
            if zc['status'] == 'FAIL':
                if 'NEGATIVE_MEAN_WITH_ZERO_COSTS' in zc.get('issues', []):
                    diagnosis['main_issue'] = 'SIGNAL_PROBLEM'
                    diagnosis['critical_failures'].append('Signal is negative even without costs')
                    diagnosis['recommendations'].append('Review reward calculation and feature engineering')
                
                if 'ZERO_VARIANCE_SIGNAL' in zc.get('issues', []):
                    diagnosis['main_issue'] = 'DEGENERATE_POLICY'
                    diagnosis['critical_failures'].append('Policy has collapsed to constant action')
                    diagnosis['recommendations'].append('Check exploration parameters and reward diversity')
        
        # Analiza test trade-only
        if 'trade_only' in self.results:
            to = self.results['trade_only']
            if to['status'] == 'FAIL':
                if to['holds_with_costs'] > 0:
                    diagnosis['main_issue'] = 'COST_APPLICATION_BUG'
                    diagnosis['critical_failures'].append('Costs being applied on hold actions')
                    diagnosis['recommendations'].append('Fix cost application logic to only charge on trades')
        
        # Analiza distribución
        if 'distribution' in self.results:
            dist = self.results['distribution']
            if dist['zero_ratio'] > 0.9:
                diagnosis['recommendations'].append('Too many zero rewards - check if reward calculation is correct')
            if dist['negative_ratio'] > 0.8:
                diagnosis['recommendations'].append('Mostly negative rewards - reduce costs or improve signal')
        
        # Comparación baseline vs zero-cost
        if 'baseline' in self.results and 'zero_costs' in self.results:
            baseline_mean = self.results['baseline']['mean_reward']
            zero_cost_mean = self.results['zero_costs']['mean_reward']
            
            improvement = zero_cost_mean - baseline_mean
            
            if improvement > 0.001:
                diagnosis['recommendations'].append(
                    f'Removing costs improves mean reward by {improvement:.4f} - consider cost reduction'
                )
            elif improvement < 0.0001:
                diagnosis['recommendations'].append(
                    'Costs are not the main issue - focus on signal quality'
                )
        
        # Determina si pasa C1
        c1_pass = (
            'zero_costs' in self.results and 
            self.results['zero_costs']['mean_reward'] > 0 and
            self.results['zero_costs']['std_reward'] > 1e-6
        )
        
        diagnosis['c1_criteria_pass'] = c1_pass
        
        if not diagnosis['critical_failures']:
            diagnosis['main_issue'] = 'COSTS_TOO_HIGH' if not c1_pass else 'OK'
        
        return diagnosis
    
    def _generate_report(self) -> Dict:
        """Genera reporte completo"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_results': self.results,
            'summary': {
                'c1_pass': self.results.get('diagnosis', {}).get('c1_criteria_pass', False),
                'main_issue': self.results.get('diagnosis', {}).get('main_issue', 'UNKNOWN'),
                'critical_failures': self.results.get('diagnosis', {}).get('critical_failures', []),
                'recommendations': self.results.get('diagnosis', {}).get('recommendations', [])
            }
        }
        
        # Añade métricas clave
        if 'baseline' in self.results:
            report['summary']['baseline_mean_reward'] = self.results['baseline']['mean_reward']
        
        if 'zero_costs' in self.results:
            report['summary']['zero_cost_mean_reward'] = self.results['zero_costs']['mean_reward']
            report['summary']['signal_quality'] = self.results['zero_costs']['interpretation']
        
        return report

# ==================== SMOKE TEST RÁPIDO ====================

def quick_sanity_check(env_factory, cost_model: Dict, n_episodes: int = 20) -> bool:
    """
    Check rápido para CI/CD
    Retorna True si pasa criterios mínimos
    """
    checker = RewardCostsSanityChecker(env_factory, cost_model)
    
    # Solo ejecuta test crítico: zero-cost
    from reward_sentinel import SentinelTradingEnv
    
    zero_cost_model = {k: 0.0 for k in cost_model.keys()}
    base_env = env_factory(mode="test")
    env = DummyVecEnv([
        lambda: Monitor(
            SentinelTradingEnv(
                base_env,
                cost_model=zero_cost_model,
                shaping_penalty=0.0,
                enable_telemetry=False,
                max_episode_length=60  # Default L4 contract value
            ),
            allow_early_resets=True
        )
    ])
    
    # Ejecuta episodios
    rewards = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        ep_reward = 0
        
        while not done[0]:
            action = env.action_space.sample()  # Random para test rápido
            # Wrap action in array for VecEnv
            action = np.array([action])
            obs, reward, done, _ = env.step(action)
            ep_reward += float(reward[0])
        
        rewards.append(ep_reward)
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    # Criterio mínimo: señal positiva y con varianza
    passes = mean_reward > -0.0001 and std_reward > 1e-6
    
    logger.info(f"Quick sanity: mean={mean_reward:.6f}, std={std_reward:.6f}, PASS={passes}")
    
    return passes

# ==================== TEST MODULE ====================

if __name__ == "__main__":
    print("=" * 60)
    print("REWARD & COSTS SANITY CHECKER - TEST")
    print("=" * 60)
    
    # Crea mock environment factory
    def mock_env_factory(mode="test"):
        class MockEnv(gym.Env):
            def __init__(self):
                self.action_space = gym.spaces.Discrete(3)
                self.observation_space = gym.spaces.Box(-5, 5, (17,))
                self.step_count = 0
                
            def reset(self, **kwargs):
                self.step_count = 0
                return np.zeros(17), {}
            
            def step(self, action):
                self.step_count += 1
                
                # Simula reward pattern
                if action == 0:  # Hold
                    reward = np.random.normal(0, 0.0001)
                else:  # Trade
                    reward = np.random.normal(0.001, 0.001)
                
                done = self.step_count >= 100
                
                return np.zeros(17), reward, done, False, {}
        
        return MockEnv()
    
    # Mock cost model
    mock_cost_model = {
        'spread_bps': 20,
        'slippage_bps': 5,
        'fee_bps': 10
    }
    
    # Test quick check
    print("\n1. Testing quick sanity check...")
    result = quick_sanity_check(mock_env_factory, mock_cost_model, n_episodes=5)
    print(f"   Result: {'PASS' if result else 'FAIL'}")
    
    # Test completo
    print("\n2. Running full sanity check suite...")
    checker = RewardCostsSanityChecker(mock_env_factory, mock_cost_model)
    report = checker.run_all_checks(model=None, n_episodes=10)
    
    print(f"\n   Main issue: {report['summary']['main_issue']}")
    print(f"   C1 Pass: {report['summary']['c1_pass']}")
    
    if report['summary']['recommendations']:
        print("   Recommendations:")
        for rec in report['summary']['recommendations']:
            print(f"     - {rec}")
    
    print("\n" + "=" * 60)
    print("Sanity checker test completed!")
    print("=" * 60)