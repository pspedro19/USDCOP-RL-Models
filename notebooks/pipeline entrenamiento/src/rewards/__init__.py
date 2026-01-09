"""
USD/COP RL Trading System - Reward Functions
=============================================

Módulo de reward functions mejoradas con curriculum learning.

Opciones disponibles:
- SymmetricCurriculumReward: Reward simétrica con curriculum (original V19)
- AlphaCurriculumReward: Reward basada en alpha, no penaliza HOLD
- AlphaCurriculumRewardV2: Optimizada para 15min con bonus de tendencia
"""

from .symmetric_curriculum import (
    SymmetricCurriculumReward,
    CurriculumCostScheduler,
    SymmetryTracker,
    PathologicalBehaviorDetector,
    OnlineSortinoCalculator,
)

from .alpha_curriculum import (
    AlphaCurriculumReward,
    AlphaCurriculumRewardV2,
    CurriculumPhase,
)


def get_reward_function(reward_type: str, **kwargs):
    """
    Factory para obtener la reward function por nombre.

    Args:
        reward_type: 'symmetric', 'alpha', o 'alpha_v2'
        **kwargs: Argumentos para la reward function
            - total_timesteps o total_steps: Total de pasos de entrenamiento
            - Para symmetric: config (RewardConfig), total_timesteps
            - Para alpha/alpha_v2: final_cost, total_steps, phase_boundaries, verbose, etc.

    Returns:
        Instancia de la reward function
    """
    # Normalizar nombres de parámetros
    # SymmetricCurriculumReward usa 'total_timesteps'
    # AlphaCurriculumReward usa 'total_steps'
    total_steps = kwargs.pop('total_steps', None)
    total_timesteps = kwargs.pop('total_timesteps', None)

    # Usar el valor que esté disponible
    steps = total_timesteps or total_steps or 200_000

    if reward_type == 'symmetric':
        # SymmetricCurriculumReward acepta: config (optional), total_timesteps
        config = kwargs.pop('config', None)
        # Ignorar parámetros no soportados
        kwargs.pop('verbose', None)
        return SymmetricCurriculumReward(config=config, total_timesteps=steps)
    elif reward_type == 'alpha':
        return AlphaCurriculumReward(total_steps=steps, **kwargs)
    elif reward_type == 'alpha_v2':
        return AlphaCurriculumRewardV2(total_steps=steps, **kwargs)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}. Use 'symmetric', 'alpha', or 'alpha_v2'")


__all__ = [
    # Symmetric Curriculum
    'SymmetricCurriculumReward',
    'CurriculumCostScheduler',
    'SymmetryTracker',
    'PathologicalBehaviorDetector',
    'OnlineSortinoCalculator',
    # Alpha Curriculum
    'AlphaCurriculumReward',
    'AlphaCurriculumRewardV2',
    'CurriculumPhase',
    # Factory
    'get_reward_function',
]
