"""
USD/COP RL Trading System - Reward Functions
=============================================

Módulo de reward functions mejoradas con curriculum learning.

Opciones disponibles:
- SymmetricCurriculumReward: Reward simétrica con curriculum (original V19)
"""

from .symmetric_curriculum import (
    SymmetricCurriculumReward,
    CurriculumCostScheduler,
    SymmetryTracker,
    PathologicalBehaviorDetector,
    OnlineSortinoCalculator,
)


def get_reward_function(reward_type: str, **kwargs):
    """
    Factory para obtener la reward function por nombre.

    Args:
        reward_type: 'symmetric'
        **kwargs: Argumentos para la reward function

    Returns:
        Instancia de la reward function
    """
    total_steps = kwargs.pop('total_steps', None)
    total_timesteps = kwargs.pop('total_timesteps', None)
    steps = total_timesteps or total_steps or 200_000

    if reward_type == 'symmetric':
        config = kwargs.pop('config', None)
        kwargs.pop('verbose', None)
        return SymmetricCurriculumReward(config=config, total_timesteps=steps)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}. Use 'symmetric'")


__all__ = [
    'SymmetricCurriculumReward',
    'CurriculumCostScheduler',
    'SymmetryTracker',
    'PathologicalBehaviorDetector',
    'OnlineSortinoCalculator',
    'get_reward_function',
]
