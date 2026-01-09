"""
USD/COP RL Trading System V11 - Training Callbacks
===================================================

Custom callbacks for PPO training.
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EntropyScheduler(BaseCallback):
    """
    Entropy coefficient scheduler.

    Linearly decreases entropy coefficient during training
    to encourage exploitation over exploration as training progresses.

    Parameters
    ----------
    init_ent : float
        Initial entropy coefficient
    final_ent : float
        Final entropy coefficient
    """

    def __init__(self, init_ent: float = 0.05, final_ent: float = 0.01):
        super().__init__()
        self.init_ent = init_ent
        self.final_ent = final_ent

    def _on_step(self) -> bool:
        progress = self.num_timesteps / self.model._total_timesteps
        new_ent = max(
            self.init_ent - progress * (self.init_ent - self.final_ent),
            self.final_ent
        )
        self.model.ent_coef = new_ent
        return True


class PositionMonitor(BaseCallback):
    """
    Position distribution monitor.

    Logs position statistics periodically during training
    to detect bias toward long/short positions.

    Parameters
    ----------
    freq : int
        Logging frequency (in timesteps)
    logger : optional
        Logger instance for output
    """

    def __init__(self, freq: int = 50000, logger=None):
        super().__init__()
        self.freq = freq
        self.positions = []
        self.logger = logger

    def _on_step(self) -> bool:
        if 'actions' in self.locals:
            self.positions.extend(self.locals['actions'].flatten().tolist())

        if self.num_timesteps % self.freq == 0 and len(self.positions) > 100:
            recent = np.array(self.positions[-10000:])
            msg = (
                f"[POS] {self.num_timesteps:,}: "
                f"mean={recent.mean():+.3f}, std={recent.std():.3f}"
            )
            if self.logger:
                self.logger.log(f"  {msg}")
            else:
                print(msg)

        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping based on reward threshold.

    Stops training if mean reward exceeds threshold.

    Parameters
    ----------
    reward_threshold : float
        Reward threshold to trigger early stopping
    check_freq : int
        Checking frequency (in timesteps)
    min_episodes : int
        Minimum episodes before checking
    """

    def __init__(
        self,
        reward_threshold: float = 100.0,
        check_freq: int = 10000,
        min_episodes: int = 100
    ):
        super().__init__()
        self.reward_threshold = reward_threshold
        self.check_freq = check_freq
        self.min_episodes = min_episodes
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if 'infos' in self.locals:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])

        if (
            self.num_timesteps % self.check_freq == 0
            and len(self.episode_rewards) >= self.min_episodes
        ):
            mean_reward = np.mean(self.episode_rewards[-100:])
            if mean_reward >= self.reward_threshold:
                print(f"Early stopping: mean reward {mean_reward:.2f}")
                return False

        return True
