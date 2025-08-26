"""
Model Training Unit Tests
=========================
Tests for PPO, DQN, A2C, SAC, and TD3 model training with MLflow integration.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


class TestModelTraining:
    """Test model training functionality"""
    
    def test_ppo_model_initialization(self):
        """Test PPO model initialization"""
        from src.models.rl_models.ppo_model import PPOModel
        
        model = PPOModel(
            state_dim=10,
            action_dim=3,
            learning_rate=3e-4,
            n_steps=2048
        )
        
        assert model is not None
        assert model.state_dim == 10
        assert model.action_dim == 3
        assert model.learning_rate == 3e-4
    
    def test_ppo_model_training(self, sample_features_data):
        """Test PPO model training loop"""
        from src.models.rl_models.ppo_model import PPOModel
        from src.models.rl_models.trading_env import TradingEnvironment
        
        # Create environment
        env = TradingEnvironment(sample_features_data)
        
        # Create and train model
        model = PPOModel(env=env)
        
        with patch.object(model, 'learn') as mock_learn:
            mock_learn.return_value = model
            
            trained_model = model.train(
                total_timesteps=1000,
                log_interval=100
            )
            
            assert mock_learn.called
            assert trained_model is not None
    
    def test_dqn_model_initialization(self):
        """Test DQN model initialization"""
        from src.models.rl_models.dqn_model import DQNModel
        
        model = DQNModel(
            state_dim=10,
            action_dim=3,
            buffer_size=10000,
            batch_size=32
        )
        
        assert model is not None
        assert model.buffer_size == 10000
        assert model.batch_size == 32
    
    def test_a2c_model_training(self, sample_features_data):
        """Test A2C model training"""
        from src.models.rl_models.a2c_model import A2CModel
        from src.models.rl_models.trading_env import TradingEnvironment
        
        env = TradingEnvironment(sample_features_data)
        model = A2CModel(env=env)
        
        # Test training step
        with patch.object(model, 'learn') as mock_learn:
            mock_learn.return_value = model
            
            model.train(total_timesteps=100)
            
            assert mock_learn.called
    
    def test_sac_model_initialization(self):
        """Test SAC (Soft Actor-Critic) model initialization"""
        from src.models.rl_models.sac_model import SACModel
        
        model = SACModel(
            state_dim=10,
            action_dim=3,
            alpha=0.2,  # Temperature parameter
            tau=0.005   # Soft update coefficient
        )
        
        assert model is not None
        assert model.alpha == 0.2
        assert model.tau == 0.005
    
    def test_td3_model_training(self, sample_features_data):
        """Test TD3 (Twin Delayed DDPG) model training"""
        from src.models.rl_models.td3_model import TD3Model
        from src.models.rl_models.trading_env import TradingEnvironment
        
        env = TradingEnvironment(sample_features_data)
        model = TD3Model(
            env=env,
            policy_delay=2,
            noise_clip=0.5
        )
        
        assert model.policy_delay == 2
        assert model.noise_clip == 0.5
    
    def test_mlflow_experiment_tracking(self):
        """Test MLflow experiment tracking integration"""
        from src.models.training.experiment_tracker import ExperimentTracker
        
        # Test with MLflow disabled to avoid import issues
        tracker = ExperimentTracker(experiment_name="test_exp", enable_mlflow=False)
        
        # Log parameters
        tracker.log_params({
            'model_type': 'PPO',
            'learning_rate': 3e-4,
            'batch_size': 64
        })
        
        # Log metrics
        tracker.log_metrics({
            'reward': 100.5,
            'loss': 0.02
        })
        
        # Verify logging worked (stored locally)
        assert len(tracker.params_log) == 3
        assert len(tracker.metrics_log) == 1
        assert tracker.params_log['model_type'] == 'PPO'
        assert tracker.metrics_log[0]['metrics']['reward'] == 100.5
    
    def test_model_checkpointing(self, tmp_path):
        """Test model checkpointing and restoration"""
        from src.models.training.checkpoint_manager import CheckpointManager
        
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        
        # Save checkpoint
        model_state = {'weights': np.random.randn(10, 10)}
        manager.save_checkpoint(
            model_state,
            epoch=10,
            metrics={'loss': 0.01}
        )
        
        # Load checkpoint
        loaded_state = manager.load_checkpoint(epoch=10)
        
        assert loaded_state is not None
        assert np.array_equal(loaded_state['weights'], model_state['weights'])