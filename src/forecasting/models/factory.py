"""
Model Factory
=============

Factory pattern for creating ML models.
Single source of truth for model instantiation.

Supported Models:
- Linear: ridge, bayesian_ridge, ard
- Boosting: xgboost, lightgbm, catboost
- Hybrid: hybrid_xgboost, hybrid_lightgbm, hybrid_catboost

@version 1.0.0
"""

from typing import Dict, Any, Optional, List, Type
import logging

from src.forecasting.models.base import BaseModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """
    Factory class for creating ML models.

    Usage:
        model = ModelFactory.create('ridge', params={'alpha': 1.0})
        model = ModelFactory.create('xgboost', horizon=15)

    All models are registered lazily to avoid import errors
    when optional dependencies are missing.
    """

    _registry: Dict[str, Type[BaseModel]] = {}
    _initialized: bool = False

    @classmethod
    def _initialize_registry(cls) -> None:
        """Initialize model registry lazily."""
        if cls._initialized:
            return

        # Import model classes
        from src.forecasting.models.ridge import RidgeModel
        from src.forecasting.models.bayesian_ridge import BayesianRidgeModel
        from src.forecasting.models.xgboost import XGBoostModel
        from src.forecasting.models.lightgbm import LightGBMModel
        from src.forecasting.models.catboost import CatBoostModel

        # Register linear models
        cls._registry['ridge'] = RidgeModel
        cls._registry['bayesian_ridge'] = BayesianRidgeModel

        # Try to import ARD (may not be available)
        try:
            from src.forecasting.models.ard import ARDModel
            cls._registry['ard'] = ARDModel
        except ImportError:
            logger.warning("ARD model not available")

        # Register boosting models
        cls._registry['xgboost'] = XGBoostModel
        cls._registry['xgboost_pure'] = XGBoostModel
        cls._registry['lightgbm'] = LightGBMModel
        cls._registry['lightgbm_pure'] = LightGBMModel
        cls._registry['catboost'] = CatBoostModel
        cls._registry['catboost_pure'] = CatBoostModel

        # Try to import hybrid models
        try:
            from src.forecasting.models.hybrids import (
                XGBoostHybridModel,
                LightGBMHybridModel,
                CatBoostHybridModel,
            )
            cls._registry['hybrid_xgboost'] = XGBoostHybridModel
            cls._registry['hybrid_lightgbm'] = LightGBMHybridModel
            cls._registry['hybrid_catboost'] = CatBoostHybridModel
            cls._registry['xgb_hybrid'] = XGBoostHybridModel
            cls._registry['lgb_hybrid'] = LightGBMHybridModel
            cls._registry['cat_hybrid'] = CatBoostHybridModel
        except ImportError:
            logger.warning("Hybrid models not available")

        cls._initialized = True
        logger.info(f"ModelFactory initialized with {len(cls._registry)} models")

    @classmethod
    def create(
        cls,
        model_name: str,
        params: Optional[Dict[str, Any]] = None,
        horizon: Optional[int] = None,
    ) -> BaseModel:
        """
        Create a model instance by name.

        Args:
            model_name: Name of the model. Options:
                - Linear: 'ridge', 'bayesian_ridge', 'ard'
                - Boosting: 'xgboost', 'lightgbm', 'catboost'
                - Hybrid: 'hybrid_xgboost', 'hybrid_lightgbm', 'hybrid_catboost'
            params: Model hyperparameters (overrides defaults)
            horizon: Prediction horizon for adaptive parameters

        Returns:
            Instantiated model ready for training

        Raises:
            ValueError: If model_name is not recognized
        """
        cls._initialize_registry()

        # Normalize model name
        name_lower = model_name.lower().replace('-', '_')

        if name_lower not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown model: {model_name}. Available: {available}"
            )

        model_class = cls._registry[name_lower]

        # Create model instance
        model = model_class(name=model_name, params=params)

        logger.debug(f"Created {model_name} model with params: {params}")
        return model

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available model names."""
        cls._initialize_registry()
        return sorted(set(cls._registry.keys()))

    @classmethod
    def get_model_class(cls, model_name: str) -> Type[BaseModel]:
        """
        Get model CLASS (not instance) by name.

        Useful for Optuna optimization or type checking.

        Args:
            model_name: Name of the model

        Returns:
            Model class (not instance)
        """
        cls._initialize_registry()

        name_lower = model_name.lower().replace('-', '_')
        if name_lower not in cls._registry:
            raise ValueError(f"Unknown model: {model_name}")

        return cls._registry[name_lower]

    @classmethod
    def requires_scaling(cls, model_name: str) -> bool:
        """
        Check if a model requires feature scaling.

        Args:
            model_name: Name of the model

        Returns:
            True if model requires scaling (linear models)
        """
        cls._initialize_registry()

        name_lower = model_name.lower().replace('-', '_')
        if name_lower not in cls._registry:
            return False

        # Create temp instance to check property
        model_class = cls._registry[name_lower]
        temp_model = model_class(name=model_name)
        return temp_model.requires_scaling

    @classmethod
    def supports_early_stopping(cls, model_name: str) -> bool:
        """
        Check if a model supports early stopping.

        Args:
            model_name: Name of the model

        Returns:
            True if model supports early stopping
        """
        cls._initialize_registry()

        name_lower = model_name.lower().replace('-', '_')
        if name_lower not in cls._registry:
            return False

        model_class = cls._registry[name_lower]
        temp_model = model_class(name=model_name)
        return temp_model.supports_early_stopping

    @classmethod
    def get_model_type(cls, model_name: str) -> str:
        """
        Get model category.

        Returns:
            'linear', 'boosting', 'hybrid', or 'unknown'
        """
        name_lower = model_name.lower().replace('-', '_')

        if name_lower in {'ridge', 'bayesian_ridge', 'ard'}:
            return 'linear'
        elif 'hybrid' in name_lower:
            return 'hybrid'
        elif name_lower in {'xgboost', 'xgboost_pure', 'lightgbm', 'lightgbm_pure',
                           'catboost', 'catboost_pure'}:
            return 'boosting'
        else:
            return 'unknown'

    @classmethod
    def register(cls, name: str, model_class: Type[BaseModel]) -> None:
        """
        Register a custom model.

        Args:
            name: Model name to register
            model_class: Model class (must inherit from BaseModel)
        """
        cls._initialize_registry()
        cls._registry[name.lower()] = model_class
        logger.info(f"Registered custom model: {name}")
