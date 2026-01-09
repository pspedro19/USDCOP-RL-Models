"""
USD/COP RL Trading System - Abstract Reward Function Base
=========================================================

Clase base abstracta para reward functions que implementa el patrón Strategy.

OBJETIVO:
- Definir interface comun para todas las reward functions
- Permitir intercambio facil de rewards en runtime
- Mantener compatibilidad con implementaciones existentes
- Facilitar testing y comparacion de estrategias de reward

PATRON: Strategy Pattern
- AbstractRewardFunction define la interface
- Cada reward concreta (Sortino, SymmetricCurriculum, etc.) hereda e implementa
- El environment/wrapper usa la interface abstracta

COMPATIBILIDAD:
- Las clases existentes pueden heredar SIN cambiar su comportamiento
- Solo necesitan agregar `: AbstractRewardFunction` a la herencia
- Todos los metodos ya implementados son compatibles

Author: Claude Code
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Any, Protocol, runtime_checkable
from enum import Enum
import numpy as np


# =============================================================================
# DATA CLASSES COMUNES
# =============================================================================

@dataclass
class RewardContext:
    """
    Contexto con toda la informacion necesaria para calcular reward.

    Encapsula todos los parametros que una reward function puede necesitar.
    Esto permite extender la interface sin romper implementaciones existentes.
    """
    portfolio_return: float
    market_return: float
    portfolio_value: float
    position: float
    prev_position: float
    volatility_percentile: float = 0.5
    transaction_cost: float = 0.0
    step_count: int = 0
    episode_count: int = 0

    # Opcionales - no todas las rewards los usan
    spread: Optional[float] = None
    volume: Optional[float] = None
    atr: Optional[float] = None
    trend_strength: Optional[float] = None

    @property
    def trade_occurred(self) -> bool:
        """Detectar si hubo cambio de posicion significativo."""
        return abs(self.position - self.prev_position) > 0.5

    @property
    def position_direction(self) -> int:
        """Direccion de la posicion: 1=LONG, -1=SHORT, 0=HOLD."""
        if self.position > 0.1:
            return 1
        elif self.position < -0.1:
            return -1
        return 0


@dataclass
class RewardResult:
    """
    Resultado del calculo de reward.

    Estructura estandarizada que todas las rewards deben retornar.
    """
    total: float
    components: Dict[str, float]

    # Metadata opcional
    phase: Optional[str] = None
    progress: Optional[float] = None

    def to_tuple(self) -> Tuple[float, Dict[str, float]]:
        """Convertir a tuple para compatibilidad con interface existente."""
        result_dict = self.components.copy()
        result_dict['total'] = self.total
        if self.phase is not None:
            result_dict['phase'] = self.phase
        if self.progress is not None:
            result_dict['progress'] = self.progress
        return self.total, result_dict


# =============================================================================
# PROTOCOL PARA CURRICULUM LEARNING
# =============================================================================

@runtime_checkable
class CurriculumAware(Protocol):
    """
    Protocol para rewards que soportan curriculum learning.

    Permite verificar en runtime si una reward soporta curriculum:
        if isinstance(reward_fn, CurriculumAware):
            reward_fn.set_timestep(current_step)
    """

    def set_timestep(self, timestep: int) -> None:
        """Actualizar timestep para ajustar curriculum."""
        ...

    @property
    def progress(self) -> float:
        """Progreso del training (0.0 a 1.0)."""
        ...


# =============================================================================
# CLASE BASE ABSTRACTA
# =============================================================================

class AbstractRewardFunction(ABC):
    """
    Clase base abstracta para todas las reward functions.

    INTERFACE OBLIGATORIA:
    - reset(): Reiniciar estado para nuevo episodio
    - calculate(): Calcular reward dado el contexto

    INTERFACE OPCIONAL (con defaults):
    - get_stats(): Obtener estadisticas del reward
    - get_name(): Nombre de la reward function
    - get_config(): Configuracion actual

    EJEMPLO DE USO:
    ```python
    # Definir reward concreta
    class MyReward(AbstractRewardFunction):
        def reset(self, initial_balance: float = 10000):
            self.balance = initial_balance

        def calculate(self, portfolio_return, market_return, portfolio_value,
                     position, prev_position, **kwargs):
            reward = portfolio_return * 100
            return reward, {'pnl': reward}

    # Usar con Strategy Pattern
    env = TradingEnv(reward_function=MyReward())

    # Cambiar reward en runtime
    env.set_reward_function(AlternativeReward())
    ```

    COMPATIBILIDAD CON CLASES EXISTENTES:
    Las clases SortinoRewardFunction y SymmetricCurriculumReward ya implementan
    esta interface. Solo necesitan agregar la herencia:

    ```python
    # Antes:
    class SortinoRewardFunction:
        ...

    # Despues:
    class SortinoRewardFunction(AbstractRewardFunction):
        ...  # Sin cambios en el codigo
    ```
    """

    @abstractmethod
    def reset(self, initial_balance: float = 10000) -> None:
        """
        Reiniciar estado interno para nuevo episodio.

        Args:
            initial_balance: Balance inicial del portfolio
        """
        pass

    @abstractmethod
    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
        **kwargs,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calcular reward para el step actual.

        Args:
            portfolio_return: Return del portfolio este step
            market_return: Return raw del mercado
            portfolio_value: Valor actual del portfolio
            position: Posicion actual (-1 a +1)
            prev_position: Posicion anterior
            volatility_percentile: Percentil de volatilidad (0 a 1)
            **kwargs: Argumentos adicionales (transaction_cost, etc.)

        Returns:
            Tuple de (reward_total, dict_componentes)

        El dict de componentes debe incluir al menos:
        - 'total': Reward total (igual al primer valor del tuple)

        Puede incluir componentes adicionales como:
        - 'pnl': Componente de profit/loss
        - 'cost': Penalizacion por costos
        - 'sortino': Componente basado en Sortino
        - 'direction': Reward por direccion correcta
        - etc.
        """
        pass

    def get_stats(self) -> Dict[str, float]:
        """
        Obtener estadisticas acumuladas del reward.

        Returns:
            Dict con estadisticas como mean, std, etc.

        Default: Dict vacio si no se implementa.
        """
        return {}

    def get_name(self) -> str:
        """
        Nombre descriptivo de la reward function.

        Returns:
            Nombre de la clase por default.
        """
        return self.__class__.__name__

    def get_config(self) -> Dict[str, Any]:
        """
        Obtener configuracion actual de la reward.

        Returns:
            Dict con parametros de configuracion.
        """
        if hasattr(self, 'config') and self.config is not None:
            if hasattr(self.config, '__dict__'):
                return self.config.__dict__.copy()
            return dict(self.config)
        return {}

    def calculate_from_context(self, ctx: RewardContext) -> RewardResult:
        """
        Calcular reward usando RewardContext.

        Metodo de conveniencia que desempaqueta el contexto.

        Args:
            ctx: RewardContext con toda la informacion

        Returns:
            RewardResult con total y componentes
        """
        total, components = self.calculate(
            portfolio_return=ctx.portfolio_return,
            market_return=ctx.market_return,
            portfolio_value=ctx.portfolio_value,
            position=ctx.position,
            prev_position=ctx.prev_position,
            volatility_percentile=ctx.volatility_percentile,
            transaction_cost=ctx.transaction_cost,
        )
        return RewardResult(total=total, components=components)


# =============================================================================
# CLASE BASE PARA CURRICULUM REWARDS
# =============================================================================

class AbstractCurriculumReward(AbstractRewardFunction, CurriculumAware):
    """
    Extension de AbstractRewardFunction para rewards con curriculum learning.

    Agrega soporte obligatorio para:
    - Tracking de timesteps
    - Fases de entrenamiento
    - Progreso del curriculum

    FASES TIPICAS:
    - EXPLORATION: Aprender movimientos basicos
    - TRANSITION: Introducir costos gradualmente
    - REALISTIC: Costos y condiciones reales
    """

    def __init__(self, total_timesteps: int = 500_000):
        self._total_timesteps = total_timesteps
        self._current_timestep = 0

    @property
    def total_timesteps(self) -> int:
        """Total de timesteps del training."""
        return self._total_timesteps

    @total_timesteps.setter
    def total_timesteps(self, value: int):
        self._total_timesteps = value

    @property
    def current_timestep(self) -> int:
        """Timestep actual."""
        return self._current_timestep

    def set_timestep(self, timestep: int) -> None:
        """Actualizar timestep actual."""
        self._current_timestep = timestep

    @property
    def progress(self) -> float:
        """Progreso del training (0.0 a 1.0)."""
        if self._total_timesteps <= 0:
            return 1.0
        return min(self._current_timestep / self._total_timesteps, 1.0)

    @abstractmethod
    def get_phase(self) -> Any:
        """
        Obtener fase actual del curriculum.

        Returns:
            Fase actual (puede ser Enum o string)
        """
        pass


# =============================================================================
# REWARD FUNCTION REGISTRY (FACTORY PATTERN)
# =============================================================================

class RewardRegistry:
    """
    Registro de reward functions disponibles.

    Permite registrar y obtener rewards por nombre, facilitando
    configuracion desde archivos YAML/JSON.

    EJEMPLO:
    ```python
    # Registrar una nueva reward
    RewardRegistry.register('my_reward', MyRewardClass)

    # Crear instancia por nombre
    reward = RewardRegistry.create('sortino', window_size=60)

    # Listar disponibles
    print(RewardRegistry.list_available())
    ```
    """

    _registry: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, reward_class: type) -> None:
        """Registrar una reward function."""
        if not issubclass(reward_class, AbstractRewardFunction):
            raise TypeError(
                f"{reward_class.__name__} debe heredar de AbstractRewardFunction"
            )
        cls._registry[name.lower()] = reward_class

    @classmethod
    def create(cls, name: str, **kwargs) -> AbstractRewardFunction:
        """Crear instancia de reward por nombre."""
        name_lower = name.lower()
        if name_lower not in cls._registry:
            available = ', '.join(cls._registry.keys())
            raise ValueError(
                f"Reward '{name}' no encontrada. Disponibles: {available}"
            )
        return cls._registry[name_lower](**kwargs)

    @classmethod
    def list_available(cls) -> list:
        """Listar nombres de rewards disponibles."""
        return list(cls._registry.keys())

    @classmethod
    def get_class(cls, name: str) -> type:
        """Obtener clase de reward por nombre."""
        return cls._registry.get(name.lower())


# =============================================================================
# DECORATOR PARA AUTO-REGISTRO
# =============================================================================

def register_reward(name: str):
    """
    Decorator para auto-registrar reward functions.

    EJEMPLO:
    ```python
    @register_reward('sortino')
    class SortinoReward(AbstractRewardFunction):
        ...

    # Ahora se puede crear con:
    reward = RewardRegistry.create('sortino')
    ```
    """
    def decorator(cls):
        RewardRegistry.register(name, cls)
        return cls
    return decorator


# =============================================================================
# COMPOSITE REWARD (PATRON COMPOSITE)
# =============================================================================

class CompositeReward(AbstractRewardFunction):
    """
    Reward compuesta de multiples sub-rewards.

    Permite combinar varias reward functions con pesos.

    EJEMPLO:
    ```python
    composite = CompositeReward()
    composite.add_reward('sortino', SortinoReward(), weight=0.6)
    composite.add_reward('direction', DirectionReward(), weight=0.4)

    # Calcular combina todas
    total, components = composite.calculate(...)
    ```
    """

    def __init__(self):
        self._rewards: Dict[str, Tuple[AbstractRewardFunction, float]] = {}
        self._initial_balance = 10000

    def add_reward(
        self,
        name: str,
        reward: AbstractRewardFunction,
        weight: float = 1.0,
    ) -> 'CompositeReward':
        """
        Agregar sub-reward con peso.

        Returns:
            self para permitir chaining
        """
        self._rewards[name] = (reward, weight)
        return self

    def remove_reward(self, name: str) -> None:
        """Remover sub-reward por nombre."""
        if name in self._rewards:
            del self._rewards[name]

    def reset(self, initial_balance: float = 10000) -> None:
        """Reset todas las sub-rewards."""
        self._initial_balance = initial_balance
        for reward, _ in self._rewards.values():
            reward.reset(initial_balance)

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
        **kwargs,
    ) -> Tuple[float, Dict[str, float]]:
        """Calcular reward combinando todas las sub-rewards."""
        total_weighted = 0.0
        all_components = {}

        for name, (reward, weight) in self._rewards.items():
            sub_total, sub_components = reward.calculate(
                portfolio_return=portfolio_return,
                market_return=market_return,
                portfolio_value=portfolio_value,
                position=position,
                prev_position=prev_position,
                volatility_percentile=volatility_percentile,
                **kwargs,
            )

            total_weighted += sub_total * weight

            # Prefixear componentes con nombre de reward
            for key, value in sub_components.items():
                if key != 'total':
                    all_components[f'{name}_{key}'] = value

            all_components[f'{name}_subtotal'] = sub_total
            all_components[f'{name}_weight'] = weight

        all_components['total'] = total_weighted

        return total_weighted, all_components

    def get_stats(self) -> Dict[str, float]:
        """Obtener estadisticas de todas las sub-rewards."""
        stats = {}
        for name, (reward, _) in self._rewards.items():
            sub_stats = reward.get_stats()
            for key, value in sub_stats.items():
                stats[f'{name}_{key}'] = value
        return stats


# =============================================================================
# ADAPTERS PARA CLASES EXISTENTES
# =============================================================================

class SortinoRewardAdapter(AbstractRewardFunction):
    """
    Adapter para usar SortinoRewardFunction con la interface abstracta.

    Esto permite usar SortinoRewardFunction sin modificarla.

    EJEMPLO:
    ```python
    from src.sortino_reward import SortinoRewardFunction, SortinoConfig

    # Crear adapter
    adapter = SortinoRewardAdapter(SortinoConfig(window_size=60))

    # Usar con interface abstracta
    adapter.reset(10000)
    reward, components = adapter.calculate(...)
    ```
    """

    def __init__(self, config=None):
        # Import dinamico para evitar dependencia circular
        from src.sortino_reward import SortinoRewardFunction, SortinoConfig
        self._config = config or SortinoConfig()
        self._reward_fn = SortinoRewardFunction(self._config)

    def reset(self, initial_balance: float = 10000) -> None:
        self._reward_fn.reset(initial_balance)

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
        **kwargs,
    ) -> Tuple[float, Dict[str, float]]:
        return self._reward_fn.calculate(
            portfolio_return=portfolio_return,
            market_return=market_return,
            portfolio_value=portfolio_value,
            position=position,
            prev_position=prev_position,
            volatility_percentile=volatility_percentile,
            transaction_cost=kwargs.get('transaction_cost', 0.0),
        )

    def get_stats(self) -> Dict[str, float]:
        return self._reward_fn.get_stats()

    def get_config(self) -> Dict[str, Any]:
        if hasattr(self._config, '__dict__'):
            return self._config.__dict__.copy()
        return {}


class SymmetricCurriculumAdapter(AbstractCurriculumReward):
    """
    Adapter para usar SymmetricCurriculumReward con la interface abstracta.

    EJEMPLO:
    ```python
    from src.rewards.symmetric_curriculum import SymmetricCurriculumReward

    # Crear adapter
    adapter = SymmetricCurriculumAdapter(total_timesteps=500_000)

    # Usar con interface curriculum-aware
    adapter.set_timestep(100_000)
    adapter.reset(10000)
    reward, components = adapter.calculate(...)
    phase = adapter.get_phase()
    ```
    """

    def __init__(self, config=None, total_timesteps: int = 500_000):
        super().__init__(total_timesteps)
        # Import dinamico
        from src.rewards.symmetric_curriculum import SymmetricCurriculumReward
        self._reward_fn = SymmetricCurriculumReward(
            config=config,
            total_timesteps=total_timesteps,
        )

    def reset(self, initial_balance: float = 10000) -> None:
        self._reward_fn.reset(initial_balance)

    def set_timestep(self, timestep: int) -> None:
        super().set_timestep(timestep)
        self._reward_fn.set_timestep(timestep)

    @property
    def total_timesteps(self) -> int:
        return self._reward_fn.total_timesteps

    @total_timesteps.setter
    def total_timesteps(self, value: int):
        self._reward_fn.total_timesteps = value
        self._total_timesteps = value

    @property
    def progress(self) -> float:
        return self._reward_fn.progress

    def get_phase(self):
        return self._reward_fn.phase

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
        **kwargs,
    ) -> Tuple[float, Dict[str, float]]:
        return self._reward_fn.calculate(
            portfolio_return=portfolio_return,
            market_return=market_return,
            portfolio_value=portfolio_value,
            position=position,
            prev_position=prev_position,
            volatility_percentile=volatility_percentile,
        )

    def get_stats(self) -> Dict[str, float]:
        return self._reward_fn.get_stats()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core abstractions
    'AbstractRewardFunction',
    'AbstractCurriculumReward',
    'CurriculumAware',

    # Data classes
    'RewardContext',
    'RewardResult',

    # Patterns
    'CompositeReward',
    'RewardRegistry',
    'register_reward',

    # Adapters
    'SortinoRewardAdapter',
    'SymmetricCurriculumAdapter',
]


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("ABSTRACT REWARD FUNCTION - Strategy Pattern Test")
    print("=" * 70)

    # Test 1: Verificar que AbstractRewardFunction no se puede instanciar
    print("\n1. Test ABC (no instanciable):")
    try:
        AbstractRewardFunction()
        print("  ERROR: Deberia fallar!")
    except TypeError as e:
        print(f"  OK: {e}")

    # Test 2: Implementacion minima
    print("\n2. Test implementacion minima:")

    class SimpleReward(AbstractRewardFunction):
        def reset(self, initial_balance: float = 10000):
            self.balance = initial_balance

        def calculate(self, portfolio_return, market_return, portfolio_value,
                     position, prev_position, volatility_percentile=0.5, **kwargs):
            reward = portfolio_return * 100
            return reward, {'pnl': reward, 'total': reward}

    simple = SimpleReward()
    simple.reset(10000)
    reward, components = simple.calculate(
        portfolio_return=0.01,
        market_return=0.01,
        portfolio_value=10100,
        position=1.0,
        prev_position=0.0,
    )
    print(f"  Reward: {reward:.4f}")
    print(f"  Components: {components}")
    print(f"  Name: {simple.get_name()}")

    # Test 3: CompositeReward
    print("\n3. Test CompositeReward:")

    class DirectionReward(AbstractRewardFunction):
        def reset(self, initial_balance: float = 10000):
            pass

        def calculate(self, portfolio_return, market_return, portfolio_value,
                     position, prev_position, volatility_percentile=0.5, **kwargs):
            correct = (position > 0 and market_return > 0) or \
                      (position < 0 and market_return < 0)
            reward = 1.0 if correct else -1.0
            return reward, {'direction': reward, 'total': reward}

    composite = CompositeReward()
    composite.add_reward('pnl', SimpleReward(), weight=0.7)
    composite.add_reward('direction', DirectionReward(), weight=0.3)
    composite.reset(10000)

    reward, components = composite.calculate(
        portfolio_return=0.01,
        market_return=0.01,
        portfolio_value=10100,
        position=1.0,
        prev_position=0.0,
    )
    print(f"  Composite Reward: {reward:.4f}")
    print(f"  Components: {list(components.keys())}")

    # Test 4: Registry
    print("\n4. Test RewardRegistry:")
    RewardRegistry.register('simple', SimpleReward)
    RewardRegistry.register('direction', DirectionReward)

    print(f"  Registered: {RewardRegistry.list_available()}")

    created = RewardRegistry.create('simple')
    print(f"  Created: {created.get_name()}")

    # Test 5: RewardContext
    print("\n5. Test RewardContext:")
    ctx = RewardContext(
        portfolio_return=0.01,
        market_return=0.01,
        portfolio_value=10100,
        position=1.0,
        prev_position=0.0,
    )
    print(f"  trade_occurred: {ctx.trade_occurred}")
    print(f"  position_direction: {ctx.position_direction}")

    result = simple.calculate_from_context(ctx)
    print(f"  RewardResult: total={result.total:.4f}")

    print("\n" + "=" * 70)
    print("AbstractRewardFunction ready for use!")
    print("=" * 70)
