# Plan Integral: Fix Trading Behavior - V5

**Fecha**: 2026-02-02
**Contexto**: Post-fix del distribution shift (rate_spread_z)
**Estado Actual**: Modelo balanceado (50% LONG, 50% SHORT) pero con comportamiento de trading deficiente

---

## 1. Diagnóstico del Problema

### 1.1 Síntomas Observados

| Síntoma | Valor Actual | Valor Esperado |
|---------|--------------|----------------|
| Trades por episodio | **1** | 5-15 |
| Longitud de episodio | **289 steps** | 600-1200 |
| PnL total (20 episodes) | **-$1,414** | > $0 |
| Distribución acciones | 50% L / 50% S | OK (balanceado) |
| HOLD % | **0.2%** | 10-30% |

### 1.2 Causas Raíz Identificadas

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ÁRBOL DE CAUSAS RAÍZ                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1 TRADE/EPISODIO                                                   │
│  ├── max_position_holding = 144 (termina episodio a ~289 steps)    │
│  ├── thresholds ±0.50 muy amplios (modelo debe estar 51% seguro)   │
│  └── holding_decay = 0.2 muy débil (no incentiva cambiar posición) │
│                                                                     │
│  PnL NEGATIVO                                                       │
│  ├── Trade cost (90 bps) > beneficio de posiciones cortas          │
│  ├── Solo 1 trade = sin oportunidad de recuperar pérdidas          │
│  └── Modelo mantiene posiciones perdedoras hasta forzar cierre     │
│                                                                     │
│  EPISODIOS CORTOS (~289 steps)                                      │
│  ├── max_position_holding = 144 fuerza cierre prematuro            │
│  └── max_episode_steps = 600 (pero nunca se alcanza)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 El Ciclo Vicioso Actual

```
Modelo abre posición (bar 1)
        ↓
Mantiene posición (holding_decay débil, no hay incentivo a cerrar)
        ↓
Alcanza max_position_holding (bar 144)
        ↓
Fuerza cierre + busca nueva posición
        ↓
Episodio termina (~bar 289)
        ↓
Solo 1 trade realizado
```

---

## 2. Solución Propuesta

### 2.1 Cambios en Environment Config

**Archivo**: `config/experiment_ssot.yaml`

```yaml
# ANTES (problemático)
environment:
  max_episode_steps: 600
  max_position_holding: 144
  thresholds:
    long: 0.50
    short: -0.50

# DESPUÉS (propuesto)
environment:
  max_episode_steps: 2000        # +233% - Permitir episodios más largos
  max_position_holding: 576      # +300% - 48 horas (2 días trading)
  thresholds:
    long: 0.25                   # -50% - Más fácil entrar LONG
    short: -0.25                 # -50% - Más fácil entrar SHORT
    # Zona HOLD: [-0.25, +0.25] = 25% del rango (antes era 50%)
```

**Justificación**:
- `max_episode_steps: 2000` → Permite 3-4 ciclos completos de posiciones
- `max_position_holding: 576` → 48 horas, suficiente para swing trades intraday
- `thresholds: ±0.25` → Modelo necesita solo 62.5% confianza (no 75%)

### 2.2 Cambios en Reward Config

**Archivo**: `config/experiment_ssot.yaml`

```yaml
# ANTES (problemático)
reward:
  pnl_weight: 0.8
  dsr_weight: 0.15
  sortino_weight: 0.05
  regime_penalty: 0.3
  holding_decay: 0.2       # MUY DÉBIL
  anti_gaming: 0.3

# DESPUÉS (propuesto)
reward:
  pnl_weight: 0.7           # -12.5% - Reducir para balancear con penalties
  dsr_weight: 0.15
  sortino_weight: 0.05
  regime_penalty: 0.4       # +33% - Penalizar trading contra tendencia
  holding_decay: 0.6        # +200% - CRÍTICO: Forzar rotación de posiciones
  anti_gaming: 0.4          # +33% - Evitar churn excesivo
```

**Justificación**:
- `holding_decay: 0.6` → Penalización fuerte por mantener posiciones demasiado tiempo
- `pnl_weight: 0.7` → Dejar espacio para que penalties tengan efecto
- Balance: `0.7 + 0.15 + 0.05 + 0.4 + 0.6 + 0.4 = 2.3` (suficiente penalización)

### 2.3 Cambios en Holding Decay Config

**Archivo**: `src/training/config.py` (clase `HoldingDecayConfig`)

```python
# ANTES
@dataclass(frozen=True)
class HoldingDecayConfig:
    half_life_bars: int = 48       # 4 horas
    max_penalty: float = 0.3
    flat_threshold: int = 0

# DESPUÉS
@dataclass(frozen=True)
class HoldingDecayConfig:
    half_life_bars: int = 72       # 6 horas (más gradual)
    max_penalty: float = 0.5       # 50% penalty máximo
    flat_threshold: int = 12       # 1 hora de gracia
```

**Justificación**:
- `half_life_bars: 72` → Decay más gradual, llega a 50% penalty en ~100 bars
- `max_penalty: 0.5` → Suficiente para superar pequeñas ganancias de PnL
- `flat_threshold: 12` → 1 hora sin penalty (permite que posición se desarrolle)

### 2.4 Opcional: Transaction Cost Reduction

Si el modelo sigue sin tradear activamente:

```yaml
environment:
  transaction_cost_bps: 3.0   # Reducir de 5.0 a 3.0
```

---

## 3. Plan de Implementación

### Fase 1: Cambios de Configuración (15 min)

| # | Archivo | Cambio | Prioridad |
|---|---------|--------|-----------|
| 1 | `experiment_ssot.yaml` | thresholds: ±0.25 | P0 |
| 2 | `experiment_ssot.yaml` | max_position_holding: 576 | P0 |
| 3 | `experiment_ssot.yaml` | max_episode_steps: 2000 | P0 |
| 4 | `experiment_ssot.yaml` | holding_decay: 0.6 | P0 |
| 5 | `src/training/config.py` | HoldingDecayConfig | P1 |

### Fase 2: Regenerar Dataset (5 min)

```bash
cd data/pipeline/06_rl_dataset_builder
python 01_build_5min_datasets.py
```

(Ya tiene rate_spread_z del fix anterior)

### Fase 3: Entrenar Modelo (35-40 min)

```bash
python scripts/run_full_pipeline.py --timesteps 500000
```

### Fase 4: Evaluar Resultados

**Criterios de Éxito**:

| Métrica | Mínimo Aceptable | Objetivo |
|---------|------------------|----------|
| Trades/episodio | ≥ 3 | 5-10 |
| Episode length | ≥ 500 | 1000+ |
| PnL (20 ep) | > -$500 | > $0 |
| Action balance | 30-70% L/S | 40-60% L/S |
| HOLD % | 5-40% | 15-25% |

---

## 4. Experimentos Alternativos

Si Fase 1-4 no logra los criterios, probar:

### Experimento A: Thresholds Más Agresivos
```yaml
thresholds:
  long: 0.15
  short: -0.15
  # Zona HOLD: solo 15% del rango
```

### Experimento B: Penalización de Holding Extrema
```yaml
reward:
  holding_decay: 0.8  # 80% penalty
```

### Experimento C: Reward Shaping para Trades
```yaml
# Agregar bonus por cambio de posición rentable
trade_profit_bonus: 0.3  # Bonus cuando cierra trade con ganancia
```

### Experimento D: Curriculum Más Agresivo
```python
# En config.py, CurriculumConfig
phase_1_steps: int = 50_000   # De 100_000
phase_2_steps: int = 150_000  # De 200_000
phase_3_steps: int = 250_000  # De 300_000
```

---

## 5. Métricas de Monitoreo

Durante entrenamiento, verificar en logs:

```
[Step X] Actions: LONG=Y%, HOLD=Z%, SHORT=W%
```

**Señales de Alerta**:
- HOLD < 5% → thresholds demasiado amplios
- HOLD > 50% → thresholds demasiado estrechos
- Episode length constante (~289) → max_position_holding sigue siendo limitante
- PnL consistentemente negativo → revisar transaction costs

---

## 6. Rollback Plan

Si los cambios empeoran los resultados:

```bash
# Restaurar configuración anterior
git checkout HEAD~1 -- config/experiment_ssot.yaml
git checkout HEAD~1 -- src/training/config.py
```

---

## 7. Resumen de Cambios

| Parámetro | Antes | Después | Cambio |
|-----------|-------|---------|--------|
| `max_episode_steps` | 600 | 2000 | +233% |
| `max_position_holding` | 144 | 576 | +300% |
| `threshold_long` | 0.50 | 0.25 | -50% |
| `threshold_short` | -0.50 | -0.25 | -50% |
| `holding_decay` | 0.2 | 0.6 | +200% |
| `pnl_weight` | 0.8 | 0.7 | -12.5% |
| `regime_penalty` | 0.3 | 0.4 | +33% |

---

## 8. Resultado Esperado

Después de implementar estos cambios:

```
=== EXPECTED BACKTEST (20 episodes) ===
Total steps analyzed: ~30,000 (vs 5,801 actual)
Avg episode length: ~1,500 steps (vs 289 actual)

Action Distribution:
  LONG:  40-50%
  HOLD:  15-25%
  SHORT: 30-40%

Trading Stats:
  Total Trades: 100-200 (vs 20 actual)
  Trades per episode: 5-10 (vs 1 actual)
  Total PnL: > $0 (vs -$1,414 actual)
```

---

**Autor**: Claude
**Versión**: 5.0
**Siguiente Paso**: Ejecutar Fase 1 (cambios de configuración)
