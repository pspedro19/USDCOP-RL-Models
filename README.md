# USD/COP RL Trading System

Este repositorio contiene el sistema de trading algorítmico basado en Reinforcement Learning (RL) para el par USD/COP.

## 📊 Resumen de Resultados y Modelos

Basado en el análisis exhaustivo del proyecto (Enero 2026), a continuación se detallan los métricas de los modelos entrenados.

### 🏆 Modelo Principal en Producción

**PPO V1 (26 Dic 2025)**
*Ubicación: `models/ppo_v1_20251226_054154.zip`*

| Métrica | Valor |
|---------|-------|
| **Sharpe Ratio** | **2.92** |
| Max Drawdown | 0.69% |
| Win Rate | 44.85% |
| Timesteps | 80,000 |
| Tiempo entrenamiento | 6.2 min |
| Distribución Acciones | Long 31% / Hold 29% / Short 40% |

### 🧪 Experimento Anti-Bias (Reducción de Varianza)

*Objetivo: Reducir sesgo direccional y mejorar robustez mediante penalización por simetría y aumento de datos.*

**Resultados Clave:**
*   **Reducción de Varianza:** 54% (vs Baseline)
*   **Mejores Semillas:**
    *   **Seed 2042:** Sharpe **2.94**, Max DD 0.72%, Win Rate 43.1%. *Modelo más equilibrado / robusto.*
    *   **Seed 3042:** Sharpe **4.96** (Ensemble) / **3.70** (Anti-bias metric), Max DD 0.35%. *Mayor retorno, pero con mayor sesgo Short.*

**Comparativa de Seeds (Resultados Ensemble 50K steps):**
| Seed | Sharpe | Max DD | Win Rate | Nota |
|------|--------|--------|----------|------|
| 42 | 1.70 | 0.77% | 44.5% | |
| 1042 | 1.13 | 1.09% | 44.2% | Rendimiento bajo |
| **2042** | **2.94** | 0.72% | 43.1% | **Balanceado/Robusto** |
| **3042** | **4.96** | 0.35% | 44.5% | **Mejor Retorno** |
| 4042 | -0.05 | 0.90% | 42.0% | Fallido |

*Ensemble Combinado: Sharpe 2.39, Max DD 0.35%.*

---

## 🛠️ Metodología y Arquitectura

### Configuración del Modelo (PPO)
*   **Algoritmo:** PPO (Proximal Policy Optimization)
*   **Red Neuronal:** `net_arch: [256, 256]` (MlpPolicy)
*   **Learning Rate:** 0.0001
*   **N Steps:** 2048
*   **Batch Size:** 128
*   **Ent Coef:** 0.05 (Exploración)

### Environment (TradingEnvironmentV19)
*   **Balance Inicial:** $10,000
*   **Longitud Episodio:** 400 barras (~1 día de trading)
*   **Gestión de Riesgo:** Max Drawdown 15% (termina episodio)
*   **Features:** Volatility Scaling ON, Regime Detection ON.

### Datos (Dataset V19)
*   **Archivo:** `RL_DS3_MACRO_CORE.csv` (Mar 2020 - Dic 2025)
*   **Tamaño:** 84,671 barras (5 min)
*   **Split:** 70% Train / 15% Val / 15% Test
*   **Features:** 15 variables (13 mercado + 2 estado)

---

## ⚙️ Pipelines Activos (Airflow DAGs)

| DAG | Función |
|-----|---------|
| `l0_ohlcv_realtime.py` | Ingesta de datos en tiempo real |
| `l1_feature_refresh.py` | Cálculo y actualización de features |
| `l5_multi_model_inference.py` | Inferencia de modelos en producción |
| `alert_monitor.py` | Monitoreo y alertas del sistema |

---

## 📝 Conclusiones y Observaciones

### Fortalezas
1.  **Alto Rendimiento:** Los mejores modelos alcanzan Sharpe Ratios entre 2.9 y 4.9.
2.  **Estabilidad Mejorada:** Las técnicas anti-bias lograron reducir la varianza entre entrenamientos en un 54%.
3.  **Infraestructura Completa:** Pipeline totalmente automatizado desde la ingesta hasta la inferencia.

### Riesgos y Desafíos
1.  **Inestabilidad de Entrenamiento:** Alta sensibilidad a la semilla aleatoria (Seed 1042 falló vs Seed 3042 sobresaliente).
2.  **Sesgo de Inactividad:** Algunos modelos tienden a mantener posiciones (Hold > 70%) excesivamente.
3.  **Validación:** Se requiere mayor validación con datos en vivo (Live Trading) para confirmar la robustez fuera del backtest.
