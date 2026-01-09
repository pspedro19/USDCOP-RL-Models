# 📋 FASE 1: VALIDACIÓN Y DIAGNÓSTICO - INSTRUCCIONES

**Versión:** 1.0
**Fecha:** 2025-11-05
**Duración estimada:** 3-5 días
**Objetivo:** Confirmar hipótesis de problema raíz antes de invertir en soluciones

---

## 🎯 RESUMEN

Fase 1 añade 3 funciones de validación al proyecto:
1. **`validate_model_robust()`** - Evaluar modelo con 10 seeds diferentes
2. **`feature_importance_analysis()`** - Medir poder predictivo de features actuales
3. **`baseline_comparison()`** - Comparar vs estrategias simples (Buy&Hold, RSI, MA)

---

## 📁 ARCHIVOS CREADOS/MODIFICADOS

### **✅ ARCHIVO NUEVO CREADO:**

```
notebooks/utils/validation_fase1.py    [NUEVO - 3 funciones]
```

Este archivo contiene las 3 funciones nuevas y puede usarse de 2 formas:

**Opción A: Importar directamente**
```python
from utils.validation_fase1 import validate_model_robust, feature_importance_analysis, baseline_comparison
```

**Opción B: Copiar al archivo `validation.py` existente**
- Abrir `notebooks/utils/validation.py`
- Copiar las 3 funciones del archivo `validation_fase1.py`
- Pegarlas al final del archivo (después de la función `detect_overfitting()`)

---

## ✅ INSTRUCCIONES PARA AÑADIR CELDAS AL NOTEBOOK

**Archivo:** `notebooks/usdcop_rl_notebook.ipynb`

Necesitas añadir **3 celdas nuevas** en el notebook. Sigue estas instrucciones:

---

### **CELDA 6.5: Validación Robusta (10 Seeds)**

**Ubicación:** Después de la celda donde entrenas el modelo, antes de backtesting

**Código a añadir:**
```python
# ============================================================================
# CELDA 6.5: VALIDACIÓN ROBUSTA CON 10 SEEDS (FASE 1)
# ============================================================================

from utils.validation_fase1 import validate_model_robust

logger.header("VALIDACIÓN ROBUSTA - 10 SEEDS")

# Evaluar modelo entrenado con múltiples seeds
results_10seeds = validate_model_robust(
    model=agent_sb3,  # O el modelo que hayas entrenado
    env=env_val,      # Environment de validación
    n_seeds=10,
    n_eval_episodes=5,
    deterministic=True
)

# Guardar resultados
results_10seeds.to_csv('./outputs/validation_10seeds.csv', index=False)

# Visualización: Boxplot de Sharpe por seed
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sharpe distribution
axes[0].boxplot(results_10seeds['sharpe_ratio'])
axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero')
axes[0].set_ylabel('Sharpe Ratio')
axes[0].set_title('Distribución de Sharpe (10 seeds)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Win Rate distribution
axes[1].boxplot(results_10seeds['win_rate'] * 100)
axes[1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50%')
axes[1].set_ylabel('Win Rate (%)')
axes[1].set_title('Distribución de Win Rate (10 seeds)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/sharpe_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# Estadísticas
logger.info(f"Sharpe medio: {results_10seeds['sharpe_ratio'].mean():.2f} ± {results_10seeds['sharpe_ratio'].std():.2f}")
logger.info(f"Win Rate medio: {results_10seeds['win_rate'].mean()*100:.1f}%")

# DECISIÓN
sharpe_mean = results_10seeds['sharpe_ratio'].mean()
if sharpe_mean < 0.2:
    logger.error("❌ PROBLEMA ESTRUCTURAL confirmado → Proceder con FASE 2")
elif sharpe_mean > 0.5:
    logger.success("✅ Problema es HYPERPARAMETERS → Saltar a FASE 4")
else:
    logger.warning("⚠️  MARGINAL → Proceder con FASE 2 y FASE 3")
```

---

### **CELDA 6.6: Feature Importance Analysis**

**Ubicación:** Después de la celda 6.5

**Código a añadir:**
```python
# ============================================================================
# CELDA 6.6: FEATURE IMPORTANCE ANALYSIS (FASE 1)
# ============================================================================

from utils.validation_fase1 import feature_importance_analysis

logger.header("FEATURE IMPORTANCE ANALYSIS")

# Cargar datos L4 (con todas las features obs_XX)
# Asumiendo que df_train tiene las features
importance_df = feature_importance_analysis(
    df=df_train,
    target_col='close',
    n_forward_steps=5,  # Predecir 5 steps adelante (25 minutos)
    n_estimators=200,
    max_depth=8,
    test_size=0.2
)

# Guardar resultados
importance_df.to_csv('./outputs/feature_importance.csv', index=False)

# Visualización: Top 10 features
fig, ax = plt.subplots(figsize=(10, 6))

top_10 = importance_df.head(10)
ax.barh(top_10['feature_name'], top_10['importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Top 10 Features - RandomForest Importance')
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# Mostrar top 5
logger.info("Top 5 features más importantes:")
for idx, row in importance_df.head(5).iterrows():
    logger.info(f"  {row['rank']}. {row['feature_name']}: {row['importance']:.4f}")

# DECISIÓN
max_importance = importance_df['importance'].max()
if max_importance < 0.10:
    logger.error("❌ Features INSUFICIENTES → PROCEDER A FASE 2 URGENTE")
elif max_importance > 0.20:
    logger.success("✅ Features tienen señal → Problema no son features")
else:
    logger.warning("⚠️  Features MARGINALES → Proceder con FASE 2")
```

---

### **CELDA 6.7: Baseline Comparison**

**Ubicación:** Después de la celda 6.6

**Código a añadir:**
```python
# ============================================================================
# CELDA 6.7: BASELINE COMPARISON (FASE 1)
# ============================================================================

from utils.validation_fase1 import baseline_comparison

logger.header("COMPARACIÓN VS BASELINES")

# Ejecutar comparación (usa df_test o df_val)
baseline_results = baseline_comparison(
    df=df_test,  # Datos de test
    initial_balance=10000
)

# Añadir resultados del modelo RL (de celda 6.5)
rl_sharpe = results_10seeds['sharpe_ratio'].mean()
rl_return = results_10seeds['total_return_pct'].mean()
rl_winrate = results_10seeds['win_rate'].mean()
rl_maxdd = results_10seeds['max_drawdown_pct'].mean()

# Crear fila RL
rl_row = pd.DataFrame([{
    'strategy': 'RL (Current)',
    'sharpe': rl_sharpe,
    'return_pct': rl_return,
    'win_rate': rl_winrate,
    'max_drawdown_pct': rl_maxdd
}])

# Combinar
comparison_full = pd.concat([baseline_results, rl_row], ignore_index=True)

# Guardar
comparison_full.to_csv('./outputs/baseline_comparison.csv', index=False)

# Visualización: Radar chart
from math import pi

categories = ['Sharpe', 'Return %', 'Win Rate', 'Max DD (abs)']
N = len(categories)

# Normalizar métricas para radar chart (0-1)
def normalize_metric(series, reverse=False):
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return [0.5] * len(series)
    normalized = (series - min_val) / (max_val - min_val)
    if reverse:
        normalized = 1 - normalized
    return normalized.tolist()

# Preparar datos
strategies = comparison_full['strategy'].tolist()
values = {
    strategy: [
        normalize_metric(comparison_full['sharpe'])[idx],
        normalize_metric(comparison_full['return_pct'])[idx],
        normalize_metric(comparison_full['win_rate'])[idx],
        normalize_metric(comparison_full['max_drawdown_pct'].abs(), reverse=True)[idx]
    ]
    for idx, strategy in enumerate(strategies)
}

# Plot radar
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

for strategy, vals in values.items():
    vals += vals[:1]
    ax.plot(angles, vals, 'o-', linewidth=2, label=strategy)
    ax.fill(angles, vals, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Comparación de Estrategias (normalizado)', size=14, y=1.08)

plt.tight_layout()
plt.savefig('./outputs/baseline_radar.png', dpi=150, bbox_inches='tight')
plt.show()

# Mostrar tabla
logger.info("\nTabla comparativa:")
print(comparison_full.to_string(index=False))

# DECISIÓN
rl_sharpe = comparison_full[comparison_full['strategy'] == 'RL (Current)']['sharpe'].values[0]
best_baseline_sharpe = baseline_results['sharpe'].max()

if rl_sharpe > best_baseline_sharpe:
    logger.success(f"✅ RL supera baselines (RL: {rl_sharpe:.2f} vs Best: {best_baseline_sharpe:.2f})")
    logger.info("   → RL tiene señal, proceder con mejoras")
elif rl_sharpe > 0:
    logger.warning(f"⚠️  RL no supera todos los baselines pero es positivo")
    logger.info("   → Proceder con FASE 2 y FASE 3")
else:
    logger.error(f"❌ RL NO supera NINGÚN baseline (RL: {rl_sharpe:.2f})")
    logger.info("   → Problema SEVERO, revisar pipeline completo")
```

---

## 📊 OUTPUTS ESPERADOS

Después de ejecutar las 3 celdas, deberías tener:

### **Archivos CSV:**
```
outputs/
  ├── validation_10seeds.csv           [Métricas por seed]
  ├── feature_importance.csv           [Ranking de features]
  └── baseline_comparison.csv          [Comparación estrategias]
```

### **Gráficos PNG:**
```
outputs/
  ├── sharpe_distribution.png          [Boxplot Sharpe y Win Rate]
  ├── feature_importance.png           [Top 10 features]
  └── baseline_radar.png               [Radar chart comparación]
```

---

## 🎯 CRITERIOS DE DECISIÓN FASE 1

Después de ejecutar las 3 celdas, analiza los resultados:

| Criterio | Umbral Verde | Umbral Amarillo | Umbral Rojo |
|----------|-------------|----------------|-------------|
| **Sharpe (10 seeds mean)** | > 0.5 | 0.2 - 0.5 | < 0.2 |
| **Max Feature Importance** | > 0.20 | 0.10 - 0.20 | < 0.10 |
| **RL supera baselines** | 3/3 | 1-2/3 | 0/3 |

### **Decisión:**

**Escenario 1: VERDE (todos verdes)**
```
✅ Problema es HYPERPARAMETERS
→ Saltar directamente a FASE 4 (Optuna)
→ Features son suficientes, modelo tiene señal
```

**Escenario 2: AMARILLO (mixto)**
```
⚠️  Problema es FEATURES + MODELO
→ Proceder con FASE 2 (Features) + FASE 3 (SAC + Reward)
→ Necesita mejoras en múltiples frentes
```

**Escenario 3: ROJO (mayoría rojos)**
```
❌ Problema ESTRUCTURAL SEVERO
→ Proceder con FASE 2 URGENTE (añadir features macro/MTF)
→ Features actuales insuficientes para predecir
```

---

## 📋 CHECKLIST EJECUCIÓN FASE 1

- [ ] Archivo `validation_fase1.py` creado en `notebooks/utils/`
- [ ] Abrir notebook `notebooks/usdcop_rl_notebook.ipynb`
- [ ] Añadir Celda 6.5: Validación 10 seeds
- [ ] Añadir Celda 6.6: Feature importance
- [ ] Añadir Celda 6.7: Baseline comparison
- [ ] Ejecutar las 3 celdas
- [ ] Verificar que los 6 archivos de output se crean correctamente
- [ ] Analizar resultados con criterios de decisión
- [ ] Documentar decisión en `reports/semana1_diagnostico.md`

---

## 📝 CREAR REPORTE FASE 1

Al final, crear archivo: `reports/semana1_diagnostico.md`

**Template:**
```markdown
# FASE 1: DIAGNÓSTICO - RESULTADOS

**Fecha:** [FECHA]
**Ejecutado por:** [NOMBRE]

## Validación 10 Seeds

- Sharpe medio: [X.XX] ± [X.XX]
- Win Rate medio: [XX.X]%
- Return medio: [±X.XX]%

## Feature Importance

- Max importance: [X.XX]
- R² Test: [X.XX]
- Top 5 features:
  1. [obs_XX]
  2. [obs_XX]
  3. [obs_XX]
  4. [obs_XX]
  5. [obs_XX]

## Baseline Comparison

| Estrategia | Sharpe | Return % | Win Rate |
|------------|--------|----------|----------|
| Buy-Hold   | [X.XX] | [±X.XX]% | [XX.X]%  |
| RSI        | [X.XX] | [±X.XX]% | [XX.X]%  |
| MA Cross   | [X.XX] | [±X.XX]% | [XX.X]%  |
| RL Current | [X.XX] | [±X.XX]% | [XX.X]%  |

## DECISIÓN GO/NO-GO

**Status:** [VERDE / AMARILLO / ROJO]

**Próximos pasos:**
- [ ] [Acción 1]
- [ ] [Acción 2]

**Justificación:**
[Explicar por qué se toma la decisión]
```

---

## ⚠️ TROUBLESHOOTING

### **Error: "ModuleNotFoundError: No module named 'sklearn'"**

**Solución:**
```bash
pip install scikit-learn
```

---

### **Error: "NameError: name 'agent_sb3' is not defined"**

**Causa:** Variable del modelo no coincide con tu notebook

**Solución:** Reemplazar `agent_sb3` con el nombre de tu variable del modelo entrenado

---

### **Warning: "Features no predicen forward returns (R² muy bajo)"**

**Esperado:** Si R² < 0.05, las features actuales tienen poco poder predictivo

**Acción:** Confirma necesidad de FASE 2 (más features)

---

### **Gráficos no se muestran**

**Solución:**
```python
# Añadir al inicio del notebook
%matplotlib inline
```

---

## ➡️ PRÓXIMOS PASOS

Una vez completada Fase 1 (todos los outputs generados):

### **Si DECISIÓN = VERDE:**
1. Saltar a **Fase 4: Optuna Optimization**
   - Leer: `PLAN_ESTRATEGICO_v2_UPDATES.md` Sección 4
   - Archivo: `notebooks/utils/optimization.py`

### **Si DECISIÓN = AMARILLO o ROJO:**
1. **Fase 2: L3/L4 Feature Engineering**
   - Leer: `PLAN_ESTRATEGICO_v2_UPDATES.md` Sección 2
   - Modificar: `airflow/dags/usdcop_m5__04_l3_feature.py`
   - Modificar: `airflow/dags/usdcop_m5__05_l4_rlready.py`

2. **Fase 3: Reward Shaping + SAC**
   - Leer: `ADDENDUM_REWARD_SHAPING.md`
   - Crear: `notebooks/utils/rewards.py`
   - Modificar: `notebooks/utils/environments.py`

---

## 📚 REFERENCIAS

### **Papers citados:**

1. **Walk-Forward Analysis:** Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies"

2. **Feature Importance:** Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32

3. **Baseline Comparison:** Faber, M. (2007). "A Quantitative Approach to Tactical Asset Allocation"

### **Archivos relacionados:**

- `PLAN_ESTRATEGICO_MEJORAS_RL.md` (v1.0) - Plan original
- `PLAN_ESTRATEGICO_v2_UPDATES.md` - Plan con gaps integrados
- `RESUMEN_EJECUTIVO_v2.md` - Overview completo

---

## 📞 SOPORTE

**Logs útiles:**
```python
# Verificar features cargadas
print(df_train.columns[df_train.columns.str.startswith('obs_')].tolist())

# Verificar shape del environment
print(env_val.observation_space.shape)

# Verificar modelo cargado
print(type(agent_sb3))
```

---

**FIN DE INSTRUCCIONES FASE 1**

*Versión 1.0 - 2025-11-05*
*Próximo: Fase 2 (L3/L4 Feature Engineering) o Fase 4 (Optuna) según decisión*
