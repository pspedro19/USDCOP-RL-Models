---
kind: roadmap
status: PLANNED
version: 2.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - config/strategy_manifests/usdcop.yaml
  - config/execution/smart_simple_v1.yaml
  - scripts/pipeline/train_and_export_smart_simple.py
  - airflow/dags/forward_ledger_weekly.py
---

# Plan de rentabilidad USD/COP

> Subplan del maestro
> `.claude/specs/audit/PLAN-RENTABILIDAD-CONSOLIDADO-2026-07.md`.
> Objetivo: juzgar honestamente `smart_simple_v11`, preservar su mecánica congelada y abrir
> carry solo si el costo real del broker demuestra que existe una prima capturable.

## 1. Estado que no se debe maquillar

- `smart_simple_v11` está congelada desde 2026-03-18. Su retraining expansivo semanal es
  parte del contrato; cambiar gate, features, stops, sizing o ejecución crea otra versión.
- El OOS-2025 fue usado durante la selección de la familia. El DSR trial-aware no demuestra
  edge de forma robusta; el p-value de la celda ganadora no repara esa selección.
- El registry declara hoy **58 trials centrales** y escenarios `[46, 58, 72]`; `sigma_trials`
  nunca se persistió. Es deuda de trazabilidad, no permiso para escoger la rejilla amable.
- La dirección desde precio queda cerrada: el zoo multi-horizonte, XLEAD y nuevas variantes
  no justifican otro barrido. El forward firmado es el juez.
- El ledger tiene una fila W30, pero todavía no una serie semanal comparable. Una cifra YTD
  no se resta contra un replay de una semana.

## 2. Contrato de estrategia y datos

| Elemento | Contrato congelado / uso |
|---|---|
| Decisión | semanal, máximo una posición por semana, reloj 52 |
| Señal | Ridge + Bayesian Ridge; XGBoost offline y fail-closed |
| Features | 25, scaling fit solo en train, macro `t−1`/`merge_asof(backward)` |
| Ejecución | sesión 08:00–12:55 `America/Bogota`; gestión M5 cada 30 min |
| Costos | maker/slippage del manifest más swap real medido; nunca asumir carry teórico |
| Retiro | `.claude/specs/assets/usdcop/WITHDRAWAL-PROTOCOL.md` |

Uso de las cuatro frecuencias:

| Tabla | Función en v11 | Regla |
|---|---|---|
| 5m | fills, HS/TP, spread, slippage y sesión | no re-modelar intradía; 60 barras válidas por sesión |
| 1h | diagnóstico de rango/liquidez | no entra a v11; incorporarla crea versión y trial |
| 4h | contexto de riesgo de sesión | misma prohibición que 1h |
| daily | features causales que agregan a la decisión semanal | disponibilidad macro y cierre conocidos antes de la señal |

## 3. Tesis operativa actual

La hipótesis defendible no es “el modelo predice COP”. Es:

1. el gate evita operar en regímenes adversos;
2. la mecánica TP/HS contiene la cola;
3. parte del retorno puede ser exposición/carry del peso;
4. el forward separará timing real de un baseline simple.

Por eso los baselines obligatorios son B1, B1′ de exposición emparejada, NULL-A/NULL-B del
registro y la mecánica sin modelo. Si W6 concluye que NULL-A ≥ v11, **NULL-A es la estrategia**.

## 4. Plan por dependencias

### U0 — Higiene y reproducibilidad (0 trials)

| ID | Acción | PASS |
|---|---|---|
| U0.1 | Reconciliar registry y artefactos | 58/[46,58,72] o cifra corregida con fuente; cuerpo y front matter idénticos |
| U0.2 | Persistir distribución de Sharpes de trials futuros | `sigma_trials` proviene de runs, no de supuesto; históricos conservan grid explícito |
| U0.3 | Verificar manifest vivo/export | Ridge+BR en config, export y serving; hash estructural en CI |
| U0.4 | Auditar PIT de las 25 features | `feature_available_at <= decision_at` en 100% de las filas utilizadas |
| U0.5 | Sanear ledger COP | `paper_week_pct`, replay de la misma estrategia/semana y missingness explícita |

### U1 — H-COP-CARRY-00: medir antes de modelar (0 trials)

El carry tradeable es el swap neto del instrumento real, no `IBR − fedfunds` en una tabla.

Artefacto mínimo por día de accrual:

- broker, cuenta, símbolo/contrato y lado;
- notional y moneda de cuenta;
- swap bruto, comisiones y conversión FX;
- carry teórico con tasas `available_at` y convención de días;
- ratio de pass-through neto y tratamiento de triple-swap/feriados.

**Gate prefirmado:** al menos 20 accruals válidos y pass-through mediano neto ≥50%, sin que
el IC95 de block bootstrap quede completamente por debajo de 50%. Si falla, H-CARRY-01 se
cancela sin gastar trial. No se cambia el umbral después de ver los statements.

### U2 — Construir el reloj correcto de cartera (0 trials)

`portfolio_daily.py` excluye COP por diseño. Antes de carry se implementa conceptualmente:

1. retorno realizado del libro diario XAU/BTC/SPX agregado por ISO week;
2. retorno de v11 en la misma semana, sin fabricar días intermedios;
3. overlay semanal con reloj 52, efectivo y costos separados;
4. tests que prohíben forward-fill, cero por missing y mezcla 252/365/52.

Sin este overlay, un “tilt de carry sobre ERC” es una categoría imposible y no se ejecuta.

### U3 — H-CARRY-01, condicional (+1 trial al abrir resultados)

Se pre-registra después de pasar U1 y U2:

- **Una variable:** multiplicador de exposición COP por carry conocido `t−1`, acotado
  `[0.5, 1.5]`; v11 permanece intacta y la candidata recibe nuevo `strategy_id`.
- **Priors:** diferencial/forward points y ventana se fijan sin mirar el juez; no hay grid.
- **Baselines:** v11 congelada, overlay sin tilt, exposición emparejada y tilt aleatorio con
  igual distribución; costos ×1/×2/×3 y swap real.
- **Juez:** forward posterior al freeze de la candidata. **2025 ya fue observado y no puede
  volver a llamarse OOS** para H-CARRY-01.
- **PASS económico:** delta Calmar vs el mejor baseline con IC95 >0, DD dentro del protocolo
  y costos ×2 positivos. Un claim de alfa además requiere DSR >0.95 con N actualizado.
- **Muestra:** con N<20 trades se reportan solo conteo y PnL; el veredicto económico espera
  52 semanas o ≥30 oportunidades, lo que ocurra después.

FAIL deja v11 sin cambios y cierra carry; no habilita otra búsqueda.

### U4 — El juez de v11 (0 trials nuevos)

- **Corte A:** 2026-09-16, 26 semanas.
- **Corte B:** 2027-03-17, 52 semanas.
- Se aplican W1–W6 sin reparametrizar en drawdown.
- Cada corte compara exactamente los mismos hashes y costos. Un bug técnico se separa de un
  fallo económico; corregir un bug no reescribe filas forward.

## 5. Explicitamente fuera de alcance

- reentrenar dirección a siete semanas;
- usar 1h/4h/M5 como votos direccionales;
- reabrir XLEAD, funding u otro zoo porque una gráfica se vea mejor;
- aplicar el carry COP al ERC diario que no contiene COP;
- llamar “rentable” a v11 antes del Corte B o de un retiro anticipado concluyente.

## 6. Criterios de cierre y verificación

```powershell
python scripts/pipeline/train_and_export_smart_simple.py --phase backtest --no-png
python scripts/pipeline/normalize_champions.py --check
python -m scripts.analysis.forward_tracker --report
python -m pytest tests/regression/test_usdcop_ensemble_consistency.py -q
```

Terminado significa: features PIT, swap medido, ledger semanal válido, v11 juzgada por el
protocolo y carry ejecutado como máximo una vez tras sus gates. No significa crear v12 por
anticipación.
