---
name: weekly-verify
description: Run the weekly operator verification for the trading system — confirm training, signal, backups and guardrails after the Sunday/Monday cycle. Use on Monday mornings, after a suspected DAG failure, or when asked whether the weekly cycle completed cleanly.
---

# Weekly verification

Los DAGs ya corren solos. Lo que sigue siendo manual es **verificar que corrieron bien** e
interpretar los guardrails.

## Checklist

```bash
for d in forecast_h5_l3_weekly_training forecast_h5_l5_weekly_signal \
         forecast_h5_l6_weekly_monitor core_l0_05_seed_backup; do
  echo "== $d"; airflow dags list-runs -d $d --limit 2
done
```

- [ ] **H5-L3 training** completó el domingo
- [ ] **H5-L5 señal** generada el lunes (depende de L3 vía sensor)
- [ ] **Seed backup** fresco — `data/backups/seeds/backup_manifest.json`
- [ ] **MACRO_DAILY_CLEAN.parquet** regenerado
- [ ] Frescura dentro de umbral (→ skill `data-recovery` si no)

## Guardrails — interpretar, no solo mirar

| Guardrail | Disparo | Qué significa |
|---|---|---|
| Circuit breaker | 5 pérdidas seguidas **o** 12% DD | Parar y revisar, no ajustar parámetros |
| Long-insistence | >60% LONGs en 8 semanas | El gate puede estar sesgado |
| DA rodante | SHORT <55% o LONG <45% en 16 semanas | Deterioro de la señal |

**Un guardrail que dispara no se re-calibra para que deje de disparar.** Los umbrales de retiro
no se relajan estando en drawdown (`.claude/rules/quant-constitution.md` §5).

## Contexto que evita falsas alarmas

- **No hay backup de fin de semana** (`0 20 * * 1-5`). Un lunes, el backup del viernes es correcto.
- **El track H1 está PAUSADO a propósito.** Que sus DAGs no corran **no es un fallo**.
  No los reactives.
- El gate de régimen **bloqueando semanas es el comportamiento deseado**: bloqueó 11 de 12 semanas
  mean-reverting en Q1 2026 y convirtió -5.17% en +0.61%. Pocas operaciones ≠ sistema roto.
- v11 está **CONGELADA**: el forward 2026 es su único juez honesto. No propongas re-tunearla con
  datos de 2025.

## Reportar

Estado de cada ítem, con evidencia (timestamps y run ids), y una recomendación explícita:
todo bien / necesita recovery / necesita atención humana.
