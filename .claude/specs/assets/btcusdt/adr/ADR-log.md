# ADR log — Integración BTC/USDT en el monorepo

> Decisiones arquitectónicas del **onboarding** (cómo BTC se enchufa al sistema). Las decisiones de
> la **estrategia en sí** (spot-only, combinación en riesgo, HMM congelado, contaminación LLM,
> retiro) están en [`../design/adr/`](../design/adr/) — ADR-0008…0012. No se duplican aquí.

| ADR | Decisión | Estado |
|---|---|---|
| ONB-01 | 5-min de BTC reusa `usdcop_m5_ohlcv` por `symbol`; daily reusa `asset_daily_ohlcv` (migr. 051). **NO** tabla nueva de OHLCV. | Aceptada |
| ONB-02 | Datos cripto-native (on-chain/funding/flows/eventos/señales) → migración **052** aditiva, keyed `(date/time,symbol)`. | Aceptada |
| ONB-03 | Sesión **24/7** modelada en el `AssetProfile` (`mode:24x7`), no con `if symbol=="BTC"` en el código. Anualización √365. | Aceptada |
| ONB-04 | Timezone de almacenamiento = **UTC** (carve-out de la golden rule COT en `data-governance.md`, igual que Gold). | Aceptada |
| ONB-05 | Sin `forced_close`; ejecución por **bandas de rebalanceo ±12.5%** = nuevo `ExecutionStrategy`, no `WeeklyTPHSExecution`. | Aceptada |
| ONB-06 | Provider canónico = **Binance/CCXT** (ADR-0008); `interim_provider: twelvedata` como fallback hasta que el extractor Binance exista. | Aceptada |
| ONB-07 | FRED se **reutiliza** (DFII10/DTWEXBGS/VIXCLS/M2SL ya ingeridas); solo se construyen extractores para lo cripto-native. | Aceptada |
| ONB-08 | Ruta de despliegue = **paper + web primero** (como Gold), difiriendo `forecast_h5_*` y OMS live. D6 al operador. | Aceptada |
| ONB-09 | Schedule de ingesta 24/7 vía **fábrica de pipelines** (no reusar la ventana de mercado COP 8-12 COT). | **Abierta** — D5, requiere operador |
| ONB-10 | Régimen: se conserva el Hurst gate del sistema **re-fit sobre BTC** (thresholds null en el perfil) además del HMM nativo de `design/`. Ambos re-fit, nada copiado de COP. | Aceptada |

## Contexto de las abiertas

- **ONB-09 (D5):** la ingesta 5-min de COP corre `*/5 13-17 UTC Mon-Fri` (mercado COP). BTC opera
  24/7 → necesita, como mínimo, una ingesta diaria post-cierre UTC 00:00 y, para el motor de 1h, una
  intradía continua. La fábrica de pipelines debe emitir un schedule cripto propio. No bloquea Fase 0
  (los seeds se pueden backfillear a mano vía el extractor); sí bloquea operación productiva.
