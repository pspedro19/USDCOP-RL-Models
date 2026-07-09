# Architecture Decision Records (ADR log)

Registro de decisiones de diseño no obvias. Formato: Contexto → Decisión → Consecuencias. Para cambiar una decisión, añade un nuevo ADR que supersede al anterior.

---

## ADR-001 — Timeframes H1 (ejecución) + Daily (régimen), nada intradía puro
**Contexto:** el edge del oro (tasas reales, DXY, safe-haven) vive en horizonte diario; RL necesita muchas muestras. Dos fuerzas opuestas.
**Decisión:** H1 como base de ejecución (~47k barras 2004–2025, entrenable con señal), Daily por encima para régimen y dirección. Prohibido < M15 en sistemático.
**Consecuencias:** signal-to-noise razonable sin explosión de overfitting; costos manejables; se evita la carrera de latencia/microestructura sin edge de ejecución. Régimen se computa en Daily (evoluciona en días, no horas).

---

## ADR-002 — La capa de riesgo vive DENTRO del entorno de entrenamiento
**Contexto:** si se entrena el agente con PnL "crudo" y se añade el riesgo (vol-targeting, blackouts, límites) solo en producción, se entrena sobre una distribución distinta a la que opera.
**Decisión:** vol-targeting, blackouts, weekend flat y breakers viven dentro de `GoldTradingEnv`. El agente ve el PnL real post-sizing/post-costos. Separación estricta: agente decide dirección, riesgo decide tamaño.
**Consecuencias:** elimina train/serve skew; el mismo motor corre en backtest, paper y live. Coste: el entorno es más complejo, pero auditable y determinista.

---

## ADR-003 — Relaciones macro como correlaciones rodantes, no supuestos fijos
**Contexto:** la relación inversa oro–DXY / oro–tasas reales NO es estable: 2022–2025 la rompió (oro subió con tasas reales altas por compras de bancos centrales). Codificar "DXY sube ⇒ oro baja" es codificar un régimen muerto.
**Decisión:** pasar la relación como **feature medida** (correlación rodante 60–120d oro–DXY y oro–tasas reales) al estado del agente. Nunca como filtro duro.
**Consecuencias:** el agente aprende *cuándo* la correlación importa según su valor vigente; el sistema se adapta a cambios estructurales de la relación macro.

---

## ADR-004 — Dos baselines obligatorios, no uno
**Contexto:** el oro tuvo un rally secular en 2024–2025; cualquier sesgo long parece genio en ese tramo. Un solo baseline no distingue "el agente aporta" de "el oro subió".
**Decisión:** B1 long-only vol-targeted (captura beta del oro bien gestionada) + B2 trend-follower Daily (sistema simple honesto). El RL debe ganar a AMBOS OOS, ajustado por riesgo, con la mediana de ≥5 seeds.
**Consecuencias:** gate honesto que separa beta de alpha y complejidad justificada de complejidad decorativa. Si B2 no gana a B1, no hay edge direccional ni a nivel de reglas.

---

## ADR-005 — Multi-seed con mediana/IQR, no el mejor run
**Contexto:** PPO tiene varianza enorme entre semillas; reportar el mejor seed es multiple testing disfrazado (exactamente lo que castiga el Deflated Sharpe).
**Decisión:** ≥5 seeds por configuración; decisiones sobre la **mediana** OOS con IQR. El conteo de configuraciones probadas (MLflow) alimenta el DSR.
**Consecuencias:** métricas honestas y reproducibles; más cómputo (mitigado con dynamic task mapping en Airflow). Separa evidencia de anécdota.

---

## ADR-006 — El Oro se integra al registro dinámico compartido, no como silo (SPEC-12)
**Contexto:** este paquete se diseñó como un repo `gold-rl/` aparte con DAGs `xau_*` a mano y su propio registry. El sistema USD/COP ya tiene operativa una columna vertebral multi-activo / multi-estrategia: `AssetProfile` por activo, registro dinámico (`registry.json` + `manifest.json`), backtests inmutables versionados, fábrica de pipelines config-driven y visibilidad automática en el frontend (dropdown + replay). Duplicarla sería deuda inmediata y divergencia de contratos.
**Decisión:** el Oro se onboarda como **un activo más**: se define en `config/assets/xauusd.yaml` (`AssetProfile`), publica sus backtests vía el contrato de salida `register_bundle` con **versionado inmutable** por `(strategy_id, version, year)`, y reutiliza aprobación (Vote 2), replay y monitoreo existentes. La evolución es **estrictamente aditiva**: no se modifican `lib/contracts/*.ts` ni las rutas `app/api/**`/JSON que el front ya consume. Los DAGs `xau_*` son el primer caso que la **fábrica de pipelines** debe poder emitir.
**Consecuencias:** cero duplicación de infraestructura; un 2º activo/estrategia entra por config + datos, no por copiar DAGs; v1 y v2 coexisten replayables. Supersede la premisa "repo aislado" del README/SPEC-00 originales.

**Estado de ejecución (2026-07-03):**
- ✅ (a) `register_bundle`/`publish_versioned_bundle` **cableado** en `scripts/pipeline/train_and_export_smart_simple.py` (aditivo).
- ✅ Versionado inmutable **implementado y probado** — 3 versiones coexisten bajo `smart_simple_v11` (2.0.0 / 3.0.0-A / 3.0.0-B) + rama `smart_simple_aggr`; blindado por 9 tests R en `tests/contracts/test_strategy_registry.py`; rutas `/api/registry`, `/api/strategies/{id}/manifest`, `/api/registry/promote` operativas.
- ⏳ (b) El export legacy aún escribe `summary_<year>.json` **en paralelo** (compatibilidad); retirar el mutable es cosmético, post-rebuild del front baked.
- ⏳ (c) La **fábrica de pipelines** aún no existe (DAGs a mano). Es el paso pendiente para "0 DAGs por activo".
- ⏳ (d) Reconciliar el drift TS↔Python del feature-contract antes de derivar el contrato del Oro.
- ⏳ El `AssetProfile` del Oro (`config/assets/xauusd.yaml`) aún **no existe** — es el bloque 0 del onboarding (test A1).
