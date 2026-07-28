# E2E BDD — Evidencia visual de los BLs (2026-07-27)

Operador: subagente CLAUDE (webapp-testing / Playwright headless, viewport 1600x1000).
Login: admin / Admin2026! (captcha aritmético resuelto automáticamente).

## HALLAZGO PRINCIPAL: el contenedor (:5000) sirve un build viejo

Contra `http://localhost:5000` (contenedor `usdcop-dashboard`):
- `/replay` → **404** (la ruta `app/replay/page.tsx` existe en el repo pero no en el build).
- `/forecasting` (ambos assets): **sin** banner "DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN".
- `/production`: **sin** panel "Candidatas A/B".

Los strings de los BLs sí existen en el código fuente (`lib/ui/forecast-disclaimer.ts`,
`components/gm/views/ForecastingView.tsx`, `components/gm/views/ProductionView.tsx`,
`app/replay/page.tsx`) → **los cambios TSX necesitan rebuild de la imagen del dashboard**.

Mitigación aplicada (según misión): `npx next dev -p 3001` dentro de
`usdcop-trading-dashboard` con env de host
(`DATABASE_URL=postgresql://…@localhost:5432`, `NEXTAUTH_URL=http://localhost:3001`,
`NODE_ENV=development`). Nota operativa: sourcing del `.env` raíz rompe `next dev`
porque exporta `NODE_ENV=production` (EvalError en middleware edge) — hay que
sobrescribirlo. La evidencia `dev_*.png` es contra el código nuevo en :3001.

## Resultados por ítem (contra dev :3001 = código nuevo)

| # | BL | Ítem | Veredicto | Evidencia |
|---|----|------|-----------|-----------|
| 1 | BL-02 | `/forecasting?asset=xauusd` banner ámbar "DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN" | **PASS** | `dev_01_forecasting_xauusd_banner.png` |
| 2 | BL-03 | `/forecasting` (USD/COP) banner + badges neutros ↑/↓ + "probabilidad estimada de subida" | **PASS** | `dev_02_forecasting_usdcop_badges.png` |
| 3 | BL-05 | `/production` panel "Candidatas A/B (paper, ancla ene-2026)" con v11/v12/v14 + nota N<20 | **PASS** | `dev_03_production_candidatas.png` |
| 4 | BL-34 | `/replay` renderiza selector de estrategias + equity, sin 404/redirect | **PASS** | `dev_04_replay.png` |
| 5 | — | Consola: errores JS nuevos en las 4 páginas | **PASS** (0 errores) | `dev_raw_results.json` |

### Detalle

1. **BL-02 (xauusd)** — Banner ámbar arriba del todo con headline
   "DIAGNÓSTICO — NO ES UNA SEÑAL DE INVERSIÓN" y subtexto "Superficie de diagnóstico,
   no de señales… ≈52% … moneda al aire tras ajustar por los 9 modelos probados". PASS.

2. **BL-03 (usdcop, default)** — Banner presente (variante COP: "Replay shadow, no señal
   ejecutable…"). Texto "probabilidad estimada de subida: 65.1%" visible (KPI decisión +
   forward H25). Badges de predicción inspeccionados por computed-style: colores
   `rgb(230,237,247)` / `rgb(196,208,227)` / `rgb(132,148,172)` (slate/blanco),
   fondo `rgba(148,163,184,.07)` y acento cian `rgb(34,211,238)` — **ningún verde/rojo**;
   dirección solo por glifos ↑/↓ + UP/DOWN. PASS.

3. **BL-05 (production)** — Panel "Candidatas A/B (paper, ancla ene-2026)" renderiza con
   `smart_simple_v11` (PRODUCCIÓN), `smart_simple_v12` y `smart_simple_v14`
   (PAPER · JUEZ SELLADO), nota "N<20 ⇒ solo conteo y PnL" en las filas, nota v13
   EXCLUIDA, y meta "Ancla 2026-01-01 (directiva operador 2026-07-22)". PASS.
   - Fuente de datos verificada: `GET /api/data/production/paper/candidates_ledger_2026.json`
     → 200 con sesión (401 anónimo, correcto por RBAC).
   - Matiz de timing: en la primera pasada el panel aún no estaba (cold-compile del dev
     server + fetch client-side); con espera ≥12 s renderiza siempre. No es un bug del BL.

4. **BL-34 (replay)** — `/replay` NO devuelve 404 ni redirige a /login (final_url
   `/replay`). Renderiza la vista completa: selector de estrategia (Smart Simple v1.1.0
   APPROVED / v2.0.0 backtest), selector de versión (v1.0.0…v2.0.0★), controles
   "Reproducir replay" (rango, 1x/2x/4x), chart de precio con señales BUY/SELL, panel
   "Curva de equity", gates Voto 1 y Aprobación humana (Voto 2/2). PASS.
   - Matiz: la curva de equity aparece vacía hasta reproducir el replay (estado inicial).

5. **Consola** — 0 errores JS en las 4 páginas contra :3001 (`console_errors: []` en
   `dev_raw_results.json`). Contra :5000 el único error fue el 404 de `/replay`
   (build viejo, ver hallazgo principal).

## Archivos

- `dev_00_post_login.png` … `dev_04_replay.png` — evidencia contra código nuevo (:3001).
- `00_post_login.png` … `04_replay.png` + `raw_results.json` — evidencia del build viejo
  del contenedor (:5000): sin banner, sin panel Candidatas, `/replay` 404.
- `dev_raw_results.json` — checks automatizados (textos, computed styles, console, http).

## Acción pendiente sugerida

Rebuild/redeploy de la imagen del dashboard (`docker compose build dashboard` o el make
target equivalente) para que :5000 sirva los BLs de hoy; el dev server en :3001 quedó
levantado solo como evidencia y puede apagarse.
