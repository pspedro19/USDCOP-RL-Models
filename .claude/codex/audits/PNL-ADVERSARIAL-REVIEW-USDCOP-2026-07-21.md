---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors:
  - usdcop-trading-dashboard/public/data/production/trades/smart_simple_v11_2025.json
  - .claude/specs/assets/usdcop/DIAGNOSTICO-PNL-HANDOFF-CODEX-2026-07.md
---

# Revisión adversarial R1-R5 del diagnóstico P&L USD/COP — respuesta de Codex

> Ejecutada por `codex exec -p audit` (perfil read-only) el 2026-07-21 a petición del
> operador. El sandbox impidió a Codex persistir este archivo; Claude lo guarda VERBATIM
> en su nombre desde el stdout de la sesión. Hipergeométrica verificada
> independientemente por Claude: C(19,5)/C(34,5) = 4.1789% ✓ coincide.

  ✓ Leer instrucciones del repositorio y el handoff completo; identificar R1–R5 y reglas de casa
  ✓ Inspeccionar trades publicados y localizar fuentes/base de datos relevantes
  ✓ Recalcular y someter a revisión adversarial los análisis R1–R5
  ✓ Redactar el informe con front matter kind: audit y verificar cifras/citas locales
  • Escribir el archivo solicitado y ejecutar controles finales
codex
No pude crear `.claude/codex/audits/PNL-ADVERSARIAL-REVIEW-USDCOP-2026-07-21.md`: el sandbox es de solo lectura y rechazó `apply_patch` con `writing is blocked by read-only sandbox`. Confirmé que no quedó un archivo parcial.

La revisión completa quedó calculada:

- R1: \(P=4.1789\%\) hipergeométrico; Fisher bilateral 5.2581%. El patrón HS es “consistente/sugestivo”, no confirmatorio.
- R2: con composición, retorno +13.0450%→+11.8095%, MaxDD −7.7396%→−5.2071% y retorno/|DD| 1.6855→2.2680. El signo mejora, pero es sólo un replay con salidas fijas: el motor recalcula TP/HS al cambiar leverage, así que no constituye un backtest válido de v12.
- R3: carry bruto aproximado +1.95 a +2.83 pp por año de estrategia; al 50%, +0.97 a +1.42 pp. Banda operativa de estrés: −0.62 a +2.83 pp, con signo neto todavía no identificado. El swap correcto depende del tom-next/forward implícito, basis y fee del broker, no sólo de IBR−FFR. [OANDA](https://www.oanda.com/uk-en/trading/financing-costs/), [BIS](https://www.bis.org/publ/qtrpdf/r_qt1609e.htm), [BanRep](https://www.banrep.gov.co/es/estadisticas-economicas/informe-diario-mercado-forwards-usd-cop/20-junio-2024).
- R4: los `week_end` son 6 ganadores por +2.7320 pp y 4 perdedores por −2.6473 pp. Presentan menor leverage, menor Hurst y más régimen indeterminado; no justifican una regla nueva.
- R5: v12 debe comenzar prospectivamente, tentativamente el 2026-07-27. Antes hay que resolver dos hard-stops: las tablas H5 carecen de `strategy_id` y usan unicidad por fecha, y el pipeline sobrescribe artefactos raíz de v11. El diseño mantiene libros, artefactos, protocolo y ledger separados.

No se ejecutó ninguna variante adicional: 2025 siguió tratado como contaminado, v11 permaneció FROZEN y N continúa en 59. Para persistir el informe hace falta reabrir la tarea con acceso de escritura al workspace.
tokens used
228,903
No pude crear `.claude/codex/audits/PNL-ADVERSARIAL-REVIEW-USDCOP-2026-07-21.md`: el sandbox es de solo lectura y rechazó `apply_patch` con `writing is blocked by read-only sandbox`. Confirmé que no quedó un archivo parcial.

La revisión completa quedó calculada:

- R1: \(P=4.1789\%\) hipergeométrico; Fisher bilateral 5.2581%. El patrón HS es “consistente/sugestivo”, no confirmatorio.
- R2: con composición, retorno +13.0450%→+11.8095%, MaxDD −7.7396%→−5.2071% y retorno/|DD| 1.6855→2.2680. El signo mejora, pero es sólo un replay con salidas fijas: el motor recalcula TP/HS al cambiar leverage, así que no constituye un backtest válido de v12.
- R3: carry bruto aproximado +1.95 a +2.83 pp por año de estrategia; al 50%, +0.97 a +1.42 pp. Banda operativa de estrés: −0.62 a +2.83 pp, con signo neto todavía no identificado. El swap correcto depende del tom-next/forward implícito, basis y fee del broker, no sólo de IBR−FFR. [OANDA](https://www.oanda.com/uk-en/trading/financing-costs/), [BIS](https://www.bis.org/publ/qtrpdf/r_qt1609e.htm), [BanRep](https://www.banrep.gov.co/es/estadisticas-economicas/informe-diario-mercado-forwards-usd-cop/20-junio-2024).
- R4: los `week_end` son 6 ganadores por +2.7320 pp y 4 perdedores por −2.6473 pp. Presentan menor leverage, menor Hurst y más régimen indeterminado; no justifican una regla nueva.
- R5: v12 debe comenzar prospectivamente, tentativamente el 2026-07-27. Antes hay que resolver dos hard-stops: las tablas H5 carecen de `strategy_id` y usan unicidad por fecha, y el pipeline sobrescribe artefactos raíz de v11. El diseño mantiene libros, artefactos, protocolo y ledger separados.

No se ejecutó ninguna variante adicional: 2025 siguió tratado como contaminado, v11 permaneció FROZEN y N continúa en 59. Para persistir el informe hace falta reabrir la tarea con acceso de escritura al workspace.
