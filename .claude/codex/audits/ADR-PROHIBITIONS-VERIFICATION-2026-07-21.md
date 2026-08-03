---
kind: audit
status: HISTORICAL
version: 1.0.0
last_verified: 2026-07-21
supersedes: []
code_anchors: []
---
# Verificación de prohibiciones ADR-0020

Ejecutado `tests/regression/test_risk_controlled_book_disclosure.py`: **4 passed**.

El test verifica que el libro de riesgo controlado:

- no se promociona como alfa/edge/señal/modelo;
- usa Calmar y drawdown como titular, no Sharpe aislado;
- publica correlaciones incondicionales junto a correlaciones condicionales;
- declara que es beta gestionada;
- mantiene forward/protocolo de retiro antes de operar.

El ADR sigue en estado `PROPOSED` y el libro `research_validated`; pasar el test no
lo convierte en producción. La evidencia OOS/forward, PIT, costes y aprobación
humana siguen siendo gates independientes.
