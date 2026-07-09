# ADR-0012 — Protocolo de retiro pre-firmado

**Estado:** Aceptado · **Fecha:** pre-registro · **Spec:** SPEC-12 · **Sección:** §14

## Contexto

Toda la literatura enseña a construir; casi nadie escribe **cuándo apagar**. Sin criterios
pre-registrados, la decisión de apagar se toma en pleno drawdown, con el peor estado
emocional y el máximo incentivo a "darle una semana más". El resultado típico es un sistema
muerto operando con capital real.

## Decisión

El **protocolo de retiro se firma ANTES del go-live** (SPEC-12), con tres niveles:
- **Suspensión automática** (DD > 1.25× OOS · divergencia shadow > 2 % NAV/30d · PSI > 0.25 ·
  QA rojo 48h) ⇒ exposición → 0.
- **Decaimiento** (Calmar 12m < B1 por 2 trimestres) ⇒ media exposición + revisión.
- **Muerte** (3 suspensiones/12m · Calmar < B1 4 trimestres · quiebre estructural) ⇒ capital →
  B1/cash; re-encender exige **v4 desde cero**.
- **Regla de humildad:** ningún umbral se relaja en drawdown.

## Alternativas descartadas

- **Sin protocolo (decidir en el momento):** el fallo (sesgo emocional).
- **Umbrales ajustables en caliente:** invita a mover la meta cuando duele; se prohíbe (solo se
  cambian con el sistema en máximos o apagado).

## Consecuencias

El sistema nace con su certificado de defunción firmado. Se acepta apagar por reglas frías en
vez de por corazonadas calientes; el costo es renunciar a la flexibilidad de "esperar a que se
recupere", que es precisamente el sesgo que arruina cuentas.
