# ADR-0010 — HMM de régimen con fit congelado walk-forward

**Estado:** Aceptado · **Fecha:** pre-registro · **Spec:** SPEC-01 · **Modo de fallo:** §4.3 (anti-look-ahead capa modelos)

## Contexto

Si el HMM de régimen se re-ajusta sobre el histórico **completo** y con él se re-etiquetan los
regímenes del pasado, cada label pasado incorpora información del futuro (las medias y
covarianzas del modelo "vieron" 2024 al etiquetar 2019). El backtest queda **silenciosamente
contaminado** — un look-ahead que todo el arsenal anti-look-ahead de *datos* no detecta,
porque vive en el *modelo*.

## Decisión

**Protocolo de fit congelado walk-forward obligatorio:**
1. Fit inicial sobre la 1.ª ventana, **congelado**.
2. Etiqueta **solo hacia adelante** con info ≤ D−1.
3. Re-fit anual sobre ventana expandida **con purga**; el nuevo modelo etiqueta solo desde su
   despliegue. **Jamás se re-etiqueta el pasado.**
4. La secuencia de labels como-se-habría-producido se versiona en DVC; el backtest usa **esa**,
   no la del modelo final.
5. Estabilidad: > 20 % de disenso entre re-fits ⇒ modelo inestable ⇒ no entra a producción.

## Alternativas descartadas

- **Re-fit sobre histórico completo:** look-ahead de modelo (el fallo).
- **Fit único sin re-fit nunca:** el modelo queda ciego a cambios estructurales (era ETF); el
  re-fit programado hacia adelante es el balance correcto.

## Consecuencias

El anti-look-ahead se eleva a **tres capas** (datos, modelos, clasificadores). Se paga con
complejidad operativa (versionar labels, orquestar re-fits), justificada porque el backtest
sin esto es una mentira.
