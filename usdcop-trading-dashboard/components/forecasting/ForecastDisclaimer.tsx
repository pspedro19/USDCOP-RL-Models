'use client';

/**
 * ForecastDisclaimer — banner de honestidad COMPARTIDO de todas las superficies
 * de forecasting (CTR-QUANT-CONSTITUTION-001, BL-02/BL-04, remediación CXD-032).
 *
 * UNA sola implementación del caveat para: GM ForecastingView (todos los activos
 * y modos), legacy ForecastingDashboard (zoo USD/COP) y legacy WeeklyInferenceView
 * (Gold/BTC). El copy viene EXCLUSIVAMENTE del SSOT lib/ui/forecast-disclaimer.ts
 * — titular Y cuerpo — y el componente se monta INCONDICIONAL en cada superficie:
 * aquí no hay ninguna rama que devuelva null, ni prop para ocultarlo, ni estado
 * de carga que lo suprima. Un disclaimer que puede desaparecer por una rama es
 * decoración, no disclosure (candado: tests/regression/test_forecasting_caveat_present.py
 * + tests/unit/components/forecasting-caveat-surfaces.test.tsx).
 *
 * `variant` elige entre las DOS ramas honestas del SSOT (zoo/weekly vs replay
 * direccional causal); `children` permite añadir una línea DERIVADA de los datos
 * cargados (p.ej. la DA media real del legacy dashboard) sin sustituir el cuerpo
 * SSOT — la honestidad dinámica se suma, nunca reemplaza al contrato.
 */

import type { ReactNode } from 'react';
import {
  FORECAST_DISCLAIMER_TESTID,
  FORECAST_DISCLAIMER_HEADLINE,
  FORECAST_DISCLAIMER_DIRECTIONAL_TITLE,
  FORECAST_DISCLAIMER_DIRECTIONAL_BODY,
  FORECAST_DISCLAIMER_ZOO_TITLE,
  FORECAST_DISCLAIMER_ZOO_BODY,
} from '@/lib/ui/forecast-disclaimer';

export type ForecastDisclaimerVariant = 'zoo' | 'directional';

export function ForecastDisclaimer({
  variant = 'zoo',
  className = '',
  children,
}: {
  /** Rama SSOT del cuerpo: 'zoo' (model zoo / weekly inference) o 'directional' (replay causal). */
  variant?: ForecastDisclaimerVariant;
  /** Clases de layout del contenedor (márgenes); nunca de ocultamiento. */
  className?: string;
  /** Línea adicional DERIVADA de los datos (opcional); se suma al cuerpo SSOT. */
  children?: ReactNode;
}) {
  const title = variant === 'directional'
    ? FORECAST_DISCLAIMER_DIRECTIONAL_TITLE
    : FORECAST_DISCLAIMER_ZOO_TITLE;
  const body = variant === 'directional'
    ? FORECAST_DISCLAIMER_DIRECTIONAL_BODY
    : FORECAST_DISCLAIMER_ZOO_BODY;

  return (
    <div
      role="note"
      aria-label={FORECAST_DISCLAIMER_HEADLINE}
      data-testid={FORECAST_DISCLAIMER_TESTID}
      className={`rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-2.5 text-xs leading-relaxed text-amber-200/90 ${className}`}
    >
      <div className="font-bold tracking-wide text-amber-200 mb-1">
        {FORECAST_DISCLAIMER_HEADLINE}
      </div>
      <span className="font-semibold">{title} </span>
      {body}
      {children != null && <div className="mt-1">{children}</div>}
    </div>
  );
}
