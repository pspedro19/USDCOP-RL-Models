/**
 * strategy-kpis — constructor SSOT de la fila de KPIs de estrategia
 * (CTR-GM-UI-001). Producción y Backtest construyen sus tiles desde AQUÍ para
 * que ambas vistas muestren exactamente las mismas métricas del bundle
 * publicado (regla Vote-2/P1: números del summary, nunca recomputados).
 *
 * quant-constitution §6: con N<20 trades solo se publica conteo y P&L —
 * `canShowRatios` (lib/contracts/ui.contract) decide, no cada vista.
 */
import { calmarRatio, canShowRatios } from '@/lib/contracts/ui.contract';
import type { StrategyStats } from '@/lib/contracts/strategy.contract';
import type { GmTone } from '@/lib/ui/gm-tokens';

export interface StrategyKpi {
  label: string;
  value: string;
  tone: GmTone;
  sub?: string;
}

/** profit_factor seguro para display (null = sin pérdidas, NUNCA Infinity). */
export function formatProfitFactor(pf: number | null | undefined): string {
  if (pf == null) return 'N/A';
  if (pf > 100) return '>100';
  return pf.toFixed(2);
}

function fmtUsd(n: number): string {
  const sign = n > 0 ? '+' : n < 0 ? '-' : '';
  return `${sign}$${Math.abs(n).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
}

/**
 * Fila de KPIs (7 con ratios / 3 con N<20) desde las stats del bundle publicado.
 * `ratiosHidden` permite a la vista mostrar la nota "solo conteo y P&L".
 */
export function buildStrategyKpis(
  stats: StrategyStats | undefined,
  opts: { initialCapital: number; nTrades: number },
): { kpis: StrategyKpi[]; ratiosHidden: boolean } {
  const { initialCapital, nTrades } = opts;
  const ret = stats?.total_return_pct ?? null;
  const pnlUsd = stats?.final_equity != null ? stats.final_equity - initialCapital : null;

  const retKpi: StrategyKpi = {
    label: 'Retorno Total',
    value: ret != null ? `${ret >= 0 ? '+' : ''}${ret.toFixed(2)}%` : 'N/A',
    tone: ret == null || ret === 0 ? 'neutral' : ret > 0 ? 'pos' : 'neg',
    sub: stats?.final_equity != null
      ? `$${stats.final_equity.toLocaleString('en-US', { maximumFractionDigits: 0 })} final`
      : undefined,
  };
  const tradesKpi: StrategyKpi = {
    label: 'Operaciones',
    value: String(nTrades),
    tone: 'neutral',
    sub: stats?.n_long != null || stats?.n_short != null
      ? `${stats?.n_long ?? 0} long · ${stats?.n_short ?? 0} short`
      : undefined,
  };

  if (!canShowRatios(nTrades)) {
    return {
      ratiosHidden: true,
      kpis: [
        retKpi,
        {
          label: 'P&L',
          value: pnlUsd != null ? fmtUsd(pnlUsd) : 'N/A',
          tone: pnlUsd == null || pnlUsd === 0 ? 'neutral' : pnlUsd > 0 ? 'pos' : 'neg',
        },
        tradesKpi,
      ],
    };
  }

  const sharpe = stats?.sharpe ?? null;
  const calmar = calmarRatio(stats?.total_return_pct, stats?.max_dd_pct, stats?.trading_days);
  const wr = stats?.win_rate_pct ?? null;
  const pf = stats?.profit_factor;
  const dd = stats?.max_dd_pct ?? null;

  return {
    ratiosHidden: false,
    kpis: [
      retKpi,
      {
        label: 'Sharpe',
        value: sharpe != null ? sharpe.toFixed(2) : 'N/A',
        tone: sharpe == null ? 'neutral' : sharpe >= 1.5 ? 'pos' : sharpe >= 1 ? 'warn' : 'neg',
        sub: 'Ajustado a riesgo',
      },
      {
        label: 'Calmar',
        value: calmar != null ? calmar.toFixed(2) : 'N/A',
        tone: calmar == null ? 'neutral' : calmar >= 1 ? 'pos' : 'warn',
        sub: 'Métrica primaria',
      },
      {
        label: 'Win Rate',
        value: wr != null ? `${wr.toFixed(1)}%` : 'N/A',
        tone: wr == null ? 'neutral' : wr >= 50 ? 'pos' : 'warn',
      },
      {
        label: 'Profit Factor',
        value: formatProfitFactor(pf),
        tone: pf == null ? 'neutral' : pf >= 1.5 ? 'pos' : pf >= 1 ? 'warn' : 'neg',
        sub: 'Ganancia / pérdida',
      },
      {
        label: 'Max Drawdown',
        value: dd != null ? `-${Math.abs(dd).toFixed(2)}%` : 'N/A',
        tone: dd == null ? 'neutral' : Math.abs(dd) < 10 ? 'pos' : Math.abs(dd) < 15 ? 'warn' : 'neg',
        sub: 'Pico a valle',
      },
      tradesKpi,
    ],
  };
}
