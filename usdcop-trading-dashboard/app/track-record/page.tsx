/**
 * /track-record — PUBLIC results page (no session).
 *
 * The companion to /metodologia: that page explains how we verify, this one shows what came
 * out. Until now every number lived behind a login, so a prospective client could not see a
 * single result before registering — friction that is fatal for a product whose credibility
 * IS the disclosure.
 *
 * The line it holds: results that already happened are marketing and go out freely; the
 * signal published BEFORE the fact is the product and never appears here. Closed trades are
 * served with a week of lag by /api/public/track-record for exactly that reason.
 *
 * Server component. It reads the published bundles DIRECTLY (via the shared composition in
 * `lib/public/track-record`) instead of fetching its own API over HTTP: a server component
 * calling its own origin gets prerendered at BUILD time, when nothing is listening, which
 * froze this page permanently in its "results unavailable" state. No network hop, one
 * composition shared with `/api/public/track-record`.
 */
import Link from 'next/link';

import { MetricBadge } from '@/components/ui/MetricBadge';
import { UI_TOKENS } from '@/lib/contracts/ui.contract';
import { buildTrackRecord } from '@/lib/public/track-record';
import type { ClosedTrade as Trade, PeriodView as Period } from '@/lib/public/track-record';

export const metadata = {
  title: 'Track record — resultados completos, sin filtro',
  description:
    'Retorno, drawdown y cada operación cerrada de nuestras estrategias en papel, con el p-valor al lado del resultado. Sin registro.',
};

// The bundle is regenerated weekly; a 5-minute cache is plenty and keeps the page instant.
export const revalidate = 300;

const pct = (v: number | null | undefined, sign = true) =>
  v == null ? '—' : `${sign && v > 0 ? '+' : ''}${v.toFixed(2)}%`;

function PeriodCard({ p }: { p: Period }) {
  const beatMarket =
    p.return_pct != null && p.buy_hold_pct != null && p.return_pct > p.buy_hold_pct;
  return (
    <div className={`${UI_TOKENS.card} p-6 space-y-4`}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-semibold">{p.label} · {p.year}</h3>
        <MetricBadge phase={p.year >= 2026 ? 'paper' : 'backtest'} />
      </div>

      <div className="flex flex-wrap items-end gap-x-8 gap-y-3">
        <div>
          <div className="text-3xl font-bold tabular-nums">{pct(p.return_pct)}</div>
          <div className={`text-xs ${UI_TOKENS.textSecondary}`}>Retorno</div>
        </div>
        {p.buy_hold_pct != null && (
          <div>
            <div className="text-xl font-semibold tabular-nums">{pct(p.buy_hold_pct)}</div>
            <div className={`text-xs ${UI_TOKENS.textSecondary}`}>Comprar y mantener</div>
          </div>
        )}
        {p.max_dd_pct != null && (
          <div>
            <div className="text-xl font-semibold tabular-nums">−{p.max_dd_pct.toFixed(2)}%</div>
            <div className={`text-xs ${UI_TOKENS.textSecondary}`}>Máx. drawdown</div>
          </div>
        )}
        <div>
          <div className="text-xl font-semibold tabular-nums">{p.trades ?? '—'}</div>
          <div className={`text-xs ${UI_TOKENS.textSecondary}`}>Operaciones</div>
        </div>
        {p.win_rate_pct != null && (
          <div>
            <div className="text-xl font-semibold tabular-nums">{p.win_rate_pct.toFixed(1)}%</div>
            <div className={`text-xs ${UI_TOKENS.textSecondary}`}>Acierto</div>
          </div>
        )}
      </div>

      {beatMarket && (
        <p className={`text-sm ${UI_TOKENS.textSecondary}`}>
          El mercado cayó {pct(p.buy_hold_pct, false)} en el período. La estrategia terminó
          en {pct(p.return_pct)} — la diferencia es pérdida que no ocurrió, no ganancia prometida.
        </p>
      )}

      {/* The statistics sit NEXT TO the return, never in a footnote. */}
      {p.ratios_withheld ? (
        <p className={`text-xs ${UI_TOKENS.textSecondary} border-t border-slate-700/50 pt-3`}>
          Con menos de 20 operaciones no publicamos Sharpe ni p-valor: a este tamaño de
          muestra esos números no significan nada. Solo conteo y P&amp;L.
        </p>
      ) : (
        <div className={`flex flex-wrap gap-x-6 gap-y-1 text-xs border-t border-slate-700/50 pt-3 ${UI_TOKENS.textSecondary}`}>
          {p.sharpe != null && <span>Sharpe <b className="tabular-nums">{p.sharpe.toFixed(2)}</b></span>}
          {p.profit_factor != null && <span>Profit factor <b className="tabular-nums">{p.profit_factor.toFixed(2)}</b></span>}
          {p.p_value != null && (
            <span>
              p = <b className="tabular-nums">{p.p_value.toFixed(4)}</b>
              {p.significant === false && ' — NO estadísticamente significativo'}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function TradeTable({ trades, title }: { trades: Trade[]; title: string }) {
  if (!trades.length) return null;
  return (
    <div className="space-y-2">
      <h3 className="font-semibold text-sm">{title}</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className={`text-left text-xs ${UI_TOKENS.textSecondary}`}>
              <th className="py-2 pr-4 font-medium">Entrada</th>
              <th className="py-2 pr-4 font-medium">Salida</th>
              <th className="py-2 pr-4 font-medium">Lado</th>
              <th className="py-2 pr-4 font-medium text-right">P&amp;L</th>
              <th className="py-2 pl-4 font-medium">Motivo</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t) => (
              <tr key={`${t.opened}-${t.closed}-${t.side}`} className="border-t border-slate-700/40">
                <td className="py-2 pr-4 tabular-nums">{t.opened}</td>
                <td className="py-2 pr-4 tabular-nums">{t.closed}</td>
                <td className="py-2 pr-4">{t.side}</td>
                <td className={`py-2 pr-4 text-right tabular-nums ${
                  (t.pnl_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {pct(t.pnl_pct)}
                </td>
                <td className={`py-2 pl-4 ${UI_TOKENS.textSecondary}`}>{t.exit_reason ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default async function TrackRecordPage() {
  const data = await buildTrackRecord();

  return (
    <main className={`min-h-screen flex flex-col items-center ${UI_TOKENS.surface} ${UI_TOKENS.textPrimary} py-16 px-4`}>
      <div className="w-full max-w-3xl mx-auto space-y-12">

        <header className="space-y-4">
          <h1 className="text-3xl sm:text-4xl font-bold">Todos los números, incluidos los malos</h1>
          <p className={UI_TOKENS.textSecondary}>
            Cada operación cerrada, el drawdown completo y el p-valor junto al retorno — no en
            una nota al pie. Sin registro.{' '}
            <Link href="/metodologia" className="text-cyan-400 hover:underline">
              Cómo lo verificamos →
            </Link>
          </p>
          {data?.strategy?.bundle_date && (
            <p className={`text-xs font-mono ${UI_TOKENS.textSecondary}`}>
              {data.strategy.name} · bundle publicado {data.strategy.bundle_date}
            </p>
          )}
        </header>

        {!data || data.unavailable ? (
          <p className={UI_TOKENS.textSecondary}>
            Los resultados publicados no están disponibles en este momento. Vuelve a intentarlo
            en unos minutos.
          </p>
        ) : (
          <>
            <section className="space-y-4">
              {data.periods.forward && <PeriodCard p={data.periods.forward} />}
              {data.periods.backtest && <PeriodCard p={data.periods.backtest} />}
              <p className={`text-xs ${UI_TOKENS.textSecondary}`}>
                El forward es el juez. Un backtest se puede ajustar hasta que se vea bien; una
                señal publicada antes del hecho, no.
              </p>
            </section>

            {data.defence.length > 0 && (
              <section className="space-y-4">
                <h2 className="text-xl font-semibold">Qué pasa cuando el mercado cae</h2>
                <p className={UI_TOKENS.textSecondary}>
                  No prometemos ganarle al mercado. La prueba que importa es el año malo.
                </p>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border-collapse">
                    <thead>
                      <tr className={`text-left text-xs ${UI_TOKENS.textSecondary}`}>
                        <th className="py-2 pr-4 font-medium">Activo</th>
                        <th className="py-2 pr-4 font-medium">Año</th>
                        <th className="py-2 pr-4 font-medium text-right">Mercado</th>
                        <th className="py-2 pr-4 font-medium text-right">Estrategia</th>
                        <th className="py-2 font-medium text-right">Ops.</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.defence.map((d) => (
                        <tr key={`${d.asset}-${d.year}-${d.strategy_name}`} className="border-t border-slate-700/40">
                          <td className="py-2 pr-4">{d.asset}</td>
                          <td className="py-2 pr-4 tabular-nums">{d.year}</td>
                          <td className="py-2 pr-4 text-right tabular-nums text-red-400">
                            {pct(d.market_pct)}
                          </td>
                          <td className={`py-2 pr-4 text-right tabular-nums ${
                            (d.strategy_pct ?? 0) >= 0 ? 'text-emerald-400' : 'text-amber-400'}`}>
                            {pct(d.strategy_pct)}
                          </td>
                          <td className={`py-2 text-right tabular-nums ${UI_TOKENS.textSecondary}`}>
                            {d.trades ?? '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className={`text-xs ${UI_TOKENS.textSecondary}`}>
                  Solo aparecen los activos cuyo bundle publica la referencia de comprar y
                  mantener. Sin esa referencia no hay comparación que hacer, así que no la
                  inventamos.
                </p>
              </section>
            )}

            <section className="space-y-6">
              <div className="space-y-2">
                <h2 className="text-xl font-semibold">Operación por operación</h2>
                <p className={UI_TOKENS.textSecondary}>
                  Todas las operaciones cerradas, ganadoras y perdedoras, con{' '}
                  {data.closed_trades.lag_days} días de rezago. El rezago existe porque la
                  señal vigente es el producto; el historial es la prueba.
                </p>
              </div>
              <TradeTable trades={data.closed_trades.forward} title="Producción forward (papel)" />
              <TradeTable trades={data.closed_trades.backtest} title="Backtest fuera de muestra 2025" />
            </section>

            <footer className={`text-xs ${UI_TOKENS.textSecondary} border-t border-slate-700/50 pt-6 space-y-2`}>
              <p>{data.disclaimer}</p>
              <p>
                Contenido informativo y educativo; no constituye asesoría financiera ni una
                oferta de valores.
              </p>
            </footer>
          </>
        )}
      </div>
    </main>
  );
}
