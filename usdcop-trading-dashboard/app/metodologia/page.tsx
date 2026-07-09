/**
 * /metodologia — public transparency page (ux-navigation §2: "el arma de ventas al
 * comprador sofisticado"). Client-facing version of the quant constitution: what
 * LIVE/BACKTEST/PAPER mean HERE, what a statistical gate is, why most weeks have no
 * trade, and the pre-signed withdrawal protocol. Server component — static, fast.
 */
import { MetricBadge } from '@/components/ui/MetricBadge';
import { UI_TOKENS } from '@/lib/contracts/ui.contract';

export const metadata = {
  title: 'Metodología — verificación antes que promesas',
  description:
    'Pre-registro, validación estadística, paper trading y protocolo de retiro: cómo verificamos cada estrategia antes y después de tocar dinero real.',
};

const PILLARS = [
  {
    title: 'Pre-registro',
    body:
      'Los parámetros de cada estrategia se declaran ANTES de ver los resultados. Nada se ajusta mirando el examen: lo decidido después de ver los datos de prueba no cuenta como evidencia.',
  },
  {
    title: 'Validación estadística',
    body:
      'Un buen resultado puede ser suerte. Corregimos por cuántas variantes se probaron (Deflated Sharpe) y exigimos significancia real. Un backtest bonito, por sí solo, no promueve nada.',
  },
  {
    title: 'Paper → Producción',
    body:
      'Toda estrategia opera primero en simulado con reglas congeladas. Solo el desempeño forward — señales publicadas ANTES del hecho, sin edición retroactiva — cuenta como prueba definitiva.',
  },
  {
    title: 'Protocolo de retiro',
    body:
      'Antes de operar se firman las condiciones de retiro: cuánta pérdida, cuántas semanas, qué umbrales. Si la estrategia deja de funcionar, se retira y lo verás publicado. Los umbrales no se relajan en drawdown.',
  },
];

export default function MetodologiaPage() {
  return (
    <main className={`min-h-screen flex flex-col items-center ${UI_TOKENS.surface} ${UI_TOKENS.textPrimary} py-16 px-4`}>
      <div className="w-full max-w-3xl mx-auto space-y-12">
        <header className="space-y-4">
          <h1 className="text-3xl sm:text-4xl font-bold">
            No vendemos backtests bonitos.
          </h1>
          <p className={`${UI_TOKENS.textSecondary} text-lg leading-relaxed`}>
            Cada estrategia se pre-registra antes de probarse, pasa control de significancia
            estadística, corre en simulado antes de producción y tiene reglas de retiro
            firmadas de antemano. Si deja de funcionar, se retira — y lo verás publicado.
          </p>
        </header>

        <section className="space-y-4">
          <h2 className="text-xl font-semibold">Qué significa cada etiqueta</h2>
          <div className={`${UI_TOKENS.card} p-6 space-y-4 text-sm`}>
            <div className="flex items-start gap-3">
              <MetricBadge phase="live" />
              <p className={UI_TOKENS.textSecondary}>
                <strong className={UI_TOKENS.textPrimary}>Producción forward.</strong>{' '}
                Señales publicadas antes del hecho, con reglas congeladas. Es el único
                número que usamos como titular. No se puede editar retroactivamente.
              </p>
            </div>
            <div className="flex items-start gap-3">
              <MetricBadge phase="backtest" />
              <p className={UI_TOKENS.textSecondary}>
                <strong className={UI_TOKENS.textPrimary}>Histórico fuera de muestra.</strong>{' '}
                Útil y visible, pero nunca el titular: todo backtest hereda el riesgo de
                sobreajuste, por eso lo deflactamos por número de intentos y lo mostramos
                en segundo plano.
              </p>
            </div>
            <div className="flex items-start gap-3">
              <MetricBadge phase="paper" />
              <p className={UI_TOKENS.textSecondary}>
                <strong className={UI_TOKENS.textPrimary}>Simulado en tiempo real.</strong>{' '}
                La antesala obligatoria de producción — y el modo inicial de toda cuenta de
                ejecución automática. Jamás se mezcla con cifras reales.
              </p>
            </div>
          </div>
        </section>

        <section className="space-y-4">
          <h2 className="text-xl font-semibold">Los cuatro pilares</h2>
          <div className="grid sm:grid-cols-2 gap-4">
            {PILLARS.map((p) => (
              <article key={p.title} className={`${UI_TOKENS.card} p-5`}>
                <h3 className="font-semibold mb-2">{p.title}</h3>
                <p className={`${UI_TOKENS.textSecondary} text-sm leading-relaxed`}>{p.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className={`${UI_TOKENS.card} p-6 space-y-3`}>
          <h2 className="text-xl font-semibold">¿Por qué hay semanas sin operación?</h2>
          <p className={`${UI_TOKENS.textSecondary} text-sm leading-relaxed`}>
            Porque la mayoría de las semanas la respuesta estadísticamente correcta es no
            operar. Los filtros de régimen y volatilidad deciden <em>si</em> hay condiciones
            antes de decidir <em>hacia dónde</em>. Un sistema que opera todas las semanas no
            está leyendo el mercado: está cobrando comisiones. Cuando no operamos, lo decimos
            — esa es la diferencia entre una decisión y una ausencia.
          </p>
        </section>

        <section className="space-y-3">
          <h2 className="text-xl font-semibold">Custodia y llaves</h2>
          <p className={`${UI_TOKENS.textSecondary} text-sm leading-relaxed`}>
            Nunca custodiamos tu dinero. La ejecución automática usa llaves API de TU
            exchange <strong>sin permiso de retiro</strong> — las llaves que permiten
            retirar fondos se rechazan en el registro. Toda cuenta pasa por un período
            simulado obligatorio antes de habilitar dinero real, y siempre tienes tu propio
            botón de parada.
          </p>
        </section>

        <footer className={`${UI_TOKENS.textMuted} text-xs leading-relaxed border-t border-slate-800 pt-6`}>
          Contenido informativo y educativo; no constituye asesoría financiera. Rendimientos
          pasados no garantizan resultados futuros. Operar divisas y criptoactivos implica
          riesgo de pérdida total del capital. La decisión y el riesgo son tuyos.
        </footer>
      </div>
    </main>
  );
}
