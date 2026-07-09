/**
 * /legal/{terminos,riesgo,privacidad} — S10 legal pages (public, static).
 * One dynamic route (DRY); unknown docs 404. Plain-language, sober; NOT legal advice —
 * final wording requires counsel before charging real money (rbac-monetization §B.7).
 */
import { notFound } from 'next/navigation';

const DOCS: Record<string, { title: string; sections: Array<{ h: string; p: string }> }> = {
  terminos: {
    title: 'Términos de servicio',
    sections: [
      { h: 'Naturaleza del servicio', p: 'Proveemos contenido informativo y educativo de carácter cuantitativo: señales, análisis y herramientas de ejecución sobre cuentas del propio usuario. No somos asesores financieros, no administramos recursos de terceros y no custodiamos fondos.' },
      { h: 'Cuentas y planes', p: 'El acceso se rige por el plan contratado (entitlements). Los pagos se procesan por la pasarela; el plan se activa al confirmarse el pago y se degrada automáticamente al vencer. El historial propio del usuario nunca se borra al degradar.' },
      { h: 'Ejecución automática', p: 'El plan Auto opera exclusivamente sobre el exchange del usuario con llaves API SIN permiso de retiro, tras un período simulado obligatorio y aceptación expresa de la divulgación de riesgo. El usuario puede detener su ejecución en cualquier momento (kill switch propio).' },
      { h: 'Limitación de responsabilidad', p: 'Las decisiones de inversión y sus resultados son responsabilidad exclusiva del usuario. El servicio se provee "tal cual", sin garantía de disponibilidad continua ni de resultados.' },
      { h: 'Cancelación', p: 'La suscripción puede cancelarse en cualquier momento; el acceso pago permanece hasta el fin del período facturado.' },
    ],
  },
  riesgo: {
    title: 'Divulgación de riesgo',
    sections: [
      { h: 'Puedes perder dinero', p: 'Operar divisas y criptoactivos implica riesgo de pérdida total del capital. Ninguna metodología —incluida la nuestra— elimina ese riesgo; lo gestiona y lo acota por operación.' },
      { h: 'Rendimientos pasados', p: 'Los rendimientos pasados, en vivo o en backtest, no garantizan resultados futuros. Los backtests, en particular, heredan riesgo de sobreajuste; por eso nunca son nuestro titular.' },
      { h: 'Riesgo de modelo', p: 'Toda estrategia puede dejar de funcionar cuando cambia el régimen de mercado. Nuestro protocolo de retiro pre-firmado define cuándo se retira una estrategia — y lo publicamos.' },
      { h: 'Riesgo operacional', p: 'Fallas de datos, del exchange o de conectividad pueden impedir la ejecución o el cierre de posiciones. La ejecución automática incluye límites de riesgo y parada individual, pero el riesgo residual existe.' },
      { h: 'No es asesoría', p: 'Nada en esta plataforma constituye asesoría financiera personalizada. La decisión de operar, el tamaño y el riesgo asumido son del usuario.' },
    ],
  },
  privacidad: {
    title: 'Política de privacidad',
    sections: [
      { h: 'Qué guardamos', p: 'Correo, rol y plan del usuario; registros de auditoría de acciones sensibles (con IP); y, para el plan Auto, llaves API cifradas (AES-256-GCM) sin permiso de retiro.' },
      { h: 'Qué NO guardamos', p: 'No custodiamos fondos, no almacenamos llaves en texto plano y no vendemos datos personales a terceros.' },
      { h: 'Pagos', p: 'La pasarela de pagos procesa los datos de tarjeta directamente; nunca pasan por nuestros servidores.' },
      { h: 'Eliminación', p: 'El usuario puede solicitar la eliminación de su cuenta y llaves; los registros de auditoría se conservan por obligación de integridad (append-only) de forma anonimizada.' },
    ],
  },
};

export function generateStaticParams() {
  return Object.keys(DOCS).map((doc) => ({ doc }));
}

export default async function LegalPage({ params }: { params: Promise<{ doc: string }> }) {
  const { doc } = await params;
  const d = DOCS[doc];
  if (!d) notFound();

  return (
    <main className="min-h-screen flex flex-col items-center bg-slate-950 text-slate-100 py-16 px-4">
      <div className="w-full max-w-2xl mx-auto">
        <h1 className="text-3xl font-bold">{d.title}</h1>
        <p className="mt-2 text-xs text-slate-500">
          Última actualización: 2026-07-06 · Documento operativo — la redacción final para cobro
          real requiere revisión legal (SFC / jurisdicción aplicable).
        </p>
        <div className="mt-8 space-y-8">
          {d.sections.map((s) => (
            <section key={s.h}>
              <h2 className="text-lg font-semibold text-white">{s.h}</h2>
              <p className="mt-2 text-sm leading-relaxed text-slate-300">{s.p}</p>
            </section>
          ))}
        </div>
        <div className="mt-12 flex gap-4 text-sm">
          {Object.keys(DOCS).filter((k) => k !== doc).map((k) => (
            <a key={k} href={`/legal/${k}`} className="text-cyan-400 hover:underline">
              {DOCS[k].title} →
            </a>
          ))}
        </div>
      </div>
    </main>
  );
}
