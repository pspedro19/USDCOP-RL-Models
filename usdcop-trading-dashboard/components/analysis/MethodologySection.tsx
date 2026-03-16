'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  TrendingUp,
  TrendingDown,
  BarChart3,
  ChevronDown,
  ChevronUp,
  Fuel,
  DollarSign,
  AlertTriangle,
  Activity,
  LineChart,
  Layers,
} from 'lucide-react';

export function MethodologySection() {
  const [expandedSection, setExpandedSection] = useState<string | null>('drivers');

  const toggle = (key: string) =>
    setExpandedSection(prev => (prev === key ? null : key));

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-6"
    >
      <h2 className="text-base font-semibold text-white mb-2 flex items-center gap-2">
        <Brain className="w-5 h-5 text-cyan-400" />
        Metodologia e Interpretabilidad del Analisis
      </h2>
      <p className="text-xs text-gray-500 mb-5">
        Como funciona este reporte, que impulsa el USD/COP, y como interpretar cada seccion.
      </p>

      <div className="space-y-2">
        {/* SECTION 1: USD/COP Macro Drivers */}
        <AccordionItem
          id="drivers"
          icon={<DollarSign className="w-4 h-4" />}
          title="Que Mueve el USD/COP"
          subtitle="Factores macroeconomicos fundamentales"
          expanded={expandedSection === 'drivers'}
          onToggle={() => toggle('drivers')}
        >
          <div className="space-y-4">
            <p className="text-xs text-gray-400 leading-relaxed">
              El tipo de cambio USD/COP es determinado por la oferta y demanda de dolares en Colombia.
              Los principales factores que influyen son:
            </p>

            <DriverCard
              icon={<Fuel className="w-4 h-4 text-amber-400" />}
              title="Petroleo (WTI / Brent)"
              impact="cop_strengthening"
              description="Colombia exporta ~40% de sus divisas en petroleo. Cuando el precio del crudo sube, entran mas dolares al pais, el peso se fortalece y el USD/COP baja."
              example="Ej: Brent sube de $72 a $78 → exportadores reciben mas USD → USD/COP cae ~100-200 pesos."
              metric="Correlacion: WTI vs USD/COP aprox. -0.6 (inversa)"
            />

            <DriverCard
              icon={<DollarSign className="w-4 h-4 text-blue-400" />}
              title="DXY (Dollar Index)"
              impact="cop_weakening"
              description="El DXY mide la fortaleza del dolar contra 6 monedas principales. Cuando sube, el dolar se fortalece globalmente y todas las monedas emergentes (incluyendo COP) se debilitan."
              example="Ej: Fed sube tasas → DXY sube de 104 a 107 → USD/COP sube ~200-400 pesos."
              metric="Correlacion: DXY vs USD/COP aprox. +0.7 (directa)"
            />

            <DriverCard
              icon={<AlertTriangle className="w-4 h-4 text-red-400" />}
              title="VIX (Indice de Volatilidad)"
              impact="cop_weakening"
              description="El VIX mide el miedo del mercado (volatilidad implicita del S&P 500). Cuando sube, los inversionistas salen de mercados emergentes (risk-off) y el COP se debilita."
              example="Ej: Crisis bancaria → VIX sube de 15 a 30 → flujos salen de Colombia → USD/COP sube 300+ pesos."
              metric="Correlacion: VIX vs USD/COP aprox. +0.5 (directa)"
            />

            <DriverCard
              icon={<Activity className="w-4 h-4 text-orange-400" />}
              title="EMBI Colombia (Riesgo Pais)"
              impact="cop_weakening"
              description="El EMBI mide la prima de riesgo de los bonos colombianos vs bonos del tesoro de EE.UU. Un spread mas alto indica mayor riesgo percibido y debilita el peso."
              example="Ej: Reforma tributaria → EMBI sube de 250 a 400 bps → inversionistas exigen mayor retorno → COP se deprecia."
              metric="Correlacion: EMBI vs USD/COP aprox. +0.6 (directa)"
            />

            <DriverCard
              icon={<TrendingUp className="w-4 h-4 text-emerald-400" />}
              title="Tasas BanRep (TPM / IBR)"
              impact="cop_strengthening"
              description="Tasas de interes altas en Colombia atraen capital extranjero (carry trade), fortaleciendo el peso. Recortes de tasas reducen el diferencial y debilitan el COP."
              example="Ej: BanRep mantiene TPM en 9.5% vs Fed en 4.5% → diferencial de 5% atrae capital → USD/COP baja."
              metric="Diferencial actual: TPM - Fed Funds ≈ 5-6 puntos porcentuales"
            />

            <DriverCard
              icon={<DollarSign className="w-4 h-4 text-yellow-400" />}
              title="Tasas Fed (FOMC)"
              impact="cop_weakening"
              description="Subidas de tasas de la Fed fortalecen el USD globalmente. El diferencial de tasas Colombia-EE.UU. se reduce, haciendo menos atractivo el carry trade hacia COP."
              example="Ej: Fed sube 25bps → Treasury yields suben → flujos regresan a EE.UU. → USD/COP sube."
              metric="Correlacion: UST10Y vs USD/COP aprox. +0.4 (directa)"
            />

            <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/30">
              <p className="text-[10px] text-gray-500 leading-relaxed">
                <strong className="text-gray-400">Resumen de impactos:</strong> Petroleo alto + carry trade alto = COP fuerte (USD/COP baja).
                DXY alto + VIX alto + EMBI alto = COP debil (USD/COP sube). En 2025-2026, el regimen
                cambio: las posiciones SHORT (apostar que USD/COP baja) tienen WR 56%, mientras LONG
                (apostar que sube) cayo de 58% a 28% de win rate (p=0.0014).
              </p>
            </div>
          </div>
        </AccordionItem>

        {/* SECTION 2: Technical Indicators */}
        <AccordionItem
          id="indicators"
          icon={<LineChart className="w-4 h-4" />}
          title="Indicadores Tecnicos Utilizados"
          subtitle="SMA, RSI, Bollinger, MACD, z-score"
          expanded={expandedSection === 'indicators'}
          onToggle={() => toggle('indicators')}
        >
          <div className="space-y-3">
            <IndicatorExplainer
              name="SMA (Simple Moving Average)"
              periods="5, 10, 20, 50 dias"
              description="Promedio de los ultimos N precios de cierre. Suaviza la serie temporal para identificar tendencias. Si el precio esta por encima de la SMA-20, la tendencia de corto plazo es alcista."
              interpretation="Precio > SMA-20 = tendencia alcista | Precio < SMA-20 = tendencia bajista | SMA-5 cruza SMA-20 hacia arriba = senal de compra"
            />

            <IndicatorExplainer
              name="RSI (Relative Strength Index)"
              periods="14 dias (EMA de Wilder)"
              description="Mide la velocidad y magnitud de cambios de precio. Va de 0 a 100. Calculado con el metodo de Wilder (no EMA estandar de pandas), que es mas estable y menos ruidoso."
              interpretation="RSI > 70 = sobrecomprado (posible caida) | RSI < 30 = sobrevendido (posible alza) | RSI entre 40-60 = zona neutral"
            />

            <IndicatorExplainer
              name="Bandas de Bollinger"
              periods="20 dias, 2 desviaciones estandar"
              description="Banda superior = SMA-20 + 2σ. Banda inferior = SMA-20 - 2σ. Mide volatilidad: bandas anchas = alta volatilidad, bandas estrechas = baja volatilidad (posible breakout)."
              interpretation="Precio toca banda superior = posible resistencia | Precio toca banda inferior = posible soporte | Ancho de banda disminuye = compresion (movimiento grande inminente)"
            />

            <IndicatorExplainer
              name="MACD (Moving Average Convergence Divergence)"
              periods="EMA-12, EMA-26, Signal-9"
              description="La linea MACD = EMA-12 - EMA-26. La linea de senal = SMA-9 del MACD. El histograma muestra la diferencia entre ambas."
              interpretation="MACD cruza signal hacia arriba = senal alcista | MACD cruza signal hacia abajo = senal bajista | Histograma creciente = momentum aumentando"
            />

            <IndicatorExplainer
              name="Z-Score"
              periods="20 dias"
              description="Mide cuantas desviaciones estandar esta el precio actual respecto al promedio de 20 dias. Z = (precio - SMA-20) / σ-20."
              interpretation="Z > +2 = extremadamente alto (sobrecomprado) | Z < -2 = extremadamente bajo (sobrevendido) | Z entre -1 y +1 = rango normal"
            />

            <IndicatorExplainer
              name="ROC (Rate of Change)"
              periods="5 y 20 dias"
              description="Porcentaje de cambio del precio en N dias: ROC = (precio_actual / precio_Ndias_atras - 1) * 100. Mide momentum puro."
              interpretation="ROC-5 > 0 = momentum alcista de corto plazo | ROC-20 < 0 = tendencia bajista de mediano plazo"
            />
          </div>
        </AccordionItem>

        {/* SECTION 3: How the AI Analysis Works */}
        <AccordionItem
          id="ai"
          icon={<Brain className="w-4 h-4" />}
          title="Como se Genera el Analisis con IA"
          subtitle="Pipeline de datos, LLM, y proceso analitico"
          expanded={expandedSection === 'ai'}
          onToggle={() => toggle('ai')}
        >
          <div className="space-y-3">
            <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/30">
              <h4 className="text-xs font-medium text-white mb-2">Pipeline Diario (Automatizado)</h4>
              <div className="space-y-1.5">
                <PipelineStep step={1} text="Ingestion de datos: OHLCV de USD/COP (intradía 5-min + diario), 13 variables macro de 7 fuentes" />
                <PipelineStep step={2} text="Ingestion de noticias: 3x/dia desde Investing.com, Portafolio, GDELT, La Republica" />
                <PipelineStep step={3} text="Calculo tecnico: SMA (5/10/20/50), Bollinger (20d, 2σ), RSI-14 (Wilder), MACD (12/26/9), z-score (20d) para cada variable macro" />
                <PipelineStep step={4} text="Enriquecimiento de noticias: categorizacion (9 categorias), relevancia (0-1), sentimiento (-1 a +1), extraccion de entidades" />
                <PipelineStep step={5} text="Generacion narrativa: GPT-4o (Azure) genera analisis diario en espanol con contexto macro + noticias + senales de trading" />
                <PipelineStep step={6} text="Exportacion: JSON estructurado para el dashboard + graficos PNG de cada variable macro" />
              </div>
            </div>

            <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/30">
              <h4 className="text-xs font-medium text-white mb-2">Modelos de Forecasting (Semanales)</h4>
              <div className="space-y-1.5">
                <PipelineStep step={1} text="H1 Diario: 9 modelos (Ridge, BayesianRidge, ARD, XGBoost, LightGBM, CatBoost, 3 hibridos) con 21 features" />
                <PipelineStep step={2} text="H5 Semanal: Ensemble Ridge + BayesianRidge, prediccion a 5 dias con confidence scoring (HIGH/MEDIUM/LOW)" />
                <PipelineStep step={3} text="Anti-leakage: todas las features macro usan T-1 (datos de ayer, no de hoy). Merge temporal con merge_asof(direction='backward')" />
                <PipelineStep step={4} text="Validacion: Walk-forward expanding window desde 2020. Backtest OOS en 2025: H1 +36.8% (p=0.018), H5 +20.0% (p=0.010)" />
              </div>
            </div>

            <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/30">
              <h4 className="text-xs font-medium text-white mb-2">Limitaciones y Advertencias</h4>
              <ul className="space-y-1 text-[11px] text-gray-400">
                <li className="flex items-start gap-2">
                  <span className="text-amber-400 mt-0.5">!</span>
                  Las narrativas de IA son generadas automaticamente y pueden contener errores de interpretacion.
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-amber-400 mt-0.5">!</span>
                  El sentimiento de noticias usa VADER (ingles) y tono GDELT — no es un modelo financiero especializado.
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-amber-400 mt-0.5">!</span>
                  Los indicadores tecnicos son descriptivos, no predictivos. Rendimientos pasados no garantizan resultados futuros.
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-amber-400 mt-0.5">!</span>
                  Este reporte NO constituye consejo de inversion. Es una herramienta de analisis para toma de decisiones informada.
                </li>
              </ul>
            </div>
          </div>
        </AccordionItem>

        {/* SECTION 4: How to Read This Report */}
        <AccordionItem
          id="howto"
          icon={<Layers className="w-4 h-4" />}
          title="Como Leer Este Reporte"
          subtitle="Guia rapida de cada seccion de la pagina"
          expanded={expandedSection === 'howto'}
          onToggle={() => toggle('howto')}
        >
          <div className="space-y-2">
            <SectionGuide
              name="Resumen Semanal"
              description="Vision general de la semana con sentimiento predominante (bullish/bearish/neutral), temas principales, y OHLCV del USD/COP."
            />
            <SectionGuide
              name="Indicadores Macro (Tarjetas)"
              description="4 variables clave (DXY, VIX, WTI, EMBI) con su valor actual, cambio porcentual, y tendencia respecto a SMA-20. Verde = a favor del COP, rojo = en contra."
            />
            <SectionGuide
              name="Graficos Macro"
              description="Series de tiempo de cada variable con SMA-20 (linea punteada cyan) y Bandas de Bollinger (area gris). Click para ver detalle con RSI y MACD."
            />
            <SectionGuide
              name="Regimen de Mercado"
              description="Clasificacion del entorno macro actual: risk_on (favorable para EM), risk_off (desfavorable), o transition. Basado en correlaciones y z-scores."
            />
            <SectionGuide
              name="Senales H1/H5"
              description="Resumen de las senales de los modelos de forecasting. H1 = prediccion diaria (9 modelos), H5 = prediccion semanal (2 modelos). Direccion SHORT = espera que USD/COP baje."
            />
            <SectionGuide
              name="Clusters de Noticias"
              description="Agrupacion automatica de noticias por tema usando similaridad de texto. Cada cluster muestra sentimiento promedio, titulos representativos, y fuentes originales con links."
            />
            <SectionGuide
              name="Timeline Diario"
              description="Analisis dia por dia de la semana. Cada entrada incluye: precio de cierre, sentimiento, eventos macroeconomicos, y resumen narrativo generado por IA."
            />
          </div>
        </AccordionItem>
      </div>
    </motion.div>
  );
}

// --- Sub-components ---

function AccordionItem({
  id,
  icon,
  title,
  subtitle,
  expanded,
  onToggle,
  children,
}: {
  id: string;
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  expanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-lg border border-gray-800/50 overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 px-4 py-3 bg-gray-800/20 hover:bg-gray-800/40 transition-colors text-left"
      >
        <span className="text-cyan-400">{icon}</span>
        <div className="flex-1 min-w-0">
          <span className="text-sm font-medium text-white">{title}</span>
          <span className="text-[10px] text-gray-500 block">{subtitle}</span>
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4 text-gray-500 shrink-0" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-500 shrink-0" />
        )}
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 py-4 border-t border-gray-800/30">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function DriverCard({
  icon,
  title,
  impact,
  description,
  example,
  metric,
}: {
  icon: React.ReactNode;
  title: string;
  impact: 'cop_strengthening' | 'cop_weakening';
  description: string;
  example: string;
  metric: string;
}) {
  const impactBg = impact === 'cop_strengthening' ? 'border-l-emerald-500' : 'border-l-red-500';
  const impactLabel = impact === 'cop_strengthening' ? 'Fortalece COP' : 'Debilita COP';
  const impactColor = impact === 'cop_strengthening' ? 'text-emerald-400' : 'text-red-400';

  return (
    <div className={`bg-gray-800/30 rounded-lg p-3 border border-gray-700/30 border-l-2 ${impactBg}`}>
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-xs font-medium text-white">{title}</span>
        </div>
        <span className={`text-[10px] font-medium ${impactColor}`}>
          {impact === 'cop_strengthening' ? '↓' : '↑'} {impactLabel}
        </span>
      </div>
      <p className="text-[11px] text-gray-400 leading-relaxed mb-1.5">{description}</p>
      <p className="text-[10px] text-cyan-400/70 leading-relaxed mb-1">{example}</p>
      <p className="text-[10px] text-gray-600">{metric}</p>
    </div>
  );
}

function IndicatorExplainer({
  name,
  periods,
  description,
  interpretation,
}: {
  name: string;
  periods: string;
  description: string;
  interpretation: string;
}) {
  return (
    <div className="bg-gray-800/30 rounded-lg p-3 border border-gray-700/30">
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-white">{name}</span>
        <span className="text-[10px] text-gray-600 bg-gray-800/60 rounded px-1.5 py-0.5">{periods}</span>
      </div>
      <p className="text-[11px] text-gray-400 leading-relaxed mb-1.5">{description}</p>
      <p className="text-[10px] text-cyan-400/70 leading-relaxed">{interpretation}</p>
    </div>
  );
}

function PipelineStep({ step, text }: { step: number; text: string }) {
  return (
    <div className="flex items-start gap-2 text-[11px]">
      <span className="bg-cyan-500/20 text-cyan-400 rounded-full w-4 h-4 flex items-center justify-center text-[9px] font-bold shrink-0 mt-0.5">
        {step}
      </span>
      <span className="text-gray-400 leading-relaxed">{text}</span>
    </div>
  );
}

function SectionGuide({ name, description }: { name: string; description: string }) {
  return (
    <div className="flex items-start gap-2 bg-gray-800/20 rounded-lg px-3 py-2">
      <BarChart3 className="w-3 h-3 text-cyan-400 shrink-0 mt-0.5" />
      <div>
        <span className="text-xs font-medium text-white">{name}</span>
        <p className="text-[10px] text-gray-500 leading-relaxed mt-0.5">{description}</p>
      </div>
    </div>
  );
}
