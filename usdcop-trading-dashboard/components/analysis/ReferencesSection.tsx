'use client';

import { motion } from 'framer-motion';
import { ExternalLink, Database, Newspaper, BarChart3, BookOpen } from 'lucide-react';
import type { WeeklyViewData, NewsArticleRef } from '@/lib/contracts/weekly-analysis.contract';

interface ReferencesSectionProps {
  weekData: WeeklyViewData;
}

interface SourceRef {
  name: string;
  description: string;
  url: string;
  variables?: string[];
  category: 'macro' | 'news' | 'methodology';
}

const MACRO_SOURCES: SourceRef[] = [
  {
    name: 'FRED (Federal Reserve Economic Data)',
    description: 'Tasas de interes (Fed Funds, UST10Y, UST2Y), inflacion (CPI US)',
    url: 'https://fred.stlouisfed.org/',
    variables: ['fedfunds', 'ust10y', 'ust2y', 'cpi_us'],
    category: 'macro',
  },
  {
    name: 'Banco de la Republica (BanRep)',
    description: 'Tasa de Politica Monetaria (TPM), IBR, EMBI Colombia',
    url: 'https://www.banrep.gov.co/es/estadisticas',
    variables: ['tpm', 'ibr', 'embi_col'],
    category: 'macro',
  },
  {
    name: 'Investing.com',
    description: 'DXY (Dollar Index), VIX, Oro, Brent, WTI, datos de mercado en tiempo real',
    url: 'https://www.investing.com/',
    variables: ['dxy', 'vix', 'gold', 'brent', 'wti'],
    category: 'macro',
  },
  {
    name: 'CBOE (Chicago Board Options Exchange)',
    description: 'Indice VIX (volatilidad implicita S&P 500)',
    url: 'https://www.cboe.com/tradable_products/vix/',
    variables: ['vix'],
    category: 'macro',
  },
  {
    name: 'ICE (Intercontinental Exchange)',
    description: 'Futuros de Brent Crude',
    url: 'https://www.theice.com/products/219/Brent-Crude-Futures',
    variables: ['brent'],
    category: 'macro',
  },
  {
    name: 'NYMEX (New York Mercantile Exchange)',
    description: 'Futuros de WTI Crude Oil',
    url: 'https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.html',
    variables: ['wti'],
    category: 'macro',
  },
  {
    name: 'JP Morgan / BanRep',
    description: 'EMBI+ Colombia (Emerging Markets Bond Index, spread de riesgo pais)',
    url: 'https://www.banrep.gov.co/es/estadisticas/indice-embi-colombia',
    variables: ['embi_col'],
    category: 'macro',
  },
  {
    name: 'BLS (Bureau of Labor Statistics)',
    description: 'Consumer Price Index (CPI) de Estados Unidos',
    url: 'https://www.bls.gov/cpi/',
    variables: ['cpi_us'],
    category: 'macro',
  },
  {
    name: 'DANE (Depto. Administrativo Nacional de Estadistica)',
    description: 'IPC Colombia (inflacion mensual y anual)',
    url: 'https://www.dane.gov.co/index.php/estadisticas-por-tema/precios-y-costos/indice-de-precios-al-consumidor-ipc',
    variables: ['cpi_col'],
    category: 'macro',
  },
];

const NEWS_SOURCES: SourceRef[] = [
  {
    name: 'GDELT Project',
    description: 'Base de datos global de eventos y noticias, tono de sentimiento por articulo',
    url: 'https://www.gdeltproject.org/',
    category: 'news',
  },
  {
    name: 'Investing.com (Noticias)',
    description: 'Articulos financieros sobre mercados emergentes, FX, commodities',
    url: 'https://www.investing.com/news/',
    category: 'news',
  },
  {
    name: 'Portafolio.co',
    description: 'Noticias economicas colombianas (economia, finanzas, negocios)',
    url: 'https://www.portafolio.co/',
    category: 'news',
  },
  {
    name: 'La Republica',
    description: 'Periodico financiero colombiano (finanzas, macroeconomia)',
    url: 'https://www.larepublica.co/',
    category: 'news',
  },
];

const METHODOLOGY_REFS: SourceRef[] = [
  {
    name: 'TwelveData API',
    description: 'Datos OHLCV intradía y diarios para USD/COP, USD/MXN, USD/BRL',
    url: 'https://twelvedata.com/',
    category: 'methodology',
  },
  {
    name: 'TimescaleDB',
    description: 'Base de datos de series de tiempo para almacenamiento de datos financieros',
    url: 'https://www.timescale.com/',
    category: 'methodology',
  },
];

export function ReferencesSection({ weekData }: ReferencesSectionProps) {
  // Collect unique article URLs from news clusters and daily entries
  const articleLinks: { title: string; source: string; url: string }[] = [];
  const seenUrls = new Set<string>();

  // From news intelligence clusters
  if (weekData.news_intelligence?.clusters) {
    for (const cluster of weekData.news_intelligence.clusters) {
      for (const article of cluster.articles) {
        if (article.url && !seenUrls.has(article.url)) {
          seenUrls.add(article.url);
          articleLinks.push({ title: article.title, source: article.source, url: article.url });
        }
      }
    }
  }

  // From top stories
  if (weekData.news_intelligence?.top_stories) {
    for (const story of weekData.news_intelligence.top_stories) {
      if (story.url && !seenUrls.has(story.url)) {
        seenUrls.add(story.url);
        articleLinks.push({ title: story.title, source: story.source, url: story.url });
      }
    }
  }

  // From daily entry highlights
  for (const entry of weekData.daily_entries || []) {
    for (const hl of entry.news_highlights || []) {
      if (hl.url && !seenUrls.has(hl.url)) {
        seenUrls.add(hl.url);
        articleLinks.push({ title: hl.title, source: hl.source, url: hl.url });
      }
    }
  }

  // From news context highlights
  if (weekData.news_context?.highlights) {
    for (const hl of weekData.news_context.highlights) {
      if (hl.url && !seenUrls.has(hl.url)) {
        seenUrls.add(hl.url);
        articleLinks.push({ title: hl.title, source: hl.source || hl.news_source || '', url: hl.url });
      }
    }
  }

  // Which macro sources are active this week
  const activeMacroKeys = new Set(Object.keys(weekData.macro_snapshots || {}));
  const activeMacroSources = MACRO_SOURCES.filter(
    s => !s.variables || s.variables.some(v => activeMacroKeys.has(v))
  );

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-gray-900/60 backdrop-blur-sm rounded-xl border border-gray-800/50 p-6"
    >
      <h2 className="text-base font-semibold text-white mb-5 flex items-center gap-2">
        <BookOpen className="w-5 h-5 text-cyan-400" />
        Referencias y Fuentes de Datos
      </h2>

      {/* Macro data sources */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2 mb-3">
          <Database className="w-4 h-4 text-blue-400" />
          Fuentes de Datos Macroeconomicos
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {activeMacroSources.map((src) => (
            <SourceLink key={src.name} source={src} />
          ))}
        </div>
      </div>

      {/* News sources */}
      <div className="mb-6">
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2 mb-3">
          <Newspaper className="w-4 h-4 text-amber-400" />
          Fuentes de Noticias
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {NEWS_SOURCES.map((src) => (
            <SourceLink key={src.name} source={src} />
          ))}
        </div>
      </div>

      {/* Article hyperlinks from this week's data */}
      {articleLinks.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2 mb-3">
            <ExternalLink className="w-4 h-4 text-emerald-400" />
            Articulos Citados Esta Semana ({articleLinks.length})
          </h3>
          <div className="space-y-1.5 max-h-64 overflow-y-auto pr-2">
            {articleLinks.map((link, i) => (
              <a
                key={i}
                href={link.url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-start gap-2 text-xs group hover:bg-gray-800/30 rounded px-2 py-1.5 transition-colors"
              >
                <span className="text-gray-600 shrink-0 mt-0.5">{i + 1}.</span>
                <div className="min-w-0 flex-1">
                  <span className="text-gray-300 group-hover:text-cyan-400 transition-colors line-clamp-1">
                    {link.title}
                  </span>
                  <span className="text-gray-600 text-[10px] block">{link.source}</span>
                </div>
                <ExternalLink className="w-3 h-3 text-gray-600 group-hover:text-cyan-400 shrink-0 mt-0.5" />
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Data infrastructure */}
      <div>
        <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2 mb-3">
          <BarChart3 className="w-4 h-4 text-purple-400" />
          Infraestructura de Datos
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
          {METHODOLOGY_REFS.map((src) => (
            <SourceLink key={src.name} source={src} />
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-5 pt-4 border-t border-gray-800/50">
        <p className="text-[10px] text-gray-600 leading-relaxed">
          Nota: Los datos macroeconomicos se actualizan automaticamente cada hora durante el horario de
          mercado (8:00-12:55 COT, Lun-Vie). Las noticias se ingestan 3 veces al dia. Los indicadores
          tecnicos (SMA, RSI, Bollinger) se calculan con ventanas de 5, 10, 20 y 50 dias usando EMA de
          Wilder para RSI. Las narrativas son generadas por inteligencia artificial y no constituyen
          consejo de inversion.
        </p>
      </div>
    </motion.div>
  );
}

function SourceLink({ source }: { source: SourceRef }) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-start gap-2.5 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 border border-gray-800/30 hover:border-cyan-500/20 px-3 py-2.5 transition-all group"
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-medium text-gray-200 group-hover:text-cyan-400 transition-colors">
            {source.name}
          </span>
          <ExternalLink className="w-2.5 h-2.5 text-gray-600 group-hover:text-cyan-400 shrink-0" />
        </div>
        <p className="text-[10px] text-gray-500 mt-0.5 leading-relaxed">{source.description}</p>
      </div>
    </a>
  );
}
