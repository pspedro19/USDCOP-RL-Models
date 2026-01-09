'use client';

/**
 * Forecasting Page - Professional Mobile-First Centered Design
 * ============================================================
 * EXACT Landing Page patterns applied:
 * - Section: w-full flex flex-col items-center
 * - Container: w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8
 * - Text: text-center + max-w-2xl mx-auto for descriptions
 * - Generous vertical padding: py-20 sm:py-28 lg:py-32
 */

import { GlobalNavbar } from '@/components/navigation/GlobalNavbar';
import { ForecastingDashboard } from '@/components/forecasting';
import { Badge } from '@/components/ui/badge';
import { Clock, Calendar, TrendingUp } from 'lucide-react';
import { useState, useEffect } from 'react';

function MarketStatusBadge() {
  const [status, setStatus] = useState<'open' | 'closed'>('closed');

  useEffect(() => {
    const checkMarket = () => {
      const now = new Date();
      const hour = now.getHours();
      const day = now.getDay();
      const isWeekday = day >= 1 && day <= 5;
      const isMarketHours = hour >= 8 && hour < 13;
      setStatus(isWeekday && isMarketHours ? 'open' : 'closed');
    };
    checkMarket();
    const interval = setInterval(checkMarket, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Badge
      variant="outline"
      className={status === 'open'
        ? 'bg-green-500/10 text-green-400 border-green-500/30 text-xs px-3 py-1'
        : 'bg-slate-500/10 text-slate-400 border-slate-500/30 text-xs px-3 py-1'
      }
    >
      <span className={`w-1.5 h-1.5 rounded-full mr-2 inline-block ${status === 'open' ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`} />
      {status === 'open' ? 'Mercado Abierto' : 'Cerrado'}
    </Badge>
  );
}

export default function ForecastingPage() {
  const [lastUpdate, setLastUpdate] = useState<string>('--:--:--');

  useEffect(() => {
    setLastUpdate(new Date().toLocaleTimeString());
    const interval = setInterval(() => {
      setLastUpdate(new Date().toLocaleTimeString());
    }, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-[#0a0a0f]">
      {/* Global Navigation - Fixed at top */}
      <GlobalNavbar currentPage="forecasting" />

      {/* Background Effects - z-0 to stay behind content */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none z-0">
        <div className="absolute top-20 left-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-[128px]" />
        <div className="absolute bottom-1/3 right-1/4 w-80 h-80 bg-pink-500/8 rounded-full blur-[100px]" />
      </div>

      {/* Main Content - proper offset from fixed navbar */}
      <main className="relative z-10">

        {/* Hero Header Section - inline style to force padding below navbar */}
        <section className="w-full relative flex flex-col items-center pb-6 sm:pb-8 border-b border-slate-800/50" style={{ paddingTop: '96px' }}>
          <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            {/* Title - Gradient text */}
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight mb-4 sm:mb-6">
              <span className="bg-gradient-to-r from-purple-400 via-pink-500 to-purple-400 bg-clip-text text-transparent">
                Forecasting Semanal
              </span>
            </h1>

            {/* Subtitle - Constrained width for readability */}
            <p className="text-sm sm:text-base lg:text-lg text-slate-400 max-w-2xl mx-auto mb-4 sm:mb-6 leading-relaxed">
              Predicciones de precio USD/COP utilizando modelos de Machine Learning avanzados
            </p>

            {/* Badges Row - Centered with flex */}
            <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3">
              <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/30 text-xs px-3 py-1">
                <TrendingUp className="w-3 h-3 mr-1.5" />
                ML PREDICTIONS
              </Badge>
              <MarketStatusBadge />
              {/* Update Time inline with badges */}
              <span className="inline-flex items-center gap-1.5 text-xs text-slate-500">
                <Clock className="w-3 h-3" />
                {lastUpdate}
              </span>
            </div>
          </div>
        </section>

        {/* Main Forecasting Dashboard Section */}
        <section className="w-full flex flex-col items-center py-8 sm:py-10 lg:py-12">
          <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            <ForecastingDashboard />
          </div>
        </section>

        {/* Footer - Compact */}
        <footer className="w-full flex flex-col items-center py-8 sm:py-10 border-t border-slate-800/50 bg-slate-950/50">
          <div className="w-full max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <p className="text-sm sm:text-base text-slate-400 font-medium">
              USDCOP Weekly Forecasting System
            </p>
            <p className="mt-2 text-xs sm:text-sm text-slate-500 max-w-lg mx-auto">
              Powered by Bayesian Regression, XGBoost, LightGBM, CatBoost & Hybrid Ensembles
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}
