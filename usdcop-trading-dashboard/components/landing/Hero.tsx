"use client";

import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, Zap, TrendingUp, Terminal } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';

// Animation variants
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.15,
      delayChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.6,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
};

const terminalVariants = {
  hidden: { opacity: 0, x: 50, scale: 0.95 },
  visible: {
    opacity: 1,
    x: 0,
    scale: 1,
    transition: {
      duration: 0.8,
      ease: [0.25, 0.46, 0.45, 0.94],
      delay: 0.5,
    },
  },
};

// Trading log entries for terminal animation
const tradingLogs = [
  { time: '09:32:15.847', type: 'SIGNAL', message: 'PPO_v3 BUY signal confidence: 0.847', color: 'text-emerald-400' },
  { time: '09:32:15.849', type: 'EXEC', message: 'Order submitted: BUY 0.5 lots @ 4,215.32', color: 'text-cyan-400' },
  { time: '09:32:15.851', type: 'FILL', message: 'Order filled: 0.5 lots @ 4,215.30 (slip: -0.02)', color: 'text-green-400' },
  { time: '09:32:16.102', type: 'RISK', message: 'Position exposure: 2.3% | Max DD: 0.8%', color: 'text-yellow-400' },
  { time: '09:32:17.445', type: 'MODEL', message: 'A2C_macro updating on DXY shift +0.12%', color: 'text-purple-400' },
  { time: '09:32:18.223', type: 'SIGNAL', message: 'SAC_HF HOLD signal confidence: 0.923', color: 'text-blue-400' },
  { time: '09:32:19.567', type: 'P&L', message: 'Unrealized P&L: +$127.45 (+0.25%)', color: 'text-emerald-400' },
  { time: '09:32:20.891', type: 'MACRO', message: 'Brent crude +1.2% | Sentiment: Bullish COP', color: 'text-orange-400' },
];

// Trust badges configuration
const trustBadges = [
  { icon: Shield, labelKey: 'security' as const, labelEs: 'Seguridad Bancaria', labelEn: 'Bank-grade Security' },
  { icon: Zap, labelKey: 'realtime' as const, labelEs: 'Tiempo Real', labelEn: 'Real-time' },
  { icon: TrendingUp, labelKey: 'ai' as const, labelEs: 'Impulsado por IA', labelEn: 'AI-Powered' },
];

function TerminalAnimation() {
  const [visibleLogs, setVisibleLogs] = useState<typeof tradingLogs>([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (currentIndex < tradingLogs.length) {
      const timer = setTimeout(() => {
        setVisibleLogs(prev => [...prev, tradingLogs[currentIndex]]);
        setCurrentIndex(prev => prev + 1);
      }, 800);
      return () => clearTimeout(timer);
    } else {
      // Reset and loop
      const resetTimer = setTimeout(() => {
        setVisibleLogs([]);
        setCurrentIndex(0);
      }, 3000);
      return () => clearTimeout(resetTimer);
    }
  }, [currentIndex]);

  return (
    <motion.div
      variants={terminalVariants}
      className="relative w-full max-w-lg mx-auto lg:mx-0"
    >
      {/* Terminal glow effect - softer emerald/teal */}
      <div className="absolute -inset-1 bg-gradient-to-r from-emerald-500/15 via-teal-500/15 to-emerald-500/15 rounded-xl blur-xl opacity-60" />

      {/* Terminal window */}
      <div className="relative bg-[#0d0d12] border border-gray-800/80 rounded-xl overflow-hidden shadow-2xl">
        {/* Terminal header */}
        <div className="flex items-center gap-2 px-4 py-3 bg-[#121218] border-b border-gray-800/50">
          <div className="flex gap-1.5">
            <div className="w-3 h-3 rounded-full bg-red-500/80" />
            <div className="w-3 h-3 rounded-full bg-yellow-500/80" />
            <div className="w-3 h-3 rounded-full bg-green-500/80" />
          </div>
          <div className="flex items-center gap-2 ml-3">
            <Terminal className="w-4 h-4 text-gray-500" />
            <span className="text-xs text-gray-500 font-mono">usdcop-trading-agent.log</span>
          </div>
          <div className="ml-auto flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-xs text-emerald-500 font-mono">LIVE</span>
          </div>
        </div>

        {/* Terminal content */}
        <div className="p-4 h-64 sm:h-72 overflow-hidden font-mono text-xs sm:text-sm">
          <div className="space-y-1.5">
            {visibleLogs.map((log, index) => (
              <motion.div
                key={`${log.time}-${index}`}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3 }}
                className="flex items-start gap-2"
              >
                <span className="text-gray-600 shrink-0">[{log.time}]</span>
                <span className="text-gray-500 shrink-0 w-14">[{log.type}]</span>
                <span className={log.color}>{log.message}</span>
              </motion.div>
            ))}
            {/* Blinking cursor */}
            <div className="flex items-center gap-1 text-gray-600">
              <span>{'>'}</span>
              <span className="w-2 h-4 bg-emerald-500/80 animate-pulse" />
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default function Hero() {
  const { t, language } = useLanguage();

  return (
    <section className="w-full relative min-h-screen bg-[#0a0a0f] overflow-hidden flex flex-col items-center">
      {/* Background grid pattern */}
      <div
        className="absolute inset-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)
          `,
          backgroundSize: '60px 60px',
        }}
      />

      {/* Gradient glow effects - contained within hero */}
      <div className="absolute top-20 left-1/4 w-96 h-96 bg-emerald-500/10 rounded-full blur-[128px] pointer-events-none" />
      <div className="absolute bottom-1/3 right-1/4 w-80 h-80 bg-teal-500/8 rounded-full blur-[100px] pointer-events-none" />

      {/* Content container - properly structured with navbar offset */}
      <div className="relative z-10 flex-1 flex items-center w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 pt-28 sm:pt-32 lg:pt-36 pb-20 sm:pb-28 lg:pb-32">
        <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center">
          {/* Left content - contained text block */}
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="text-center lg:text-left"
          >
            {/* Badge */}
            <motion.div variants={itemVariants} className="inline-flex items-center gap-2 mb-6 sm:mb-8">
              <div className="flex items-center gap-2 px-4 py-2 rounded-full border border-emerald-500/30 bg-emerald-500/5 backdrop-blur-sm">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <span className="text-xs sm:text-sm font-medium text-emerald-400 tracking-wide uppercase">
                  {t.hero.badge}
                </span>
              </div>
            </motion.div>

            {/* Main headline */}
            <motion.h1
              variants={itemVariants}
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold tracking-tight mb-8"
            >
              <span className="text-white block mb-3">
                {t.hero.title_start}
              </span>
              <span className="bg-gradient-to-r from-emerald-400 via-teal-400 to-green-400 bg-clip-text text-transparent">
                {t.hero.title_highlight}
              </span>
            </motion.h1>

            {/* Subheadline */}
            <motion.p
              variants={itemVariants}
              className="text-base sm:text-lg md:text-xl text-gray-400 max-w-xl mx-auto lg:mx-0 mb-10 sm:mb-12 leading-relaxed"
            >
              {t.hero.subtitle}
            </motion.p>

            {/* CTAs */}
            <motion.div
              variants={itemVariants}
              className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4 sm:gap-5 mb-12 sm:mb-14"
            >
              <button className="w-full sm:w-auto px-8 py-4 bg-white text-black font-semibold rounded-lg hover:bg-gray-100 transition-all duration-200 hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-white/10">
                {t.hero.cta_primary}
              </button>
              <button className="w-full sm:w-auto px-8 py-4 border border-gray-700 text-white font-semibold rounded-lg hover:border-gray-500 hover:bg-white/5 transition-all duration-200">
                {t.hero.cta_secondary}
              </button>
            </motion.div>

            {/* Trust badges */}
            <motion.div
              variants={itemVariants}
              className="flex flex-wrap items-center justify-center lg:justify-start gap-6 sm:gap-8"
            >
              {trustBadges.map((badge) => (
                <div
                  key={badge.labelKey}
                  className="flex items-center gap-2 text-gray-400"
                >
                  <badge.icon className="w-4 h-4 sm:w-5 sm:h-5 text-emerald-500" />
                  <span className="text-xs sm:text-sm">
                    {language === 'es' ? badge.labelEs : badge.labelEn}
                  </span>
                </div>
              ))}
            </motion.div>

            {/* Scarcity indicator */}
            <motion.div
              variants={itemVariants}
              className="mt-8 sm:mt-10"
            >
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-md bg-amber-500/10 border border-amber-500/20">
                <span className="w-1.5 h-1.5 rounded-full bg-amber-500 animate-pulse" />
                <span className="text-xs text-amber-400/90 font-medium">
                  {t.hero.scarcity}
                </span>
              </div>
            </motion.div>
          </motion.div>

          {/* Right content - Terminal (contained in grid) */}
          <motion.div
            initial="hidden"
            animate="visible"
            className="relative w-full max-w-lg mx-auto lg:mx-0 lg:max-w-none"
          >
            <TerminalAnimation />
          </motion.div>
        </div>
      </div>

      {/* Bottom gradient fade - visual separator */}
      <div className="absolute bottom-0 left-0 right-0 h-40 bg-gradient-to-t from-slate-950 via-[#0a0a0f]/80 to-transparent pointer-events-none" />
    </section>
  );
}
