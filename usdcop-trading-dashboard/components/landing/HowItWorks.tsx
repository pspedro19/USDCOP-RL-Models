"use client";

import { motion } from "framer-motion";
import { Activity, BarChart3, Brain, Zap } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.3 },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 32 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: [0.25, 0.46, 0.45, 0.94] },
  },
};

export function HowItWorks() {
  const { language } = useLanguage();

  const content = {
    es: {
      title: "Proceso de Inferencia y Señales",
      subtitle: "Tecnología de vanguardia en Machine Learning aplicada al mercado USD/COP",
      steps: [
        {
          number: 1,
          icon: Activity,
          title: "Monitoreo en Vivo",
          description: "Captura continua de datos del mercado USD/COP durante la sesión activa de 8:00 AM a 12:55 PM (hora Colombia)."
        },
        {
          number: 2,
          icon: BarChart3,
          title: "Análisis Estadístico",
          description: "Técnicas avanzadas de análisis de series temporales, detección de regímenes de volatilidad y correlaciones macro."
        },
        {
          number: 3,
          icon: Brain,
          title: "Inferencia ML",
          description: "Modelos de Reinforcement Learning (PPO, SAC, A2C) entrenados con datos históricos para reconocimiento de patrones."
        },
        {
          number: 4,
          icon: Zap,
          title: "Señales en Tiempo Real",
          description: "Generación de señales BUY/SELL/HOLD con niveles de confianza y métricas de riesgo actualizadas cada tick."
        }
      ]
    },
    en: {
      title: "Inference & Signals Process",
      subtitle: "Cutting-edge Machine Learning technology applied to the USD/COP market",
      steps: [
        {
          number: 1,
          icon: Activity,
          title: "Live Monitoring",
          description: "Continuous USD/COP market data capture during the active session from 8:00 AM to 12:55 PM (Colombia time)."
        },
        {
          number: 2,
          icon: BarChart3,
          title: "Statistical Analysis",
          description: "Advanced time-series analysis techniques, volatility regime detection, and macro correlations."
        },
        {
          number: 3,
          icon: Brain,
          title: "ML Inference",
          description: "Reinforcement Learning models (PPO, SAC, A2C) trained on historical data for pattern recognition."
        },
        {
          number: 4,
          icon: Zap,
          title: "Real-Time Signals",
          description: "BUY/SELL/HOLD signal generation with confidence levels and risk metrics updated every tick."
        }
      ]
    }
  };

  const t = content[language];

  return (
    <section
      id="how-it-works"
      className="w-full relative overflow-hidden bg-slate-950 py-40 sm:py-52 lg:py-64 border-t border-slate-800/50 flex flex-col items-center"
    >
      <div className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-20 sm:mb-24 lg:mb-28 text-center flex flex-col items-center"
        >
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl lg:text-5xl">
            {t.title}
          </h2>
          <p className="mt-6 text-slate-400 max-w-2xl text-base sm:text-lg">
            {t.subtitle}
          </p>
          {/* Decorative underline */}
          <div className="mt-8 flex items-center justify-center gap-1">
            <div className="h-1 w-8 rounded-full bg-gradient-to-r from-transparent to-teal-500" />
            <div className="h-1 w-16 rounded-full bg-gradient-to-r from-teal-500 to-emerald-500" />
            <div className="h-1 w-8 rounded-full bg-gradient-to-r from-emerald-500 to-transparent" />
          </div>
        </motion.div>

        {/* Steps Grid */}
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-50px" }}
          className="grid grid-cols-1 gap-16 sm:grid-cols-2 sm:gap-20 lg:grid-cols-4 lg:gap-12"
        >
          {t.steps.map((step) => {
            const Icon = step.icon;
            return (
              <motion.div
                key={step.number}
                variants={itemVariants}
                className="group relative flex flex-col items-center text-center"
              >
                {/* Number Circle */}
                <div className="relative mb-8">
                  <div className="relative flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-teal-500 to-emerald-500 shadow-lg shadow-emerald-500/25">
                    <span className="text-2xl font-bold text-white">
                      {step.number}
                    </span>
                  </div>
                  {/* Icon Badge */}
                  <div className="absolute -bottom-1 -right-1 flex h-10 w-10 items-center justify-center rounded-full border-4 border-slate-950 bg-slate-800">
                    <Icon className="h-5 w-5 text-emerald-400" />
                  </div>
                </div>

                {/* Content */}
                <h3 className="mb-3 text-xl font-semibold text-white">
                  {step.title}
                </h3>
                <p className="max-w-xs text-sm leading-relaxed text-slate-400">
                  {step.description}
                </p>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Trading Hours Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: 0.6 }}
          className="mt-20 flex justify-center"
        >
          <div className="inline-flex items-center gap-3 px-6 py-3 rounded-full border border-emerald-500/30 bg-emerald-500/5">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span className="text-sm text-slate-300">
              {language === 'es'
                ? 'Sesión activa: Lun-Vie 8:00 AM - 12:55 PM (COT)'
                : 'Active session: Mon-Fri 8:00 AM - 12:55 PM (COT)'}
            </span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default HowItWorks;
