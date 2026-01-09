"use client";

import { motion } from "framer-motion";
import { Sparkles, Clock } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

export function Pricing() {
  const { language } = useLanguage();

  const content = {
    es: {
      badge: "Acceso Limitado",
      title: "Early Access",
      subtitle: "Sé parte de los primeros en operar con inteligencia artificial en el mercado USD/COP",
      cta: "Solicitar Acceso",
      coming: "Apertura al público",
      comingSoon: "Próximamente",
      spots: "Cupos limitados disponibles"
    },
    en: {
      badge: "Limited Access",
      title: "Early Access",
      subtitle: "Be among the first to trade with artificial intelligence in the USD/COP market",
      cta: "Request Access",
      coming: "Open to Public",
      comingSoon: "Coming Soon",
      spots: "Limited spots available"
    }
  };

  const t = content[language];

  return (
    <section id="pricing" className="w-full bg-slate-950 py-40 sm:py-52 lg:py-64 border-t border-slate-800/50 flex flex-col items-center">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="flex flex-col items-center text-center"
        >
          {/* Badge */}
          <div className="inline-flex items-center gap-2 mb-10 px-4 py-2 rounded-full border border-emerald-500/30 bg-emerald-500/10">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <span className="text-sm font-medium text-emerald-400 uppercase tracking-wider">
              {t.badge}
            </span>
          </div>

          {/* Title */}
          <h2 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-10">
            {t.title}
          </h2>

          {/* Subtitle */}
          <p className="text-lg sm:text-xl text-slate-400 max-w-2xl text-center mb-14 leading-relaxed">
            {t.subtitle}
          </p>

          {/* CTA Button */}
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="px-10 py-5 bg-white text-black font-semibold text-lg rounded-xl hover:bg-gray-100 transition-all duration-200 shadow-lg shadow-white/10 mb-16"
          >
            {t.cta}
          </motion.button>

          {/* Coming Soon Badge */}
          <div className="flex flex-col items-center gap-4">
            <div className="flex items-center gap-3 text-slate-500">
              <Clock className="w-5 h-5" />
              <span className="text-base font-medium">{t.coming}</span>
            </div>
            <span className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-slate-400 to-slate-500 bg-clip-text text-transparent">
              {t.comingSoon}
            </span>
            <p className="text-sm text-slate-600 mt-2">
              {t.spots}
            </p>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default Pricing;
