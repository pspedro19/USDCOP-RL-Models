"use client";

import { motion } from "framer-motion";
import { useLanguage } from "@/contexts/LanguageContext";

interface MetricItem {
  value: string;
  labelKey: "sharpe" | "winRate" | "drawdown" | "volatility";
  gradient: string;
}

// Fixed: "High" changed to numeric, colors follow UX logic
const metrics: MetricItem[] = [
  {
    value: "3.24",
    labelKey: "sharpe",
    gradient: "from-emerald-400 to-green-500", // Green = good
  },
  {
    value: "68.4%",
    labelKey: "winRate",
    gradient: "from-emerald-400 to-green-500", // Green = good
  },
  {
    value: "-4.2%",
    labelKey: "drawdown",
    gradient: "from-rose-400 to-red-500", // Red = risk metric
  },
  {
    value: "92%",
    labelKey: "volatility",
    gradient: "from-sky-400 to-cyan-500", // Neutral blue = informational
  },
];

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 24 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.5,
      ease: [0.25, 0.46, 0.45, 0.94],
    },
  },
};

export function Metrics() {
  const { t } = useLanguage();

  return (
    <section className="w-full py-40 sm:py-52 lg:py-64 bg-slate-950 flex flex-col items-center">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, margin: "-100px" }}
          className="grid grid-cols-2 gap-12 md:grid-cols-4 md:gap-16"
        >
          {metrics.map((metric) => (
            <motion.div
              key={metric.labelKey}
              variants={itemVariants}
              className="flex flex-col items-center text-center"
            >
              <span
                className={`bg-gradient-to-r ${metric.gradient} bg-clip-text text-5xl sm:text-6xl font-bold tracking-tight text-transparent`}
              >
                {metric.value}
              </span>
              <span className="mt-4 text-xs font-semibold uppercase tracking-[0.2em] text-slate-500">
                {t.credibility[metric.labelKey]}
              </span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}

export default Metrics;
