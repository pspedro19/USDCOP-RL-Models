"use client";

import { motion } from "framer-motion";
import { Brain, Globe, LineChart, LucideIcon } from "lucide-react";
import { useLanguage } from "@/contexts/LanguageContext";

interface FeatureCardProps {
  icon: LucideIcon;
  iconColor: string;
  iconBgColor: string;
  title: string;
  description: string;
  delay: number;
}

function FeatureCard({
  icon: Icon,
  iconColor,
  iconBgColor,
  title,
  description,
  delay,
}: FeatureCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-100px" }}
      transition={{ duration: 0.5, delay }}
      className="group relative rounded-2xl border border-slate-800 bg-[#0c1020] p-8 sm:p-10 transition-all duration-300 hover:border-slate-700 hover:shadow-xl hover:shadow-slate-900/50"
    >
      {/* Larger icon container */}
      <div
        className={`mb-6 inline-flex h-16 w-16 items-center justify-center rounded-xl ${iconBgColor}`}
      >
        <Icon className={`h-8 w-8 ${iconColor}`} />
      </div>
      <h3 className="mb-4 text-xl font-bold text-white">{title}</h3>
      <p className="text-base leading-relaxed text-slate-400">{description}</p>
    </motion.div>
  );
}

export default function Features() {
  const { t } = useLanguage();

  const features = [
    {
      icon: Brain,
      iconColor: "text-cyan-400",
      iconBgColor: "bg-cyan-400/10",
      title: t.features.f1_title,
      description: t.features.f1_desc,
    },
    {
      icon: Globe,
      iconColor: "text-emerald-400",
      iconBgColor: "bg-emerald-400/10",
      title: t.features.f2_title,
      description: t.features.f2_desc,
    },
    {
      icon: LineChart,
      iconColor: "text-purple-400",
      iconBgColor: "bg-purple-400/10",
      title: t.features.f3_title,
      description: t.features.f3_desc,
    },
  ];

  return (
    <section id="features" className="w-full relative bg-[#0c0c14] py-40 sm:py-52 lg:py-64 border-t border-slate-800/50 flex flex-col items-center">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Section header with proper breathing room */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.5 }}
          className="mb-20 sm:mb-24 lg:mb-28 text-center"
        >
          <h2 className="text-3xl font-bold text-white sm:text-4xl lg:text-5xl">
            {t.features.title}
          </h2>
          <p className="mt-6 text-slate-400 max-w-2xl mx-auto text-base sm:text-lg">
            Tecnología de punta para maximizar tu rendimiento en el mercado de divisas.
          </p>
        </motion.div>

        {/* Cards with more gap */}
        <div className="grid grid-cols-1 gap-10 sm:gap-12 lg:grid-cols-3 lg:gap-10">
          {features.map((feature, index) => (
            <FeatureCard
              key={feature.title}
              icon={feature.icon}
              iconColor={feature.iconColor}
              iconBgColor={feature.iconBgColor}
              title={feature.title}
              description={feature.description}
              delay={index * 0.1}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
