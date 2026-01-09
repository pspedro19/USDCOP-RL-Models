export type Language = 'en' | 'es';

export const translations = {
    en: {
        nav: {
            login: "Member Login",
            request: "Request Access",
            features: "Features",
            howItWorks: "How it Works",
            pricing: "Pricing",
            faq: "FAQ"
        },
        hero: {
            badge: "Restricted Access • Institutional Grade",
            title_start: "Trade USD/COP with",
            title_highlight: "Institutional Grade AI",
            subtitle: "3 RL strategies specifically trained on high-frequency Colombian Peso data, competing in real-time to maximize your Alpha.",
            cta_primary: "Start Trading Now",
            cta_secondary: "View Live Demo",
            scarcity: "Access: Private Beta (3/50 spots left)"
        },
        credibility: {
            title: "Trusted by Data",
            sharpe: "Sharpe Ratio",
            winRate: "Win Rate",
            drawdown: "Max Drawdown",
            volatility: "Volatility Capture",
            tech_stack: "Powered by strict institutional infrastructure"
        },
        features: {
            title: "Why Choose Our Algorithms?",
            f1_title: "Multi-Model Competition",
            f1_desc: "Reinforcement Learning, supervised ML, and LLMs compete for every trade signal.",
            f2_title: "Macro-Economic Awareness",
            f2_desc: "Real-time ingestion of Brent Oil, DXY, and local political sentiment analysis.",
            f3_title: "Institutional Backtesting",
            f3_desc: "Validated on 10 years of tick-by-tick data including the 2020 and 2022 volatility spikes."
        },
        howItWorks: {
            title: "Deployment Process",
            step1_title: "Connect Broker",
            step1_desc: "API integration with supported execution venues.",
            step2_title: "Risk Profile",
            step2_desc: "Define your max drawdown and daily stop-loss limits.",
            step3_title: "AI Execution",
            step3_desc: "The RL agent takes over execution with <4ms latency.",
            step4_title: "Live Monitoring",
            step4_desc: "Watch performance in real-time on your institutional dashboard."
        },
        pricing: {
            title: "Access Tiers",
            monthly: "/month",
            starter_title: "Starter",
            starter_price: "$0",
            starter_desc: "Perfect for backtesting and paper trading.",
            pro_title: "Pro Trader",
            pro_price: "$199",
            pro_desc: "Live execution for individual accredited investors.",
            inst_title: "Institutional",
            inst_price: "Custom",
            inst_desc: "Full dedicated nodes and API access for firms.",
            features: {
                paper: "Paper Trading",
                backtest: "Backtesting Engine",
                live: "Live Execution",
                priority: "Priority Support",
                api: "API Access",
                dedicated: "Dedicated Server"
            },
            cta: "Select Plan"
        },
        faq: {
            title: "Frequently Asked Questions",
            q1: "Is FX trading legal in Colombia?",
            a1: "Yes, trading derivatives on foreign currency is legal for Colombian residents through regulated international brokers.",
            q2: "What is the minimum capital required?",
            a2: "We recommend a minimum of $5,000 USD for optimal position sizing and risk management.",
            q3: "How do I withdraw profits?",
            a3: "Withdrawals are handled directly through your connected broker. We never hold client funds.",
            q4: "What happens if the system fails?",
            a4: "Our 'Kill Switch' technology automatically liquidates positions if connection is lost or anomaly is detected."
        },
        footer: {
            tagline: "Advanced algorithmic trading systems for emerging market currencies.",
            risk: "Risk Warning: Trading foreign exchange carries a high level of risk and may not be suitable for all investors. Past performance is not indicative of future results.",
            rights: "© 2026 USDCOP AI Trading Platform. All rights reserved.",
            privacy: "Privacy Policy",
            terms: "Terms of Service"
        }
    },
    es: {
        nav: {
            login: "Ingreso Miembros",
            request: "Solicitar Acceso",
            features: "Características",
            howItWorks: "Cómo Funciona",
            pricing: "Planes",
            faq: "Preguntas"
        },
        hero: {
            badge: "Acceso Restringido • Grado Institucional",
            title_start: "Opera USD/COP con",
            title_highlight: "Inteligencia Artificial",
            subtitle: "3 estrategias de RL entrenadas específicamente en datos de alta frecuencia del Peso Colombiano, compitiendo en tiempo real para maximizar tu Alpha.",
            cta_primary: "Comenzar Ahora",
            cta_secondary: "Ver Demo en Vivo",
            scarcity: "Acceso: Beta Privada (3/50 cupos restantes)"
        },
        credibility: {
            title: "Respaldado por Datos",
            sharpe: "Ratio de Sharpe",
            winRate: "Tasa de Acierto",
            drawdown: "Drawdown Máximo",
            volatility: "Captura de Volatilidad",
            tech_stack: "Potenciado por infraestructura institucional estricta"
        },
        features: {
            title: "¿Por Qué Nuestros Algoritmos?",
            f1_title: "Competencia Multi-Modelo",
            f1_desc: "Reinforcement Learning, ML supervisado y LLMs compiten por cada señal de trading.",
            f2_title: "Conciencia Macroeconómica",
            f2_desc: "Ingesta en tiempo real de Petróleo Brent, DXY y análisis de sentimiento político local.",
            f3_title: "Backtesting Institucional",
            f3_desc: "Validado en 10 años de datos tick-by-tick incluyendo los picos de volatilidad de 2020 y 2022."
        },
        howItWorks: {
            title: "Proceso de Despliegue",
            step1_title: "Conecta tu Broker",
            step1_desc: "Integración vía API con plataformas de ejecución soportadas.",
            step2_title: "Perfil de Riesgo",
            step2_desc: "Define tu límite de pérdida diaria y drawdown máximo.",
            step3_title: "Ejecución IA",
            step3_desc: "El agente de RL toma el control de la ejecución con latencia <4ms.",
            step4_title: "Monitoreo en Vivo",
            step4_desc: "Observa el rendimiento en tiempo real en tu dashboard institucional."
        },
        pricing: {
            title: "Niveles de Acceso",
            monthly: "/mes",
            starter_title: "Starter",
            starter_price: "$0",
            starter_desc: "Perfecto para backtesting y paper trading.",
            pro_title: "Pro Trader",
            pro_price: "$199",
            pro_desc: "Ejecución en vivo para inversores acreditados.",
            inst_title: "Institucional",
            inst_price: "Personalizado",
            inst_desc: "Nodos dedicados completos y acceso API para firmas.",
            features: {
                paper: "Paper Trading",
                backtest: "Motor de Backtesting",
                live: "Ejecución en Vivo",
                priority: "Soporte Prioritario",
                api: "Acceso API",
                dedicated: "Servidor Dedicado"
            },
            cta: "Seleccionar Plan"
        },
        faq: {
            title: "Preguntas Frecuentes",
            q1: "¿Es legal operar FX en Colombia?",
            a1: "Sí, operar derivados de divisas es legal para residentes colombianos a través de brokers internacionales regulados.",
            q2: "¿Cuál es el capital mínimo requerido?",
            a2: "Recomendamos un mínimo de $5,000 USD para una gestión de riesgo y tamaño de posición óptimos.",
            q3: "¿Cómo retiro mis ganancias?",
            a3: "Los retiros se manejan directamente a través de tu broker conectado. Nosotros nunca custodiamos fondos.",
            q4: "¿Qué pasa si el sistema falla?",
            a4: "Nuestra tecnología 'Kill Switch' liquida posiciones automáticamente si se pierde la conexión o se detecta una anomalía."
        },
        footer: {
            tagline: "Sistemas avanzados de trading algoritmico para divisas de mercados emergentes.",
            risk: "Advertencia de Riesgo: Operar divisas conlleva un alto nivel de riesgo y puede no ser adecuado para todos los inversores. El rendimiento pasado no garantiza resultados futuros.",
            rights: "© 2026 USDCOP AI Trading Platform. Todos los derechos reservados.",
            privacy: "Politica de Privacidad",
            terms: "Terminos de Servicio"
        }
    }
};
