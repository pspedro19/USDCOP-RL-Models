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
            badge: "Verified in production • Pre-registered statistical gates",
            title_start: "Quantitative USD/COP signals,",
            title_highlight: "verified in production",
            subtitle: "A weekly system with institutional risk management. Every strategy passes pre-registered statistical gates before touching real money — and we publish the results, win or lose.",
            cta_primary: "See signals for free",
            cta_secondary: "How we verify it",
            scarcity: "Most weeks the right answer is NOT to trade — and the system says so"
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
            f1_desc: "A supervised ensemble generates the weekly forecast; pre-registered statistical gates decide IF it trades.",
            f2_title: "Macro-Economic Awareness",
            f2_desc: "Real-time ingestion of Brent Oil, DXY, and local political sentiment analysis.",
            f3_title: "Institutional Backtesting",
            f3_desc: "Out-of-sample backtests deflated by number of attempts (Deflated Sharpe) — never the headline; the LIVE forward record is."
        },
        howItWorks: {
            title: "Deployment Process",
            step1_title: "Connect Broker",
            step1_desc: "API integration with supported execution venues.",
            step2_title: "Risk Profile",
            step2_desc: "Define your max drawdown and daily stop-loss limits.",
            step3_title: "Executable Signal",
            step3_desc: "Direction, entry, stop and target — or auto-execution on YOUR exchange (Auto plan).",
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
            q1: "Can I lose money?",
            a1: "Yes. Every trade risks capital; our risk management caps the loss per signal (hard stop) but cannot eliminate it. If anyone promises otherwise, run.",
            q2: "What if the system stops working?",
            a2: "Every strategy has a pre-signed withdrawal protocol: loss and drawdown thresholds defined BEFORE going live. If it stops working, it is retired — and you will see it published.",
            q3: "Do you hold my money?",
            a3: "Never. Auto-execution uses YOUR exchange's API keys WITHOUT withdrawal permission — keys that can withdraw funds are rejected at registration.",
            q4: "Is this financial advice?",
            a4: "No. This is informational and educational quantitative content. The decision, the sizing and the risk are yours. See our full risk disclosure.",
            q5: "How do I cancel?",
            a5: "Anytime, from your account. Paid access remains until the end of the billing period; your own history is never deleted."
        },
        footer: {
            tagline: "Verifiable quantitative signals. Pre-registered gates, published results — win or lose.",
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
            badge: "Verificado en producción • Gates estadísticos pre-registrados",
            title_start: "Señales cuantitativas USD/COP,",
            title_highlight: "verificadas en producción",
            subtitle: "Sistema semanal con gestión de riesgo institucional. Cada estrategia pasa gates estadísticos pre-registrados antes de tocar dinero real — y publicamos los resultados, ganen o pierdan.",
            cta_primary: "Ver señales gratis",
            cta_secondary: "Cómo lo verificamos",
            scarcity: "La mayoría de las semanas la respuesta correcta es NO operar — y el sistema lo dice"
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
            f1_desc: "Un ensemble supervisado genera el forecast semanal; gates estadísticos pre-registrados deciden SI se opera.",
            f2_title: "Conciencia Macroeconómica",
            f2_desc: "Ingesta en tiempo real de Petróleo Brent, DXY y análisis de sentimiento político local.",
            f3_title: "Backtesting Institucional",
            f3_desc: "Backtests fuera de muestra deflactados por número de intentos (Deflated Sharpe) — nunca el titular; el registro LIVE forward lo es."
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
            q1: "¿Puedo perder dinero?",
            a1: "Sí. Toda operación arriesga capital; la gestión de riesgo acota la pérdida por señal (hard stop) pero no la elimina. Si alguien te promete lo contrario, corre.",
            q2: "¿Qué pasa si el sistema deja de funcionar?",
            a2: "Cada estrategia tiene un protocolo de retiro pre-firmado: umbrales de pérdida y drawdown definidos ANTES de operar. Si deja de funcionar, se retira — y lo verás publicado.",
            q3: "¿Custodian mi dinero?",
            a3: "Jamás. La ejecución automática usa llaves API de TU exchange SIN permiso de retiro — las llaves que pueden retirar fondos se rechazan al registrarlas.",
            q4: "¿Esto es asesoría financiera?",
            a4: "No. Es contenido cuantitativo informativo y educativo. La decisión, el tamaño y el riesgo son tuyos. Lee la divulgación de riesgo completa.",
            q5: "¿Cómo cancelo?",
            a5: "Cuando quieras, desde tu cuenta. El acceso pago dura hasta el fin del período facturado; tu histórico nunca se borra."
        },
        footer: {
            tagline: "Señales cuantitativas verificables. Gates pre-registrados, resultados publicados — ganen o pierdan.",
            risk: "Advertencia de Riesgo: Operar divisas conlleva un alto nivel de riesgo y puede no ser adecuado para todos los inversores. El rendimiento pasado no garantiza resultados futuros.",
            rights: "© 2026 USDCOP AI Trading Platform. Todos los derechos reservados.",
            privacy: "Politica de Privacidad",
            terms: "Terminos de Servicio"
        }
    }
};
