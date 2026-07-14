'use client';

/**
 * GM shell/common dictionary (CTR-GM-UI-001 · prototype Var B `T` lines 1777-1794).
 *
 * Every visible string of the chrome (TerminalShell + PublicChrome) lives here —
 * zero hardcoded copy in the components. EN pairs come from the prototype dict
 * (`T.en.nav/common/hub`); strings without a prototype pair get a sober EN
 * equivalent. Consume with `useGmT(GM_DICT)` from `lib/i18n/gm-core`.
 */
import { defineGmDict } from './gm-core';

export const GM_DICT = defineGmDict({
  es: {
    // brand (proper nouns — identical in both langs, kept in the dict per §i18n)
    brand: 'GlobalMarkets',
    terminal: 'Terminal',
    brandHome: 'GlobalMarkets — inicio',

    // sidebar / nav (prototype T.es.nav)
    quickAccess: 'Accesos',
    sections: 'Secciones',
    navHub: 'Inicio',
    navCatalog: 'Activos',
    navBacktest: 'Backtest',
    navSignals: 'Señales',
    navProduction: 'Producción',
    navForecasting: 'Forecasting',
    navAnalysis: 'Análisis',
    navExecution: 'SignalBridge',
    navAdmin: 'Admin',

    // topbar (prototype T.es.hub/common)
    myAssets: 'Mis activos',
    plans: 'Planes',
    cart: 'Carrito',
    openCart: 'Abrir carrito',
    goCatalog: 'Ver en el catálogo',

    // controls
    language: 'Idioma',
    menu: 'Menú',
    toggleMenu: 'Alternar menú',
    closeMenu: 'Cerrar menú',
    logout: 'Cerrar sesión',
    user: 'Usuario',

    // public chrome
    signIn: 'Iniciar sesión',
    signUp: 'Crear cuenta',
    legal: 'Legal',
    methodology: 'Metodología',
    terms: 'Términos',
    disclaimer:
      'Contenido informativo y educativo; no constituye asesoría financiera. Rendimientos ' +
      'pasados no garantizan resultados futuros. Operar divisas y criptoactivos implica ' +
      'riesgo de pérdida total del capital.',
    copyright: '© 2026 GlobalMarkets Terminal',
  },
  en: {
    brand: 'GlobalMarkets',
    terminal: 'Terminal',
    brandHome: 'GlobalMarkets — home',

    quickAccess: 'Quick access',
    sections: 'Sections',
    navHub: 'Home',
    navCatalog: 'Assets',
    navBacktest: 'Backtest',
    navSignals: 'Signals',
    navProduction: 'Production',
    navForecasting: 'Forecasting',
    navAnalysis: 'Analysis',
    navExecution: 'SignalBridge',
    navAdmin: 'Admin',

    myAssets: 'My assets',
    plans: 'Plans',
    cart: 'Cart',
    openCart: 'Open cart',
    goCatalog: 'View in catalog',

    language: 'Language',
    menu: 'Menu',
    toggleMenu: 'Toggle menu',
    closeMenu: 'Close menu',
    logout: 'Sign out',
    user: 'User',

    signIn: 'Sign in',
    signUp: 'Create account',
    legal: 'Legal',
    methodology: 'Methodology',
    terms: 'Terms',
    disclaimer:
      'Informational and educational content; not financial advice. Past performance ' +
      'does not guarantee future results. Trading currencies and crypto assets involves ' +
      'risk of total capital loss.',
    copyright: '© 2026 GlobalMarkets Terminal',
  },
});
