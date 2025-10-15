import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: ["class"],
  theme: {
    extend: {
      // === Bloomberg Terminal Professional Color System ===
      colors: {
        // Core Brand Colors
        terminal: {
          bg: "var(--terminal-bg)",
          surface: "var(--terminal-surface)",
          "surface-variant": "var(--terminal-surface-variant)",
          border: "var(--terminal-border)",
          accent: "var(--terminal-accent)",
          "accent-dim": "var(--terminal-accent-dim)",
          text: "var(--terminal-text)",
          "text-dim": "var(--terminal-text-dim)",
          "text-muted": "var(--terminal-text-muted)",
        },

        // Enhanced Market Semantics
        market: {
          up: "var(--market-up)",
          down: "var(--market-down)",
          neutral: "var(--market-neutral)",
        },

        // Professional Status System
        status: {
          live: "var(--status-live)",
          delayed: "var(--status-delayed)",
          offline: "var(--status-offline)",
          replay: "var(--status-replay)",
        },

        // Glassmorphism Background System
        glass: {
          primary: "var(--bg-glass-primary)",
          secondary: "var(--bg-glass-secondary)",
          accent: "var(--bg-glass-accent)",
          elevated: "var(--bg-elevated)",
          overlay: "var(--bg-overlay)",
          interactive: "var(--bg-interactive)",
        },

        // Enhanced Semantic Colors
        positive: "var(--positive)",
        "positive-dim": "var(--positive-dim)",
        "positive-bg": "var(--positive-bg)",
        negative: "var(--negative)",
        "negative-dim": "var(--negative-dim)",
        "negative-bg": "var(--negative-bg)",
        neutral: "var(--neutral)",
        "neutral-bg": "var(--neutral-bg)",
        "info-blue": "var(--info-blue)",
        "info-blue-bg": "var(--info-blue-bg)",

        // Chart System Colors
        chart: {
          grid: "var(--chart-grid)",
          axis: "var(--chart-axis)",
          volume: "var(--chart-volume)",
          bg: "var(--chart-bg)",
          overlay: "var(--chart-overlay)",
        },

        // Enhanced Text System
        text: {
          primary: "var(--text-primary)",
          secondary: "var(--text-secondary)",
          tertiary: "var(--text-tertiary)",
          inverse: "var(--text-inverse)",
        },

        // Shadcn UI Compatibility
        background: "var(--terminal-bg)",
        foreground: "var(--terminal-text)",
        card: {
          DEFAULT: "var(--terminal-surface)",
          foreground: "var(--terminal-text)",
        },
        popover: {
          DEFAULT: "var(--terminal-surface-variant)",
          foreground: "var(--terminal-text)",
        },
        primary: {
          DEFAULT: "var(--terminal-accent)",
          foreground: "var(--terminal-bg)",
        },
        secondary: {
          DEFAULT: "var(--terminal-surface-variant)",
          foreground: "var(--terminal-text)",
        },
        muted: {
          DEFAULT: "var(--terminal-surface)",
          foreground: "var(--terminal-text-muted)",
        },
        accent: {
          DEFAULT: "var(--terminal-accent)",
          foreground: "var(--terminal-bg)",
        },
        destructive: {
          DEFAULT: "var(--negative)",
          foreground: "var(--terminal-text)",
        },
        border: "var(--terminal-border)",
        input: "var(--terminal-surface-variant)",
        ring: "var(--terminal-accent)",
      },

      // === Professional Typography System ===
      fontFamily: {
        terminal: ["var(--font-terminal)"],
        system: ["var(--font-system)"],
        sans: ["var(--font-system)"],
        mono: ["var(--font-terminal)"],
      },

      fontSize: {
        "xs": ["0.75rem", { lineHeight: "1rem" }],
        "sm": ["0.875rem", { lineHeight: "1.25rem" }],
        "base": ["1rem", { lineHeight: "1.5rem" }],
        "lg": ["1.125rem", { lineHeight: "1.75rem" }],
        "xl": ["1.25rem", { lineHeight: "1.75rem" }],
        "2xl": ["1.5rem", { lineHeight: "2rem" }],
        "3xl": ["1.875rem", { lineHeight: "2.25rem" }],
        "4xl": ["2.25rem", { lineHeight: "2.5rem" }],
        "5xl": ["3rem", { lineHeight: "1" }],
        "6xl": ["3.75rem", { lineHeight: "1" }],
        "7xl": ["4.5rem", { lineHeight: "1" }],
        "8xl": ["6rem", { lineHeight: "1" }],
        "9xl": ["8rem", { lineHeight: "1" }],
        // Terminal-specific sizes
        "terminal-xs": ["0.6875rem", { lineHeight: "1rem", letterSpacing: "0.025em" }],
        "terminal-sm": ["0.8125rem", { lineHeight: "1.125rem", letterSpacing: "0.025em" }],
        "terminal-base": ["0.9375rem", { lineHeight: "1.375rem", letterSpacing: "0.025em" }],
        "terminal-lg": ["1.0625rem", { lineHeight: "1.5rem", letterSpacing: "0.025em" }],
      },

      // === Professional Box Shadow System ===
      boxShadow: {
        "glass-sm": "var(--shadow-glass-sm)",
        "glass-md": "var(--shadow-glass-md)",
        "glass-lg": "var(--shadow-glass-lg)",
        "glass-xl": "var(--shadow-glass-xl)",
        "glow-cyan": "var(--shadow-glow-cyan)",
        "glow-purple": "var(--shadow-glow-purple)",
        "glow-mixed": "var(--shadow-glow-mixed)",
        "market-up": "var(--market-up-glow)",
        "market-down": "var(--market-down-glow)",
        "status-live": "var(--status-live-glow)",
        "status-delayed": "var(--status-delayed-glow)",
        "status-offline": "var(--status-offline-glow)",
        "status-replay": "var(--status-replay-glow)",
        "hover-glow": "var(--hover-glow)",
        "hover-lift": "var(--hover-lift)",
        "terminal-glow": "0 0 5px rgba(245, 158, 11, 0.3), 0 0 10px rgba(245, 158, 11, 0.2), 0 0 15px rgba(245, 158, 11, 0.1)",
      },

      // === Advanced Backdrop Blur System ===
      backdropBlur: {
        "xs": "var(--blur-xs)",
        "sm": "var(--blur-sm)",
        "md": "var(--blur-md)",
        "lg": "var(--blur-lg)",
        "xl": "var(--blur-xl)",
        "2xl": "var(--blur-2xl)",
        "3xl": "var(--blur-3xl)",
        "professional": "var(--blur-xl) saturate(1.5) brightness(1.1)",
        "intense": "var(--blur-2xl) saturate(1.8) contrast(1.1)",
      },

      // === Professional Border System ===
      borderColor: {
        "glass": "rgba(6, 182, 212, 0.2)",
        "glass-strong": "rgba(6, 182, 212, 0.4)",
        "terminal": "var(--terminal-border)",
        "chart": "var(--chart-grid)",
      },

      // === Enhanced Gradient System ===
      backgroundImage: {
        "gradient-primary": "var(--gradient-primary)",
        "gradient-secondary": "var(--gradient-secondary)",
        "gradient-accent": "var(--gradient-accent)",
        "gradient-surface": "var(--gradient-surface)",
        "gradient-glass": "var(--gradient-glass)",
        "gradient-border": "var(--gradient-border)",
        "gradient-positive": "var(--gradient-positive)",
        "gradient-negative": "var(--gradient-negative)",
        "gradient-info": "var(--gradient-info)",
        // Terminal patterns
        "terminal-grid": "linear-gradient(rgba(6, 182, 212, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(6, 182, 212, 0.08) 1px, transparent 1px)",
        "terminal-radial": "radial-gradient(ellipse at center, rgba(6, 182, 212, 0.15) 0%, transparent 25%)",
      },

      // === Professional Spacing System ===
      spacing: {
        "18": "4.5rem",
        "88": "22rem",
        "112": "28rem",
        "128": "32rem",
        "144": "36rem",
        "terminal-xs": "0.125rem",
        "terminal-sm": "0.25rem",
        "terminal-md": "0.5rem",
        "terminal-lg": "0.75rem",
        "terminal-xl": "1rem",
        "terminal-2xl": "1.25rem",
        "terminal-3xl": "1.5rem",
        "safe-area-inset-bottom": "env(safe-area-inset-bottom)",
        "safe-area-inset-top": "env(safe-area-inset-top)",
        "safe-area-inset-left": "env(safe-area-inset-left)",
        "safe-area-inset-right": "env(safe-area-inset-right)",
      },

      // === Professional Border Radius System ===
      borderRadius: {
        "lg": "var(--radius)",
        "md": "calc(var(--radius) - 2px)",
        "sm": "calc(var(--radius) - 4px)",
        "terminal": "0.375rem",
        "terminal-lg": "0.75rem",
        "terminal-xl": "1rem",
        "glass": "1.5rem",
        "glass-lg": "2rem",
      },

      // === Professional Animation System ===
      animation: {
        // Existing animations
        "pulse": "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "ping": "ping 1s cubic-bezier(0, 0, 0.2, 1) infinite",
        "bounce": "bounce 1s infinite",
        "spin": "spin 1s linear infinite",

        // Bloomberg Terminal Professional Animations
        "terminal-blink": "terminal-blink 2s ease-in-out infinite",
        "terminal-scan": "terminal-scan 3s linear infinite",
        "data-flow": "data-flow 3s linear infinite",
        "background-pulse": "backgroundPulse 20s ease-in-out infinite",

        // Glassmorphism Animations
        "glass-slide-in": "glassSlideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
        "glass-entrance": "glassSlideIn 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
        "glass-float": "float-glass 6s ease-in-out infinite",
        "glass-hover": "hover-glow-glass 0.8s ease-in-out",
        "glass-spin": "spin-glass 2s linear infinite",

        // Price Update Animations
        "price-flash-up": "price-flash-up-glass 800ms cubic-bezier(0.4, 0, 0.2, 1)",
        "price-flash-down": "price-flash-down-glass 800ms cubic-bezier(0.4, 0, 0.2, 1)",
        "price-update-up": "price-flash-up-glass 800ms cubic-bezier(0.4, 0, 0.2, 1)",
        "price-update-down": "price-flash-down-glass 800ms cubic-bezier(0.4, 0, 0.2, 1)",

        // Status Animations
        "status-pulse": "status-pulse-pro 2s ease-in-out infinite",
        "status-pulse-pro": "status-pulse-pro 2s ease-in-out infinite",

        // Loading States
        "shimmer": "shimmer-glass 2s infinite",
        "shimmer-glass": "shimmer-glass 2s infinite",
        "loading-shimmer": "shimmer-glass 2s infinite",

        // Data Visualization
        "data-glow": "data-glow 4s ease-in-out infinite",
        "data-point-new": "slide-in-right-glass 400ms cubic-bezier(0.4, 0, 0.2, 1)",

        // Slide Animations
        "slide-in-right": "slide-in-right-glass 400ms cubic-bezier(0.4, 0, 0.2, 1)",
        "slide-in-left": "slide-in-left-glass 400ms cubic-bezier(0.4, 0, 0.2, 1)",
        "slide-in-up": "slide-in-up-glass 400ms cubic-bezier(0.4, 0, 0.2, 1)",

        // Gradient Border Animation
        "gradient-border": "gradient-border 3s ease infinite",
      },

      // === Professional Keyframes ===
      keyframes: {
        "terminal-blink": {
          "0%, 50%": { opacity: "1" },
          "51%, 100%": { opacity: "0.3" },
        },
        "terminal-scan": {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
        "data-flow": {
          "0%": { opacity: "0", transform: "translateX(-10px)" },
          "50%": { opacity: "1" },
          "100%": { opacity: "0", transform: "translateX(10px)" },
        },
        "backgroundPulse": {
          "0%, 100%": {
            opacity: "0.3",
            transform: "scale(1) rotate(0deg)",
          },
          "33%": {
            opacity: "0.5",
            transform: "scale(1.1) rotate(1deg)",
          },
          "66%": {
            opacity: "0.4",
            transform: "scale(0.95) rotate(-0.5deg)",
          },
        },
        "glassSlideIn": {
          "0%": {
            opacity: "0",
            transform: "translateY(30px) scale(0.95)",
          },
          "100%": {
            opacity: "1",
            transform: "translateY(0) scale(1)",
          },
        },
        "shimmer-glass": {
          "0%": {
            "background-position": "-200% center",
          },
          "100%": {
            "background-position": "200% center",
          },
        },
        "gradient-border": {
          "0%": { "background-position": "0% 50%" },
          "50%": { "background-position": "100% 50%" },
          "100%": { "background-position": "0% 50%" },
        },
      },

      // === Professional Screen Sizes ===
      screens: {
        'xs': '475px',
        'sm': '640px',
        'md': '768px',
        'lg': '1024px',
        'xl': '1280px',
        '2xl': '1536px',
        '3xl': '1920px',
        '4xl': '2560px',
        // Device-specific breakpoints
        'mobile': '320px',
        'tablet': '768px',
        'desktop': '1024px',
        'ultrawide': '1920px',
        // Terminal-specific sizes
        'terminal-sm': '640px',
        'terminal-md': '896px',
        'terminal-lg': '1152px',
        'terminal-xl': '1408px',
      },

      // === Professional Z-Index Scale ===
      zIndex: {
        'modal': '1000',
        'dropdown': '900',
        'tooltip': '800',
        'navbar': '700',
        'sidebar': '600',
        'overlay': '500',
        'terminal': '100',
        'chart': '50',
        'content': '10',
        'background': '0',
        'behind': '-1',
      },

      // === Professional Line Heights ===
      lineHeight: {
        'terminal': '1.4',
        'terminal-tight': '1.25',
        'terminal-relaxed': '1.6',
      },

      // === Professional Letter Spacing ===
      letterSpacing: {
        'terminal': '0.025em',
        'terminal-wide': '0.1em',
        'terminal-wider': '0.15em',
      },

      // === Professional Opacity Scale ===
      opacity: {
        '15': '0.15',
        '35': '0.35',
        '65': '0.65',
        '85': '0.85',
      },

      // === Professional Scale ===
      scale: {
        '102': '1.02',
        '105': '1.05',
        '98': '0.98',
      },

      // === Professional Transitions ===
      transitionDuration: {
        '400': '400ms',
        '600': '600ms',
        '800': '800ms',
        '900': '900ms',
      },

      transitionTimingFunction: {
        'terminal': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'smooth': 'cubic-bezier(0.4, 0, 0.2, 1)',
        'bounce-soft': 'cubic-bezier(0.68, -0.55, 0.265, 1.55)',
      },
    },
  },
  plugins: [
    // Add custom utilities for glassmorphism
    function({ addUtilities, theme }: any) {
      const newUtilities = {
        // Glassmorphism Surface Utilities
        '.glass-surface-primary': {
          background: 'var(--bg-glass-primary)',
          backdropFilter: 'var(--blur-lg)',
          '-webkit-backdrop-filter': 'var(--blur-lg)',
          border: 'var(--border-glass)',
          boxShadow: 'var(--shadow-glass-md)',
        },
        '.glass-surface-secondary': {
          background: 'var(--bg-glass-secondary)',
          backdropFilter: 'var(--blur-md)',
          '-webkit-backdrop-filter': 'var(--blur-md)',
          border: 'var(--border-glass)',
          boxShadow: 'var(--shadow-glass-sm)',
        },
        '.glass-surface-elevated': {
          background: 'var(--bg-elevated)',
          backdropFilter: 'var(--blur-xl)',
          '-webkit-backdrop-filter': 'var(--blur-xl)',
          border: 'var(--border-glass-strong)',
          boxShadow: 'var(--shadow-glass-lg)',
        },

        // Professional Card Utilities
        '.glass-card': {
          background: 'var(--gradient-surface)',
          backdropFilter: 'var(--blur-lg)',
          '-webkit-backdrop-filter': 'var(--blur-lg)',
          border: 'var(--border-glass)',
          borderRadius: '1.5rem',
          boxShadow: 'var(--shadow-glass-md)',
          position: 'relative',
          overflow: 'hidden',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        },

        // Professional Text Utilities
        '.text-glow': {
          textShadow: 'var(--text-glow)',
        },
        '.text-accent-glow': {
          textShadow: 'var(--text-accent-glow)',
          color: '#06B6D4',
        },
        '.text-gradient-primary': {
          background: 'var(--gradient-primary)',
          '-webkit-background-clip': 'text',
          '-webkit-text-fill-color': 'transparent',
          backgroundClip: 'text',
        },

        // Terminal Specific Utilities
        '.terminal-glow': {
          boxShadow: '0 0 5px rgba(245, 158, 11, 0.3), 0 0 10px rgba(245, 158, 11, 0.2), 0 0 15px rgba(245, 158, 11, 0.1)',
        },
        '.terminal-text-glow': {
          textShadow: '0 0 5px rgba(245, 158, 11, 0.5)',
        },
        '.terminal-border': {
          border: '1px solid var(--terminal-border)',
        },

        // Interactive States
        '.interactive-element': {
          transition: 'all 150ms cubic-bezier(0.4, 0, 0.2, 1)',
        },

        // Focus States
        '.glass-focus:focus-visible': {
          outline: 'none',
          boxShadow: 'var(--focus-ring), var(--shadow-glass-md)',
          borderColor: 'rgba(6, 182, 212, 0.8)',
        },

        // Safe Area Utilities
        '.safe-area-padding-bottom': {
          paddingBottom: 'env(safe-area-inset-bottom)',
        },
        '.safe-area-inset-top': {
          paddingTop: 'env(safe-area-inset-top)',
        },
        '.safe-area-inset-left': {
          paddingLeft: 'env(safe-area-inset-left)',
        },
        '.safe-area-inset-right': {
          paddingRight: 'env(safe-area-inset-right)',
        },

        // Line Clamp Utilities
        '.line-clamp-1': {
          overflow: 'hidden',
          display: '-webkit-box',
          '-webkit-box-orient': 'vertical',
          '-webkit-line-clamp': '1',
        },
        '.line-clamp-2': {
          overflow: 'hidden',
          display: '-webkit-box',
          '-webkit-box-orient': 'vertical',
          '-webkit-line-clamp': '2',
        },
        '.line-clamp-3': {
          overflow: 'hidden',
          display: '-webkit-box',
          '-webkit-box-orient': 'vertical',
          '-webkit-line-clamp': '3',
        },
      };

      addUtilities(newUtilities, ['responsive', 'hover']);
    },
  ],
};

export default config;