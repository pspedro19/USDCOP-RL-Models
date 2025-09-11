/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      // 2025 Fintech Color Palette - Optimized for Dark Mode
      colors: {
        // Enhanced Base Grays (WCAG AA Compliant)
        'fintech-dark': {
          50: '#F8FAFC',   // Off-white text
          100: '#F1F5F9',  // Light gray text
          200: '#E2E8F0',  // Secondary text
          300: '#CBD5E1',  // Tertiary text
          400: '#94A3B8',  // Muted text
          500: '#64748B',  // Neutral
          600: '#475569',  // Dark neutral
          700: '#334155',  // Darker neutral
          800: '#1E293B',  // Surface elevated
          850: '#172033',  // Custom surface
          900: '#0F172A',  // Surface primary
          925: '#0A0E1A',  // Custom darker
          950: '#020617',  // Background
        },
        
        // Attenuated Primary Colors (Reduced Saturation)
        'fintech-cyan': {
          50: '#ECFEFF',
          100: '#CFFAFE', 
          400: '#22D3EE',
          500: '#0D9E75',  // Attenuated from #10B981
          600: '#0891B2',
          700: '#0E7490',
          800: '#155E75',
          900: '#164E63',
        },

        'fintech-purple': {
          400: '#A78BFA',
          500: '#7C5CF6',  // Slightly attenuated from #8B5CF6
          600: '#7C3AED',
        },

        // Market Semantics
        'market-up': '#0D9E75',    // Attenuated green
        'market-down': '#DC2626',  // Attenuated red
        'market-neutral': '#64748B',

        // Glass Variables
        'glass': {
          'bg-primary': 'rgba(15, 20, 27, 0.40)',
          'bg-secondary': 'rgba(30, 41, 59, 0.60)', 
          'bg-elevated': 'rgba(51, 65, 85, 0.75)',
          'border': 'rgba(6, 182, 212, 0.2)',
          'border-strong': 'rgba(6, 182, 212, 0.4)',
        }
      },

      // Custom Glassmorphism Effects
      backdropBlur: {
        'fintech-sm': '4px',
        'fintech-md': '8px', 
        'fintech-lg': '16px',
        'fintech-xl': '24px',
        'fintech-2xl': '40px',
        'fintech-3xl': '64px',
      },

      // Professional Shadows
      boxShadow: {
        'glass-sm': '0 2px 8px rgba(0, 0, 0, 0.1), 0 1px 4px rgba(6, 182, 212, 0.1)',
        'glass-md': '0 4px 16px rgba(0, 0, 0, 0.15), 0 2px 8px rgba(6, 182, 212, 0.15)',
        'glass-lg': '0 8px 32px rgba(0, 0, 0, 0.2), 0 4px 16px rgba(6, 182, 212, 0.2)',
        'glass-xl': '0 16px 64px rgba(0, 0, 0, 0.25), 0 8px 32px rgba(6, 182, 212, 0.25)',
        'glow-cyan': '0 0 20px rgba(6, 182, 212, 0.4)',
        'glow-purple': '0 0 20px rgba(139, 92, 246, 0.4)',
        'glow-mixed': '0 0 30px rgba(6, 182, 212, 0.3), 0 0 60px rgba(139, 92, 246, 0.2)',
        'market-up': '0 0 20px rgba(13, 158, 117, 0.4)',
        'market-down': '0 0 20px rgba(220, 38, 38, 0.4)',
      },

      // Enhanced Typography
      fontFamily: {
        'terminal': ['Monaco', 'Menlo', 'Courier New', 'monospace'],
        'fintech': ['Inter', 'system-ui', 'sans-serif'],
      },

      // Professional Gradients  
      backgroundImage: {
        'fintech-primary': 'linear-gradient(135deg, #0D9E75 0%, #0891B2 50%, #7C5CF6 100%)',
        'fintech-surface': 'linear-gradient(135deg, rgba(15, 20, 27, 0.95) 0%, rgba(30, 41, 59, 0.90) 100%)',
        'fintech-glass': 'linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(124, 92, 246, 0.10) 50%, rgba(13, 158, 117, 0.15) 100%)',
        'fintech-border': 'linear-gradient(135deg, rgba(6, 182, 212, 0.5) 0%, rgba(124, 92, 246, 0.3) 100%)',
      },

      // Enhanced Animations
      animation: {
        'pulse-glow': 'pulse-glow 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in-glass': 'slide-in-glass 0.6s cubic-bezier(0.4, 0, 0.2, 1)',
        'float-glass': 'float-glass 6s ease-in-out infinite',
        'shimmer-glass': 'shimmer-glass 2s infinite',
        'price-flash-up': 'price-flash-up 800ms cubic-bezier(0.4, 0, 0.2, 1)',
        'price-flash-down': 'price-flash-down 800ms cubic-bezier(0.4, 0, 0.2, 1)',
      },

      keyframes: {
        'pulse-glow': {
          '0%, 100%': { 
            opacity: '1',
            transform: 'scale(1)',
            boxShadow: '0 0 10px rgba(6, 182, 212, 0.3)'
          },
          '50%': { 
            opacity: '0.8',
            transform: 'scale(1.05)',
            boxShadow: '0 0 20px rgba(6, 182, 212, 0.5)'
          },
        },
        'slide-in-glass': {
          '0%': {
            opacity: '0',
            transform: 'translateY(30px) scale(0.95)',
          },
          '100%': {
            opacity: '1', 
            transform: 'translateY(0) scale(1)',
          },
        },
        'float-glass': {
          '0%, 100%': {
            transform: 'translateY(0px) scale(1)',
          },
          '50%': {
            transform: 'translateY(-4px) scale(1.01)',
          },
        },
        'shimmer-glass': {
          '0%': {
            backgroundPosition: '-200% center',
          },
          '100%': {
            backgroundPosition: '200% center',
          },
        },
        'price-flash-up': {
          '0%': { 
            backgroundColor: 'rgba(13, 158, 117, 0.2)',
            transform: 'scale(1.02)',
          },
          '50%': {
            backgroundColor: 'rgba(13, 158, 117, 0.3)',
            transform: 'scale(1.05)',
          },
          '100%': { 
            backgroundColor: 'rgba(13, 158, 117, 0.05)',
            transform: 'scale(1)',
          },
        },
        'price-flash-down': {
          '0%': { 
            backgroundColor: 'rgba(220, 38, 38, 0.2)',
            transform: 'scale(1.02)',
          },
          '50%': {
            backgroundColor: 'rgba(220, 38, 38, 0.3)',
            transform: 'scale(1.05)',
          },
          '100%': { 
            backgroundColor: 'rgba(220, 38, 38, 0.05)',
            transform: 'scale(1)',
          },
        },
      },

      // Responsive Breakpoints
      screens: {
        'xs': '475px',
        'sm': '640px',
        'md': '768px', 
        'lg': '1024px',
        'xl': '1280px',
        '2xl': '1536px',
        '3xl': '1920px',
      },
    },
  },
  plugins: [
    // Custom plugin for glass effects
    function({ addUtilities, theme }) {
      const newUtilities = {
        '.glass-surface': {
          background: 'rgba(15, 20, 27, 0.40)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(6, 182, 212, 0.2)',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15), 0 2px 8px rgba(6, 182, 212, 0.15)',
        },
        '.glass-card': {
          background: 'linear-gradient(135deg, rgba(15, 20, 27, 0.95) 0%, rgba(30, 41, 59, 0.90) 100%)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(6, 182, 212, 0.2)',
          borderRadius: '24px',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.15), 0 2px 8px rgba(6, 182, 212, 0.15)',
          position: 'relative',
          overflow: 'hidden',
          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        '.glass-button': {
          background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(124, 92, 246, 0.10) 50%, rgba(13, 158, 117, 0.15) 100%)',
          backdropFilter: 'blur(8px)',
          WebkitBackdropFilter: 'blur(8px)',
          border: '1px solid rgba(6, 182, 212, 0.2)',
          borderRadius: '16px',
          transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        },
        '.text-glow': {
          textShadow: '0 0 8px rgba(248, 250, 252, 0.3)',
        },
        '.text-accent-glow': {
          textShadow: '0 0 12px rgba(6, 182, 212, 0.6)',
          color: '#06B6D4',
        },
      }
      addUtilities(newUtilities)
    }
  ],
}