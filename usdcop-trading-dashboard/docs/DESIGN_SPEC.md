# World-Class Fintech/Algorithmic Trading Landing Page Design Specification

> Research-based design system for a premium dark-theme trading platform landing page

---

## Table of Contents

1. [Color System](#1-color-system)
2. [Typography Hierarchy](#2-typography-hierarchy)
3. [Section Layout & Order](#3-section-layout--order)
4. [Trust Signals & Social Proof](#4-trust-signals--social-proof)
5. [CTA Button Design](#5-cta-button-design)
6. [Animation Patterns](#6-animation-patterns)
7. [Tailwind CSS Implementation](#7-tailwind-css-implementation)
8. [Mobile-First Breakpoint Strategy](#8-mobile-first-breakpoint-strategy)
9. [Component Specifications](#9-component-specifications)

---

## 1. Color System

### Primary Dark Theme Palette

Based on analysis of top fintech platforms (Stripe, Binance, Robinhood, Mercury), the following palette establishes trust, professionalism, and modern appeal:

| Role | Color Name | Hex Code | Tailwind Class | Usage |
|------|------------|----------|----------------|-------|
| **Background Primary** | Obsidian | `#0A0A0F` | `bg-[#0A0A0F]` | Main page background |
| **Background Secondary** | Deep Slate | `#0F0F15` | `bg-[#0F0F15]` | Card backgrounds, sections |
| **Background Elevated** | Charcoal | `#16161D` | `bg-[#16161D]` | Elevated cards, modals |
| **Background Accent** | Dark Navy | `#0D1B2A` | `bg-[#0D1B2A]` | Feature sections |
| **Border Primary** | Smoke | `#1E1E28` | `border-[#1E1E28]` | Subtle borders |
| **Border Accent** | Steel | `#2A2A3A` | `border-[#2A2A3A]` | Interactive borders |

### Accent Colors (Trust & Action)

| Role | Color Name | Hex Code | Tailwind Class | Usage |
|------|------------|----------|----------------|-------|
| **Primary Accent** | Electric Blue | `#0EA5E9` | `text-sky-500` | CTAs, links, highlights |
| **Primary Hover** | Bright Blue | `#38BDF8` | `text-sky-400` | Hover states |
| **Success** | Emerald | `#10B981` | `text-emerald-500` | Positive metrics, confirmations |
| **Success Glow** | Light Emerald | `#34D399` | `text-emerald-400` | Profit indicators |
| **Warning** | Amber | `#F59E0B` | `text-amber-500` | Alerts, pending states |
| **Danger** | Coral Red | `#EF4444` | `text-red-500` | Loss indicators, errors |
| **Premium** | Violet | `#8B5CF6` | `text-violet-500` | Premium features |
| **Gold** | Champagne | `#D4AF37` | `text-[#D4AF37]` | Awards, premium badges |

### Text Colors

| Role | Hex Code | Tailwind Class | Usage |
|------|----------|----------------|-------|
| **Heading Primary** | `#FFFFFF` | `text-white` | Main headings |
| **Heading Secondary** | `#F8FAFC` | `text-slate-50` | Subheadings |
| **Body Primary** | `#E2E8F0` | `text-slate-200` | Primary body text |
| **Body Secondary** | `#94A3B8` | `text-slate-400` | Secondary descriptions |
| **Muted** | `#64748B` | `text-slate-500` | Captions, labels |
| **Disabled** | `#475569` | `text-slate-600` | Disabled states |

### Gradient Presets

```css
/* Hero Gradient - Sophisticated depth */
.gradient-hero {
  background: linear-gradient(135deg, #0A0A0F 0%, #0D1B2A 50%, #0A0A0F 100%);
}

/* Premium Accent Gradient */
.gradient-accent {
  background: linear-gradient(90deg, #0EA5E9 0%, #8B5CF6 100%);
}

/* Success Gradient (for profit displays) */
.gradient-success {
  background: linear-gradient(90deg, #10B981 0%, #34D399 100%);
}

/* Glass Effect */
.glass {
  background: rgba(15, 15, 21, 0.8);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255, 255, 255, 0.05);
}
```

### Tailwind Gradient Classes

```html
<!-- Hero Background -->
<div class="bg-gradient-to-br from-[#0A0A0F] via-[#0D1B2A] to-[#0A0A0F]">

<!-- Accent Text Gradient -->
<span class="bg-gradient-to-r from-sky-500 to-violet-500 bg-clip-text text-transparent">

<!-- CTA Button Gradient -->
<button class="bg-gradient-to-r from-sky-500 to-sky-600 hover:from-sky-400 hover:to-sky-500">

<!-- Glow Effect -->
<div class="shadow-[0_0_50px_rgba(14,165,233,0.15)]">
```

---

## 2. Typography Hierarchy

### Font Stack

```css
/* Primary Font - Modern Sans-Serif */
font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;

/* Monospace - For numbers, code, metrics */
font-family: 'JetBrains Mono', 'SF Mono', 'Fira Code', monospace;
```

### Mobile-First Type Scale

| Element | Mobile (base) | sm (640px) | md (768px) | lg (1024px) | xl (1280px) | Tailwind Classes |
|---------|---------------|------------|------------|-------------|-------------|------------------|
| **Hero H1** | 36px / 2.25rem | 42px | 48px | 56px | 64px | `text-4xl sm:text-5xl md:text-5xl lg:text-6xl xl:text-7xl` |
| **Section H2** | 28px / 1.75rem | 32px | 36px | 40px | 48px | `text-3xl sm:text-4xl lg:text-5xl` |
| **Card H3** | 20px / 1.25rem | 22px | 24px | 26px | 28px | `text-xl sm:text-2xl lg:text-3xl` |
| **Subtitle** | 18px / 1.125rem | 18px | 20px | 20px | 22px | `text-lg md:text-xl` |
| **Body Large** | 16px / 1rem | 17px | 18px | 18px | 18px | `text-base md:text-lg` |
| **Body** | 15px / 0.938rem | 16px | 16px | 16px | 16px | `text-[15px] sm:text-base` |
| **Caption** | 13px / 0.813rem | 14px | 14px | 14px | 14px | `text-sm` |
| **Micro** | 11px / 0.688rem | 12px | 12px | 12px | 12px | `text-xs` |

### Font Weights

```html
<!-- Hero Headlines -->
<h1 class="font-bold tracking-tight">

<!-- Section Headlines -->
<h2 class="font-semibold tracking-tight">

<!-- Subheadings -->
<h3 class="font-medium">

<!-- Body Text -->
<p class="font-normal">

<!-- Emphasis -->
<span class="font-medium">

<!-- Numbers/Metrics -->
<span class="font-mono font-semibold tabular-nums">
```

### Line Heights

| Type | Line Height | Tailwind Class |
|------|-------------|----------------|
| Headlines | 1.1 - 1.2 | `leading-tight` or `leading-none` |
| Subheadings | 1.3 | `leading-snug` |
| Body Text | 1.6 - 1.7 | `leading-relaxed` |
| UI Elements | 1.4 | `leading-normal` |

### Letter Spacing

```html
<!-- Tight for large headlines -->
<h1 class="tracking-tight">

<!-- Normal for body -->
<p class="tracking-normal">

<!-- Wide for labels/badges -->
<span class="tracking-wide uppercase text-xs">
```

---

## 3. Section Layout & Order

### Optimal Section Sequence (Research-Based)

Based on analysis of high-converting fintech landing pages (Robinhood, Stripe, Mercury, Plaid):

```
1. Navigation Bar (sticky)
2. Hero Section
3. Social Proof Bar (logos/metrics)
4. Features/Benefits Grid
5. Live Terminal/Demo
6. How It Works
7. Performance Metrics
8. Testimonials/Case Studies
9. Pricing (if applicable)
10. FAQ
11. Final CTA
12. Footer
```

### Section Spacing

```html
<!-- Section Wrapper -->
<section class="py-16 sm:py-20 md:py-24 lg:py-32">

<!-- Content Container -->
<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
```

### Layout Patterns

#### Hero Section
```html
<section class="relative min-h-screen flex items-center justify-center overflow-hidden">
  <!-- Background gradient -->
  <div class="absolute inset-0 bg-gradient-to-br from-[#0A0A0F] via-[#0D1B2A] to-[#0A0A0F]" />

  <!-- Grid/mesh overlay for depth -->
  <div class="absolute inset-0 bg-[url('/grid.svg')] opacity-20" />

  <!-- Glow orbs -->
  <div class="absolute top-1/4 left-1/4 w-96 h-96 bg-sky-500/10 rounded-full blur-3xl" />

  <!-- Content -->
  <div class="relative z-10 text-center px-4 sm:px-6 lg:px-8 max-w-5xl">
    <!-- Badge -->
    <div class="inline-flex items-center px-4 py-2 rounded-full bg-sky-500/10 border border-sky-500/20 mb-6">
      <span class="text-sky-400 text-sm font-medium">New: AI-Powered Signals</span>
    </div>

    <!-- Headline -->
    <h1 class="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold text-white tracking-tight mb-6">
      Algorithmic Trading
      <span class="bg-gradient-to-r from-sky-400 to-violet-500 bg-clip-text text-transparent">
        Reimagined
      </span>
    </h1>

    <!-- Subheadline -->
    <p class="text-lg sm:text-xl text-slate-400 max-w-2xl mx-auto mb-8">
      Professional-grade RL models for USD/COP trading.
      Real-time signals, backtested strategies, zero complexity.
    </p>

    <!-- CTA Group -->
    <div class="flex flex-col sm:flex-row items-center justify-center gap-4">
      <button class="w-full sm:w-auto px-8 py-4 bg-gradient-to-r from-sky-500 to-sky-600
                     text-white font-semibold rounded-xl hover:from-sky-400 hover:to-sky-500
                     transition-all duration-200 shadow-lg shadow-sky-500/25">
        Start Trading Free
      </button>
      <button class="w-full sm:w-auto px-8 py-4 bg-white/5 border border-white/10
                     text-white font-medium rounded-xl hover:bg-white/10
                     transition-all duration-200">
        View Live Demo
      </button>
    </div>
  </div>
</section>
```

#### Features Grid
```html
<section class="py-24 bg-[#0F0F15]">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <!-- Section Header -->
    <div class="text-center max-w-3xl mx-auto mb-16">
      <h2 class="text-3xl sm:text-4xl lg:text-5xl font-semibold text-white mb-4">
        Built for Serious Traders
      </h2>
      <p class="text-lg text-slate-400">
        Enterprise-grade infrastructure with institutional reliability
      </p>
    </div>

    <!-- Grid -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
      <!-- Feature Card -->
      <div class="group relative p-6 lg:p-8 bg-[#16161D] rounded-2xl border border-[#1E1E28]
                  hover:border-sky-500/30 transition-all duration-300">
        <!-- Icon -->
        <div class="w-12 h-12 rounded-xl bg-sky-500/10 flex items-center justify-center mb-4
                    group-hover:bg-sky-500/20 transition-colors">
          <svg class="w-6 h-6 text-sky-400">...</svg>
        </div>

        <!-- Content -->
        <h3 class="text-xl font-semibold text-white mb-2">Real-Time Signals</h3>
        <p class="text-slate-400 leading-relaxed">
          Sub-second latency trading signals powered by reinforcement learning models.
        </p>
      </div>
    </div>
  </div>
</section>
```

---

## 4. Trust Signals & Social Proof

### Essential Trust Elements

#### Logo Bar (After Hero)
```html
<section class="py-12 border-y border-[#1E1E28]">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <p class="text-center text-slate-500 text-sm mb-8">
      Trusted by leading financial institutions
    </p>
    <div class="flex flex-wrap items-center justify-center gap-8 lg:gap-16 opacity-60 grayscale">
      <!-- Partner logos -->
      <img src="/logos/partner1.svg" alt="Partner" class="h-8" />
      <!-- ... more logos -->
    </div>
  </div>
</section>
```

#### Metrics Bar
```html
<div class="grid grid-cols-2 md:grid-cols-4 gap-8 py-12">
  <!-- Metric -->
  <div class="text-center">
    <div class="text-3xl sm:text-4xl font-bold text-white font-mono">
      $2.4M+
    </div>
    <div class="text-slate-400 text-sm mt-1">Trading Volume</div>
  </div>

  <div class="text-center">
    <div class="text-3xl sm:text-4xl font-bold text-emerald-400 font-mono">
      +47.3%
    </div>
    <div class="text-slate-400 text-sm mt-1">Avg. Annual Return</div>
  </div>

  <div class="text-center">
    <div class="text-3xl sm:text-4xl font-bold text-white font-mono">
      99.9%
    </div>
    <div class="text-slate-400 text-sm mt-1">Uptime SLA</div>
  </div>

  <div class="text-center">
    <div class="text-3xl sm:text-4xl font-bold text-white font-mono">
      &lt;50ms
    </div>
    <div class="text-slate-400 text-sm mt-1">Signal Latency</div>
  </div>
</div>
```

#### Security Badges
```html
<div class="flex flex-wrap items-center justify-center gap-4">
  <!-- Security Badge -->
  <div class="flex items-center gap-2 px-4 py-2 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
    <svg class="w-5 h-5 text-emerald-400"><!-- Lock icon --></svg>
    <span class="text-emerald-400 text-sm font-medium">256-bit Encryption</span>
  </div>

  <!-- Compliance Badge -->
  <div class="flex items-center gap-2 px-4 py-2 bg-sky-500/10 rounded-lg border border-sky-500/20">
    <svg class="w-5 h-5 text-sky-400"><!-- Shield icon --></svg>
    <span class="text-sky-400 text-sm font-medium">SOC 2 Compliant</span>
  </div>

  <!-- Uptime Badge -->
  <div class="flex items-center gap-2 px-4 py-2 bg-violet-500/10 rounded-lg border border-violet-500/20">
    <svg class="w-5 h-5 text-violet-400"><!-- Server icon --></svg>
    <span class="text-violet-400 text-sm font-medium">99.99% Uptime</span>
  </div>
</div>
```

#### Testimonial Card
```html
<div class="bg-[#16161D] rounded-2xl p-8 border border-[#1E1E28]">
  <!-- Stars -->
  <div class="flex gap-1 mb-4">
    <svg class="w-5 h-5 text-amber-400 fill-current"><!-- Star --></svg>
    <!-- ... 5 stars -->
  </div>

  <!-- Quote -->
  <blockquote class="text-lg text-slate-200 mb-6 leading-relaxed">
    "The RL models have transformed our USD/COP trading strategy.
    We've seen consistent alpha generation with exceptional risk-adjusted returns."
  </blockquote>

  <!-- Author -->
  <div class="flex items-center gap-4">
    <img src="/avatars/trader.jpg" alt="" class="w-12 h-12 rounded-full" />
    <div>
      <div class="font-medium text-white">Carlos Rodriguez</div>
      <div class="text-sm text-slate-400">Head of Trading, LatAm Capital</div>
    </div>
  </div>
</div>
```

---

## 5. CTA Button Design

### Primary CTA (High Emphasis)

```html
<button class="
  relative overflow-hidden
  px-8 py-4
  bg-gradient-to-r from-sky-500 to-sky-600
  text-white font-semibold text-base
  rounded-xl
  shadow-lg shadow-sky-500/25
  hover:from-sky-400 hover:to-sky-500
  hover:shadow-xl hover:shadow-sky-500/30
  hover:-translate-y-0.5
  active:translate-y-0
  transition-all duration-200
  focus:outline-none focus:ring-2 focus:ring-sky-500 focus:ring-offset-2 focus:ring-offset-[#0A0A0F]
">
  Start Trading Free
</button>
```

### Secondary CTA (Medium Emphasis)

```html
<button class="
  px-8 py-4
  bg-white/5
  border border-white/10
  text-white font-medium text-base
  rounded-xl
  hover:bg-white/10
  hover:border-white/20
  transition-all duration-200
  focus:outline-none focus:ring-2 focus:ring-white/20
">
  View Live Demo
</button>
```

### Ghost CTA (Low Emphasis)

```html
<button class="
  px-6 py-3
  text-sky-400 font-medium text-sm
  hover:text-sky-300
  hover:bg-sky-400/5
  rounded-lg
  transition-all duration-200
">
  Learn more
  <svg class="inline-block w-4 h-4 ml-1"><!-- Arrow --></svg>
</button>
```

### CTA Button Sizes

| Size | Padding | Font Size | Tailwind Classes |
|------|---------|-----------|------------------|
| Small | 12px 20px | 14px | `px-5 py-3 text-sm` |
| Medium | 16px 28px | 15px | `px-7 py-4 text-[15px]` |
| Large | 18px 32px | 16px | `px-8 py-[18px] text-base` |
| XLarge | 20px 40px | 18px | `px-10 py-5 text-lg` |

### Mobile CTA Pattern

```html
<!-- Full-width on mobile, auto on larger screens -->
<button class="w-full sm:w-auto px-8 py-4 ...">
  Get Started
</button>
```

### CTA Copy Best Practices

| Avoid | Use Instead |
|-------|-------------|
| Submit | Start Trading |
| Click Here | View Live Demo |
| Learn More | See How It Works |
| Download | Get Your Free Report |
| Sign Up | Create Free Account |

**First-person CTAs convert better:**
- "Start My Trial" > "Start Your Trial"
- "Get My Signals" > "Get Signals"

---

## 6. Animation Patterns

### Framer Motion Configurations

#### Fade Up (Entry Animation)
```tsx
const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }
};
```

#### Stagger Children
```tsx
const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.2
    }
  }
};

const staggerItem = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 }
};
```

#### Scale on Hover (Cards)
```tsx
<motion.div
  whileHover={{
    scale: 1.02,
    transition: { duration: 0.2 }
  }}
  whileTap={{ scale: 0.98 }}
>
```

#### Scroll-Triggered Reveal
```tsx
const scrollReveal = {
  initial: { opacity: 0, y: 50 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-100px" },
  transition: { duration: 0.6, ease: "easeOut" }
};
```

### CSS Keyframes for Subtle Effects

```css
/* Gentle pulse for live indicators */
@keyframes pulse-soft {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

/* Gradient shimmer */
@keyframes shimmer {
  0% { background-position: -200% center; }
  100% { background-position: 200% center; }
}

/* Floating animation */
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-10px); }
}
```

### Tailwind Animation Classes

```html
<!-- Pulse for live status -->
<span class="animate-pulse inline-block w-2 h-2 bg-emerald-400 rounded-full" />

<!-- Subtle bounce on scroll -->
<div class="animate-bounce">
  <svg><!-- Down arrow --></svg>
</div>

<!-- Custom animation -->
<div class="animate-[float_6s_ease-in-out_infinite]">
```

### Performance Guidelines

1. **Only animate `transform` and `opacity`** - GPU accelerated
2. **Use `will-change` sparingly** - only for elements that will animate
3. **Prefer CSS for simple animations** - less JavaScript overhead
4. **Use `reduced-motion` media query** for accessibility

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

---

## 7. Tailwind CSS Implementation

### tailwind.config.js Extensions

```javascript
module.exports = {
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Background colors
        'bg-primary': '#0A0A0F',
        'bg-secondary': '#0F0F15',
        'bg-elevated': '#16161D',
        'bg-accent': '#0D1B2A',

        // Border colors
        'border-primary': '#1E1E28',
        'border-accent': '#2A2A3A',

        // Brand accent
        'brand': {
          50: '#f0f9ff',
          100: '#e0f2fe',
          200: '#bae6fd',
          300: '#7dd3fc',
          400: '#38bdf8',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
          800: '#075985',
          900: '#0c4a6e',
        },
      },

      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'SF Mono', 'monospace'],
      },

      fontSize: {
        '2xs': ['0.625rem', { lineHeight: '0.75rem' }],
      },

      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'fade-up': 'fadeUp 0.5s ease-out',
        'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'float': 'float 6s ease-in-out infinite',
      },

      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(20px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseSoft: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% center' },
          '100%': { backgroundPosition: '200% center' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
      },

      boxShadow: {
        'glow-sm': '0 0 20px rgba(14, 165, 233, 0.15)',
        'glow': '0 0 40px rgba(14, 165, 233, 0.2)',
        'glow-lg': '0 0 60px rgba(14, 165, 233, 0.25)',
        'inner-glow': 'inset 0 0 20px rgba(14, 165, 233, 0.1)',
      },

      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'grid-pattern': 'url("/grid.svg")',
      },
    },
  },
  plugins: [],
};
```

### Global CSS Variables

```css
:root {
  /* Colors */
  --color-bg-primary: #0A0A0F;
  --color-bg-secondary: #0F0F15;
  --color-bg-elevated: #16161D;
  --color-accent: #0EA5E9;
  --color-success: #10B981;
  --color-danger: #EF4444;

  /* Spacing */
  --section-padding-mobile: 4rem;
  --section-padding-desktop: 8rem;

  /* Transitions */
  --transition-fast: 150ms ease;
  --transition-normal: 200ms ease;
  --transition-slow: 300ms ease;
}
```

---

## 8. Mobile-First Breakpoint Strategy

### Tailwind Breakpoints

| Breakpoint | Min Width | Target Devices |
|------------|-----------|----------------|
| base | 0px | Mobile phones (portrait) |
| sm | 640px | Mobile phones (landscape), small tablets |
| md | 768px | Tablets |
| lg | 1024px | Laptops, small desktops |
| xl | 1280px | Desktops |
| 2xl | 1536px | Large desktops, ultra-wide |

### Responsive Patterns

#### Container
```html
<div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
```

#### Grid Systems
```html
<!-- 1 → 2 → 3 columns -->
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">

<!-- 1 → 2 → 4 columns -->
<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6">

<!-- 2-column feature layout -->
<div class="grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-16 items-center">
```

#### Typography Scaling
```html
<h1 class="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl">
<h2 class="text-2xl sm:text-3xl lg:text-4xl xl:text-5xl">
<p class="text-base md:text-lg lg:text-xl">
```

#### Spacing Scaling
```html
<!-- Section padding -->
<section class="py-16 sm:py-20 md:py-24 lg:py-32">

<!-- Content gaps -->
<div class="space-y-6 md:space-y-8 lg:space-y-12">

<!-- Component margins -->
<div class="mt-8 sm:mt-12 lg:mt-16">
```

#### Show/Hide Elements
```html
<!-- Hide on mobile, show on desktop -->
<div class="hidden lg:block">

<!-- Show on mobile, hide on desktop -->
<div class="lg:hidden">

<!-- Different layouts per breakpoint -->
<div class="flex flex-col sm:flex-row">
```

---

## 9. Component Specifications

### Navbar (Sticky)

```html
<nav class="fixed top-0 left-0 right-0 z-50 bg-[#0A0A0F]/80 backdrop-blur-lg border-b border-[#1E1E28]">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <div class="flex items-center justify-between h-16 lg:h-20">
      <!-- Logo -->
      <a href="/" class="flex items-center gap-2">
        <img src="/logo.svg" alt="Logo" class="h-8 w-8" />
        <span class="font-semibold text-white text-lg">TradingBot</span>
      </a>

      <!-- Desktop Nav -->
      <div class="hidden lg:flex items-center gap-8">
        <a href="#features" class="text-slate-400 hover:text-white transition-colors">Features</a>
        <a href="#pricing" class="text-slate-400 hover:text-white transition-colors">Pricing</a>
        <a href="#docs" class="text-slate-400 hover:text-white transition-colors">Documentation</a>
      </div>

      <!-- CTA -->
      <div class="hidden sm:flex items-center gap-4">
        <a href="/login" class="text-slate-400 hover:text-white transition-colors">Log in</a>
        <button class="px-5 py-2.5 bg-sky-500 text-white font-medium rounded-lg hover:bg-sky-400 transition-colors">
          Get Started
        </button>
      </div>

      <!-- Mobile menu button -->
      <button class="lg:hidden p-2">
        <svg class="w-6 h-6 text-white"><!-- Hamburger --></svg>
      </button>
    </div>
  </div>
</nav>
```

### Feature Card

```html
<div class="group relative p-6 lg:p-8 bg-[#16161D] rounded-2xl border border-[#1E1E28]
            hover:border-sky-500/30 transition-all duration-300
            hover:shadow-lg hover:shadow-sky-500/5">
  <!-- Gradient overlay on hover -->
  <div class="absolute inset-0 rounded-2xl bg-gradient-to-br from-sky-500/5 to-transparent
              opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

  <!-- Content -->
  <div class="relative">
    <!-- Icon container -->
    <div class="w-12 h-12 rounded-xl bg-gradient-to-br from-sky-500/20 to-sky-500/5
                flex items-center justify-center mb-5
                group-hover:from-sky-500/30 group-hover:to-sky-500/10 transition-colors">
      <svg class="w-6 h-6 text-sky-400"><!-- Icon --></svg>
    </div>

    <!-- Title -->
    <h3 class="text-xl font-semibold text-white mb-3">Feature Title</h3>

    <!-- Description -->
    <p class="text-slate-400 leading-relaxed">
      Detailed description of the feature and its benefits for the user.
    </p>

    <!-- Optional link -->
    <a href="#" class="inline-flex items-center gap-1 mt-4 text-sky-400 text-sm font-medium
                       hover:text-sky-300 transition-colors">
      Learn more
      <svg class="w-4 h-4"><!-- Arrow --></svg>
    </a>
  </div>
</div>
```

### Metric Display

```html
<div class="text-center p-6">
  <!-- Value -->
  <div class="text-4xl sm:text-5xl font-bold font-mono tracking-tight">
    <span class="text-emerald-400">+47.3</span>
    <span class="text-emerald-400/60 text-2xl">%</span>
  </div>

  <!-- Label -->
  <div class="text-slate-500 text-sm mt-2">Annual Return</div>

  <!-- Trend indicator (optional) -->
  <div class="flex items-center justify-center gap-1 mt-2 text-emerald-400 text-xs">
    <svg class="w-3 h-3"><!-- Up arrow --></svg>
    <span>+12% from last month</span>
  </div>
</div>
```

### Live Status Indicator

```html
<div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
  <span class="relative flex h-2 w-2">
    <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
    <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-400"></span>
  </span>
  <span class="text-emerald-400 text-sm font-medium">Live</span>
</div>
```

---

## Quick Reference: Complete Page Structure

```html
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>USDCOP Trading Platform</title>
</head>
<body class="bg-[#0A0A0F] text-white font-sans antialiased">

  <!-- 1. Navbar (sticky) -->
  <nav class="fixed top-0 z-50 w-full">...</nav>

  <!-- 2. Hero Section -->
  <section class="min-h-screen pt-20">...</section>

  <!-- 3. Social Proof / Logo Bar -->
  <section class="py-12 border-y border-[#1E1E28]">...</section>

  <!-- 4. Features Grid -->
  <section class="py-24 bg-[#0F0F15]">...</section>

  <!-- 5. Live Terminal Demo -->
  <section class="py-24">...</section>

  <!-- 6. How It Works -->
  <section class="py-24 bg-[#0F0F15]">...</section>

  <!-- 7. Performance Metrics -->
  <section class="py-24">...</section>

  <!-- 8. Testimonials -->
  <section class="py-24 bg-[#0F0F15]">...</section>

  <!-- 9. Pricing -->
  <section class="py-24">...</section>

  <!-- 10. FAQ -->
  <section class="py-24 bg-[#0F0F15]">...</section>

  <!-- 11. Final CTA -->
  <section class="py-24 bg-gradient-to-b from-[#0A0A0F] to-[#0D1B2A]">...</section>

  <!-- 12. Footer -->
  <footer class="py-12 border-t border-[#1E1E28]">...</footer>

</body>
</html>
```

---

## Sources & References

### Fintech Design Best Practices
- [Fintech design guide with patterns that build trust](https://www.eleken.co/blog-posts/modern-fintech-design-guide)
- [24 Best Fintech Website Design Examples in 2025](https://www.webstacks.com/blog/fintech-websites)
- [25 Best Fintech Website Designs in 2025](https://www.ballisticdesignstudio.com/post/fintech-website-designs)
- [Fintech Landing Pages: 59 Examples](https://www.lapa.ninja/category/fintech/)

### Color Palettes
- [Dark mode fintech app cryptocurrency colors palette](https://colorswall.com/palette/135062)
- [Fintech dark mode Revolut colors palette](https://colorswall.com/palette/6595)
- [21+ Fintech Platform Color Palettes](https://produkto.io/color-palettes/fintech-platform)

### Typography
- [Mobile Typography: Font Usage Tips and Best Practices](https://www.toptal.com/designers/typography/typography-for-mobile-apps)
- [Font Size Guidelines for Responsive Websites](https://www.learnui.design/blog/mobile-desktop-website-font-size-guidelines.html)
- [Financial App Design: UX Strategies](https://www.netguru.com/blog/financial-app-design)

### CTA & Conversion
- [15 call to action examples for 2025](https://unbounce.com/conversion-rate-optimization/call-to-action-examples/)
- [10 CTA Button Best Practices for High-Converting Landing Pages](https://bitly.com/blog/cta-button-best-practices-for-landing-pages/)
- [CTA Button UX Design Best Practices](https://www.designstudiouiux.com/blog/cta-button-design-best-practices/)

### Landing Page Structure
- [Breaking down the "Perfect" SaaS Landing page](https://www.cortes.design/post/saas-landing-page-breakdown-example)
- [51 High-Converting SaaS Landing Pages](https://www.klientboost.com/landing-pages/saas-landing-page/)
- [How Robinhood Turned a Landing Page into a 1M Person Waitlist](https://thegrowthplaybook.substack.com/p/how-robinhood-turned-a-landing-page)

### Trust Signals
- [Fintech UX design: patterns that build trust and credibility](https://phenomenonstudio.com/article/fintech-ux-design-patterns-that-build-trust-and-credibility/)
- [Trust Signals in Fintech: Security, Transparency, and Compliance](https://eseospace.com/blog/trust-signals-in-fintech-security/)
- [5 Trust Signals That Instantly Boost Conversion Rates](https://www.crazyegg.com/blog/trust-signals/)

### Animation
- [Interactive UI Animations with GSAP & Framer Motion](https://medium.com/@toukir.ahamed.pigeon/interactive-ui-animations-with-gsap-framer-motion-f2765ae8a051)
- [CSS / JS Animation Trends 2025](https://webpeak.org/blog/css-js-animation-trends/)
- [Motion.dev - JavaScript & React animation library](https://motion.dev/)

### Tailwind CSS
- [Tailwind CSS Dark Mode](https://tailwindcss.com/docs/dark-mode)
- [Gradients for Tailwind CSS | Hypercolor](https://hypercolor.dev/)
- [Tailscan Gradients](https://tailscan.com/gradients)

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Created for: USDCOP Trading Dashboard*
