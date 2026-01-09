"use client";

import { LanguageProvider } from "@/contexts/LanguageContext";
import Navbar from "@/components/landing/Navbar";
import Hero from "@/components/landing/Hero";
import Metrics from "@/components/landing/Metrics";
import Features from "@/components/landing/Features";
import HowItWorks from "@/components/landing/HowItWorks";
import Pricing from "@/components/landing/Pricing";
import FAQ from "@/components/landing/FAQ";
import Footer from "@/components/landing/Footer";

export default function LandingPage() {
  return (
    <LanguageProvider>
      {/* Contenedor raíz que neutraliza los backgrounds asimétricos del body */}
      <div className="landing-root text-white antialiased">
        {/* Navigation - Fixed at top */}
        <Navbar />

        {/* Main content wrapper - flex column for proper stacking */}
        <main className="relative w-full flex flex-col">
          {/* Hero Section */}
          <Hero />

          {/* Metrics/Stats */}
          <Metrics />

          {/* Features */}
          <Features />

          {/* How It Works */}
          <HowItWorks />

          {/* Early Access */}
          <Pricing />

          {/* FAQ */}
          <FAQ />

          {/* Footer */}
          <Footer />
        </main>
      </div>
    </LanguageProvider>
  );
}
