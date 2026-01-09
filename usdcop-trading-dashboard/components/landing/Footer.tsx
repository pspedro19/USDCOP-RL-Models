"use client";

import Link from "next/link";
import { useLanguage } from "@/contexts/LanguageContext";

export default function Footer() {
  const { t } = useLanguage();

  return (
    <footer className="w-full border-t border-slate-800/50 bg-[#02040a] py-12 sm:py-16 lg:py-20 flex flex-col items-center">
      <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Logo and Tagline */}
        <div className="mb-10 text-center">
          <div className="mb-4 inline-flex items-center gap-2">
            <span className="text-xl font-bold tracking-tight text-white sm:text-2xl">
              USDCOP
            </span>
            <span className="rounded bg-gradient-to-r from-emerald-500 to-teal-500 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-white">
              AI
            </span>
          </div>
          <p className="mx-auto max-w-md text-sm text-slate-400">
            {t.footer.tagline}
          </p>
        </div>

        {/* Risk Warning */}
        <div className="mb-10 rounded-lg border border-slate-800/50 bg-slate-900/30 px-5 py-4">
          <p className="text-center text-xs leading-relaxed text-slate-500">
            {t.footer.risk}
          </p>
        </div>

        {/* Links and Copyright */}
        <div className="flex flex-col items-center gap-4 sm:flex-row sm:justify-between">
          {/* Links - Stack on mobile, horizontal on desktop */}
          <nav className="flex flex-col items-center gap-3 text-center sm:flex-row sm:gap-6">
            <Link
              href="/privacy"
              className="text-sm text-slate-400 transition-colors hover:text-white"
            >
              {t.footer.privacy}
            </Link>
            <Link
              href="/terms"
              className="text-sm text-slate-400 transition-colors hover:text-white"
            >
              {t.footer.terms}
            </Link>
          </nav>

          {/* Copyright */}
          <p className="text-xs text-slate-500">
            {t.footer.rights}
          </p>
        </div>
      </div>
    </footer>
  );
}
