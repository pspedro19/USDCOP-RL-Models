"use client";

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, Globe, ChevronDown } from 'lucide-react';
import { useLanguage } from '@/contexts/LanguageContext';
import { Language } from '@/lib/translations';

interface NavLinkProps {
  href: string;
  children: React.ReactNode;
  onClick?: () => void;
}

const NavLink: React.FC<NavLinkProps> = ({ href, children, onClick }) => (
  <a
    href={href}
    onClick={onClick}
    className="text-gray-300 hover:text-white transition-colors duration-200 text-sm font-medium"
  >
    {children}
  </a>
);

const MobileNavLink: React.FC<NavLinkProps> = ({ href, children, onClick }) => (
  <motion.a
    href={href}
    onClick={onClick}
    className="block px-4 py-3 text-gray-300 hover:text-white hover:bg-white/5 transition-colors duration-200 text-base font-medium rounded-lg"
    whileHover={{ x: 4 }}
    whileTap={{ scale: 0.98 }}
  >
    {children}
  </motion.a>
);

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [isLangOpen, setIsLangOpen] = useState(false);
  const { language, setLanguage, t } = useLanguage();

  // Handle scroll effect
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu on resize
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setIsOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Prevent body scroll when mobile menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [isOpen]);

  const toggleMenu = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  const handleLanguageChange = (lang: Language) => {
    setLanguage(lang);
    setIsLangOpen(false);
  };

  const navLinks = [
    { href: '#features', label: t.nav.features },
    { href: '#how-it-works', label: t.nav.howItWorks },
    { href: '#pricing', label: t.nav.pricing },
    { href: '#faq', label: t.nav.faq },
  ];

  const menuVariants = {
    closed: {
      opacity: 0,
      height: 0,
      transition: {
        duration: 0.3,
        ease: [0.4, 0, 0.2, 1],
        when: "afterChildren",
      },
    },
    open: {
      opacity: 1,
      height: "auto",
      transition: {
        duration: 0.3,
        ease: [0.4, 0, 0.2, 1],
        when: "beforeChildren",
        staggerChildren: 0.05,
      },
    },
  };

  const itemVariants = {
    closed: { opacity: 0, y: -10 },
    open: { opacity: 1, y: 0 },
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-[1000] transition-all duration-300 ${
        isScrolled
          ? 'bg-[#0a0a0f]/95 backdrop-blur-xl border-b border-white/10 shadow-lg shadow-black/30'
          : 'bg-[#0a0a0f]/70 backdrop-blur-md border-b border-white/5'
      }`}
    >
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-20">
          {/* Logo */}
          <a href="/" className="flex-shrink-0 flex items-center gap-2">
            <motion.span
              className="text-xl lg:text-2xl font-bold bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 bg-clip-text text-transparent"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              USDCOP
            </motion.span>
          </a>

          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center gap-8">
            {navLinks.map((link) => (
              <NavLink key={link.href} href={link.href}>
                {link.label}
              </NavLink>
            ))}
          </div>

          {/* Desktop Right Section */}
          <div className="hidden lg:flex items-center gap-4">
            {/* Language Toggle */}
            <div className="relative">
              <button
                onClick={() => setIsLangOpen(!isLangOpen)}
                className="flex items-center gap-1.5 px-3 py-2 text-gray-400 hover:text-white transition-colors duration-200 text-sm font-medium rounded-lg hover:bg-white/5"
              >
                <Globe className="w-4 h-4" />
                <span className="uppercase">{language}</span>
                <ChevronDown
                  className={`w-3 h-3 transition-transform duration-200 ${
                    isLangOpen ? 'rotate-180' : ''
                  }`}
                />
              </button>

              <AnimatePresence>
                {isLangOpen && (
                  <motion.div
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    transition={{ duration: 0.15 }}
                    className="absolute right-0 mt-2 w-32 bg-[#0a0a0f]/95 backdrop-blur-xl border border-white/10 rounded-lg shadow-xl overflow-hidden"
                  >
                    <button
                      onClick={() => handleLanguageChange('en')}
                      className={`w-full px-4 py-2.5 text-left text-sm transition-colors duration-200 ${
                        language === 'en'
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : 'text-gray-300 hover:bg-white/5 hover:text-white'
                      }`}
                    >
                      English
                    </button>
                    <button
                      onClick={() => handleLanguageChange('es')}
                      className={`w-full px-4 py-2.5 text-left text-sm transition-colors duration-200 ${
                        language === 'es'
                          ? 'bg-emerald-500/20 text-emerald-400'
                          : 'text-gray-300 hover:bg-white/5 hover:text-white'
                      }`}
                    >
                      Espanol
                    </button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* Login Button (Ghost) */}
            <a
              href="/login"
              className="px-4 py-2 text-gray-300 hover:text-white transition-colors duration-200 text-sm font-medium rounded-lg hover:bg-white/5"
            >
              {t.nav.login}
            </a>

            {/* Get Started Button (Gradient) */}
            <motion.a
              href="/register"
              className="relative px-5 py-2.5 text-sm font-semibold text-white rounded-lg overflow-hidden group"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <span className="absolute inset-0 bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500" />
              <span className="absolute inset-0 bg-gradient-to-r from-emerald-400 via-teal-400 to-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <span className="relative">{t.nav.request}</span>
            </motion.a>
          </div>

          {/* Mobile Menu Button */}
          <div className="flex lg:hidden items-center gap-3">
            {/* Mobile Language Toggle */}
            <button
              onClick={() => setLanguage(language === 'en' ? 'es' : 'en')}
              className="flex items-center gap-1 px-2 py-1.5 text-gray-400 hover:text-white transition-colors duration-200 text-sm font-medium rounded-lg hover:bg-white/5"
            >
              <Globe className="w-4 h-4" />
              <span className="uppercase text-xs">{language}</span>
            </button>

            <button
              onClick={toggleMenu}
              className="p-2 text-gray-400 hover:text-white transition-colors duration-200 rounded-lg hover:bg-white/5"
              aria-label={isOpen ? 'Close menu' : 'Open menu'}
              aria-expanded={isOpen}
            >
              <AnimatePresence mode="wait">
                {isOpen ? (
                  <motion.div
                    key="close"
                    initial={{ opacity: 0, rotate: -90 }}
                    animate={{ opacity: 1, rotate: 0 }}
                    exit={{ opacity: 0, rotate: 90 }}
                    transition={{ duration: 0.2 }}
                  >
                    <X className="w-6 h-6" />
                  </motion.div>
                ) : (
                  <motion.div
                    key="menu"
                    initial={{ opacity: 0, rotate: 90 }}
                    animate={{ opacity: 1, rotate: 0 }}
                    exit={{ opacity: 0, rotate: -90 }}
                    transition={{ duration: 0.2 }}
                  >
                    <Menu className="w-6 h-6" />
                  </motion.div>
                )}
              </AnimatePresence>
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        <AnimatePresence>
          {isOpen && (
            <motion.div
              variants={menuVariants}
              initial="closed"
              animate="open"
              exit="closed"
              className="lg:hidden overflow-hidden"
            >
              <div className="py-4 space-y-1 border-t border-white/5">
                {navLinks.map((link) => (
                  <motion.div key={link.href} variants={itemVariants}>
                    <MobileNavLink href={link.href} onClick={closeMenu}>
                      {link.label}
                    </MobileNavLink>
                  </motion.div>
                ))}

                {/* Mobile CTAs */}
                <motion.div
                  variants={itemVariants}
                  className="pt-4 mt-4 border-t border-white/5 space-y-3 px-4"
                >
                  <a
                    href="/login"
                    onClick={closeMenu}
                    className="block w-full py-3 text-center text-gray-300 hover:text-white transition-colors duration-200 text-base font-medium rounded-lg border border-white/10 hover:border-white/20 hover:bg-white/5"
                  >
                    {t.nav.login}
                  </a>

                  <a
                    href="/register"
                    onClick={closeMenu}
                    className="block w-full py-3 text-center text-white font-semibold rounded-lg bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500 hover:from-emerald-400 hover:via-teal-400 hover:to-cyan-400 transition-all duration-300"
                  >
                    {t.nav.request}
                  </a>
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </nav>

      {/* Mobile Menu Backdrop */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 top-16 bg-[#0a0a0f]/90 backdrop-blur-sm lg:hidden -z-10"
            onClick={closeMenu}
          />
        )}
      </AnimatePresence>
    </header>
  );
}
