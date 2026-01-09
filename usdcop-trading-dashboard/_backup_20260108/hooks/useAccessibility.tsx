"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { toast } from "react-hot-toast";

export interface AccessibilitySettings {
  screenReaderMode: boolean;
  highContrastMode: boolean;
  reducedMotion: boolean;
  focusIndicators: boolean;
  keyboardNavigation: boolean;
  fontSize: "small" | "medium" | "large" | "extra-large";
  colorBlindSupport: "none" | "protanopia" | "deuteranopia" | "tritanopia";
  announcements: boolean;
  skipLinks: boolean;
}

interface UseAccessibilityProps {
  onSettingsChange?: (settings: AccessibilitySettings) => void;
  autoDetect?: boolean;
}

export const useAccessibility = ({
  onSettingsChange,
  autoDetect = true,
}: UseAccessibilityProps = {}) => {
  const [settings, setSettings] = useState<AccessibilitySettings>({
    screenReaderMode: false,
    highContrastMode: false,
    reducedMotion: false,
    focusIndicators: true,
    keyboardNavigation: true,
    fontSize: "medium",
    colorBlindSupport: "none",
    announcements: true,
    skipLinks: true,
  });

  const [focusVisible, setFocusVisible] = useState(false);
  const [announceQueue, setAnnounceQueue] = useState<string[]>([]);
  const [currentFocus, setCurrentFocus] = useState<HTMLElement | null>(null);

  const announceTimeoutRef = useRef<NodeJS.Timeout>();
  const focusRingRef = useRef<HTMLElement | null>(null);
  const skipLinkRef = useRef<HTMLAnchorElement | null>(null);

  // Auto-detect user preferences
  useEffect(() => {
    if (!autoDetect || typeof window === "undefined") return;

    const detectPreferences = () => {
      const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      const prefersHighContrast = window.matchMedia("(prefers-contrast: high)").matches;
      const prefersLargeFonts = window.matchMedia("(prefers-reduced-data: reduce)").matches;

      setSettings(prev => ({
        ...prev,
        reducedMotion: prefersReducedMotion,
        highContrastMode: prefersHighContrast,
        fontSize: prefersLargeFonts ? "large" : prev.fontSize,
      }));
    };

    detectPreferences();

    // Listen for preference changes
    const mediaQueries = [
      window.matchMedia("(prefers-reduced-motion: reduce)"),
      window.matchMedia("(prefers-contrast: high)"),
      window.matchMedia("(prefers-reduced-data: reduce)"),
    ];

    mediaQueries.forEach(mq => mq.addEventListener("change", detectPreferences));

    return () => {
      mediaQueries.forEach(mq => mq.removeEventListener("change", detectPreferences));
    };
  }, [autoDetect]);

  // Load saved settings
  useEffect(() => {
    if (typeof window === "undefined") return;

    const saved = localStorage.getItem("accessibility-settings");
    if (saved) {
      try {
        const parsedSettings = JSON.parse(saved);
        setSettings(prev => ({ ...prev, ...parsedSettings }));
      } catch (error) {
        console.error("Failed to load accessibility settings:", error);
      }
    }
  }, []);

  // Save settings
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem("accessibility-settings", JSON.stringify(settings));
    }
    onSettingsChange?.(settings);
  }, [settings, onSettingsChange]);

  // Apply CSS custom properties for accessibility
  useEffect(() => {
    if (typeof window === "undefined") return;

    const root = document.documentElement;

    // Font size
    const fontSizeMap = {
      small: "14px",
      medium: "16px",
      large: "18px",
      "extra-large": "20px",
    };
    root.style.setProperty("--accessibility-font-size", fontSizeMap[settings.fontSize]);

    // High contrast
    if (settings.highContrastMode) {
      root.classList.add("high-contrast");
    } else {
      root.classList.remove("high-contrast");
    }

    // Reduced motion
    if (settings.reducedMotion) {
      root.classList.add("reduced-motion");
    } else {
      root.classList.remove("reduced-motion");
    }

    // Color blind support
    root.setAttribute("data-colorblind-filter", settings.colorBlindSupport);

    // Focus indicators
    if (settings.focusIndicators) {
      root.classList.add("show-focus-indicators");
    } else {
      root.classList.remove("show-focus-indicators");
    }
  }, [settings]);

  // Keyboard navigation management
  useEffect(() => {
    if (!settings.keyboardNavigation) return;

    let isUsingKeyboard = false;

    const handleKeyDown = (event: KeyboardEvent) => {
      isUsingKeyboard = true;
      setFocusVisible(true);

      // Handle keyboard navigation
      const { key, target, shiftKey, ctrlKey, altKey } = event;
      const activeElement = document.activeElement as HTMLElement;

      // Skip link activation
      if (key === "Tab" && !shiftKey && activeElement === document.body) {
        skipLinkRef.current?.focus();
        return;
      }

      // Roving tabindex for complex widgets
      if (key === "ArrowDown" || key === "ArrowUp" || key === "ArrowLeft" || key === "ArrowRight") {
        const focusableElements = getFocusableElements();
        const currentIndex = focusableElements.indexOf(activeElement);

        if (currentIndex !== -1) {
          let nextIndex;

          switch (key) {
            case "ArrowDown":
            case "ArrowRight":
              nextIndex = (currentIndex + 1) % focusableElements.length;
              break;
            case "ArrowUp":
            case "ArrowLeft":
              nextIndex = currentIndex === 0 ? focusableElements.length - 1 : currentIndex - 1;
              break;
            default:
              nextIndex = currentIndex;
          }

          focusableElements[nextIndex]?.focus();
          event.preventDefault();
        }
      }

      // Escape key handling
      if (key === "Escape") {
        const modal = document.querySelector('[role="dialog"], [role="alertdialog"]');
        if (modal) {
          const closeButton = modal.querySelector('[aria-label="Close"], [data-dismiss]') as HTMLElement;
          closeButton?.click();
        }
      }
    };

    const handleMouseDown = () => {
      isUsingKeyboard = false;
      setFocusVisible(false);
    };

    const handleFocus = (event: FocusEvent) => {
      setCurrentFocus(event.target as HTMLElement);

      if (isUsingKeyboard) {
        setFocusVisible(true);
        showFocusRing(event.target as HTMLElement);
      }
    };

    const handleBlur = () => {
      setCurrentFocus(null);
      hideFocusRing();
    };

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("mousedown", handleMouseDown);
    document.addEventListener("focus", handleFocus, true);
    document.addEventListener("blur", handleBlur, true);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("mousedown", handleMouseDown);
      document.removeEventListener("focus", handleFocus, true);
      document.removeEventListener("blur", handleBlur, true);
    };
  }, [settings.keyboardNavigation]);

  // Screen reader announcements
  useEffect(() => {
    if (!settings.announcements || announceQueue.length === 0) return;

    if (announceTimeoutRef.current) {
      clearTimeout(announceTimeoutRef.current);
    }

    announceTimeoutRef.current = setTimeout(() => {
      const message = announceQueue[0];
      announce(message);
      setAnnounceQueue(prev => prev.slice(1));
    }, 100);

    return () => {
      if (announceTimeoutRef.current) {
        clearTimeout(announceTimeoutRef.current);
      }
    };
  }, [announceQueue, settings.announcements]);

  // Get focusable elements
  const getFocusableElements = useCallback((): HTMLElement[] => {
    const selector = [
      'button:not([disabled])',
      'input:not([disabled])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      'a[href]',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable]',
      'summary',
    ].join(', ');

    return Array.from(document.querySelectorAll(selector)) as HTMLElement[];
  }, []);

  // Show focus ring
  const showFocusRing = useCallback((element: HTMLElement) => {
    if (!settings.focusIndicators) return;

    const rect = element.getBoundingClientRect();

    if (!focusRingRef.current) {
      focusRingRef.current = document.createElement("div");
      focusRingRef.current.className = "accessibility-focus-ring";
      document.body.appendChild(focusRingRef.current);
    }

    const ring = focusRingRef.current;
    ring.style.position = "fixed";
    ring.style.left = `${rect.left - 2}px`;
    ring.style.top = `${rect.top - 2}px`;
    ring.style.width = `${rect.width + 4}px`;
    ring.style.height = `${rect.height + 4}px`;
    ring.style.border = "2px solid #0066cc";
    ring.style.borderRadius = "4px";
    ring.style.pointerEvents = "none";
    ring.style.zIndex = "10000";
    ring.style.display = "block";
  }, [settings.focusIndicators]);

  // Hide focus ring
  const hideFocusRing = useCallback(() => {
    if (focusRingRef.current) {
      focusRingRef.current.style.display = "none";
    }
  }, []);

  // Announce to screen readers
  const announce = useCallback((message: string, priority: "polite" | "assertive" = "polite") => {
    if (!settings.announcements) return;

    const announcer = document.createElement("div");
    announcer.setAttribute("aria-live", priority);
    announcer.setAttribute("aria-atomic", "true");
    announcer.style.position = "absolute";
    announcer.style.left = "-10000px";
    announcer.style.width = "1px";
    announcer.style.height = "1px";
    announcer.style.overflow = "hidden";

    document.body.appendChild(announcer);
    announcer.textContent = message;

    setTimeout(() => {
      document.body.removeChild(announcer);
    }, 1000);
  }, [settings.announcements]);

  // Queue announcement
  const queueAnnouncement = useCallback((message: string) => {
    if (!settings.announcements) return;

    setAnnounceQueue(prev => [...prev, message]);
  }, [settings.announcements]);

  // Update setting
  const updateSetting = useCallback(<K extends keyof AccessibilitySettings>(
    key: K,
    value: AccessibilitySettings[K]
  ) => {
    setSettings(prev => ({ ...prev, [key]: value }));

    const settingNames = {
      screenReaderMode: "Screen Reader Mode",
      highContrastMode: "High Contrast Mode",
      reducedMotion: "Reduced Motion",
      focusIndicators: "Focus Indicators",
      keyboardNavigation: "Keyboard Navigation",
      fontSize: "Font Size",
      colorBlindSupport: "Color Blind Support",
      announcements: "Announcements",
      skipLinks: "Skip Links",
    };

    toast.success(`${settingNames[key]} ${value ? "enabled" : "updated"}`, {
      duration: 2000,
    });
  }, []);

  // Skip to main content
  const skipToMain = useCallback(() => {
    const main = document.querySelector("main") || document.querySelector('[role="main"]');
    if (main) {
      (main as HTMLElement).focus();
      main.scrollIntoView({ behavior: settings.reducedMotion ? "auto" : "smooth" });
    }
  }, [settings.reducedMotion]);

  // Check color contrast
  const checkContrast = useCallback((foreground: string, background: string): number => {
    const getRGB = (color: string) => {
      const canvas = document.createElement("canvas");
      canvas.width = canvas.height = 1;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = color;
      ctx.fillRect(0, 0, 1, 1);
      return ctx.getImageData(0, 0, 1, 1).data;
    };

    const getLuminance = (rgb: Uint8ClampedArray) => {
      const [r, g, b] = Array.from(rgb).map(c => {
        c = c / 255;
        return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
      });
      return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };

    const fg = getRGB(foreground);
    const bg = getRGB(background);

    const l1 = getLuminance(fg);
    const l2 = getLuminance(bg);

    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);

    return (lighter + 0.05) / (darker + 0.05);
  }, []);

  // Create skip link
  const createSkipLink = useCallback(() => {
    if (!settings.skipLinks) return null;

    return (
      <a
        ref={skipLinkRef}
        href="#main-content"
        className="accessibility-skip-link"
        onClick={(e) => {
          e.preventDefault();
          skipToMain();
        }}
        style={{
          position: "absolute",
          top: "-40px",
          left: "6px",
          background: "#000",
          color: "#fff",
          padding: "8px",
          textDecoration: "none",
          zIndex: 100000,
          borderRadius: "0 0 4px 4px",
          transform: "translateY(-100%)",
          transition: "transform 0.3s",
        }}
        onFocus={(e) => {
          e.target.style.transform = "translateY(0)";
        }}
        onBlur={(e) => {
          e.target.style.transform = "translateY(-100%)";
        }}
      >
        Skip to main content
      </a>
    );
  }, [settings.skipLinks, skipToMain]);

  return {
    // State
    settings,
    focusVisible,
    currentFocus,

    // Methods
    updateSetting,
    announce,
    queueAnnouncement,
    skipToMain,
    checkContrast,
    getFocusableElements,
    createSkipLink,

    // Utilities
    isScreenReaderMode: settings.screenReaderMode,
    isHighContrastMode: settings.highContrastMode,
    isReducedMotion: settings.reducedMotion,
    shouldShowFocusIndicators: settings.focusIndicators,
    shouldUseKeyboardNavigation: settings.keyboardNavigation,
  };
};