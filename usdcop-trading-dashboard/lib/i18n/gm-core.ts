'use client';

/**
 * GM i18n core (CTR-GM-UI-001 · prototipo Var B bilingüe ES/EN).
 *
 * Patrón: cada área declara su diccionario tipado `{ es: {...}, en: {...} }` con
 * `defineGmDict()` y lo consume con `useGmT(dict)` — cero strings hardcodeados en
 * componentes GM. El idioma vive en LanguageContext (persistido en localStorage
 * `gm-lang`; default 'es'). Si el provider no está montado (páginas fuera del árbol),
 * cae a 'es' sin romper.
 */
import { useCallback, useSyncExternalStore } from 'react';

export type GmLang = 'es' | 'en';

const KEY = 'gm-lang';
const listeners = new Set<() => void>();

function readLang(): GmLang {
  if (typeof window === 'undefined') return 'es';
  return window.localStorage.getItem(KEY) === 'en' ? 'en' : 'es';
}

export function setGmLang(lang: GmLang): void {
  window.localStorage.setItem(KEY, lang);
  listeners.forEach((l) => l());
}

function subscribe(cb: () => void): () => void {
  listeners.add(cb);
  const onStorage = (e: StorageEvent) => { if (e.key === KEY) cb(); };
  window.addEventListener('storage', onStorage);
  return () => { listeners.delete(cb); window.removeEventListener('storage', onStorage); };
}

/** Idioma actual, reactivo (SSR-safe: servidor siempre 'es' para evitar hydration). */
export function useGmLang(): GmLang {
  return useSyncExternalStore(subscribe, readLang, () => 'es');
}

/** Declara un diccionario por área con paridad de claves ES/EN comprobada por tipos. */
export function defineGmDict<T extends Record<string, string>>(dict: { es: T; en: Record<keyof T, string> }) {
  return dict;
}

/** Hook de traducción: `const t = useGmT(DICT); t('title')`. */
export function useGmT<T extends Record<string, string>>(dict: { es: T; en: Record<keyof T, string> }) {
  const lang = useGmLang();
  return useCallback(<K extends keyof T>(key: K): string => dict[lang][key] ?? dict.es[key], [dict, lang]);
}
